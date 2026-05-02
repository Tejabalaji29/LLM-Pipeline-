"""
Phase 2 — Model Composition & Merging
=======================================
Implements:
  • Architecture introspection  (DOM-tree-style layer parsing)
  • Union merging               (SLERP, TIES, DARE-TIES, Task Arithmetic)
  • Intersection merging        (Breadcrumbs / consensus filtering)
  • mergekit YAML generation    (for mergekit-based merges)
  • Direct torch merging        (for custom strategies)

Usage:
    python -m phase2_merging.merge --strategy ties --models a/model1 b/model2 --output ./merged
"""

from __future__ import annotations

import copy
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
import typer
from rich.console import Console
from rich.tree import Tree

from configs.settings import CFG, MERGES_DIR, HF_TOKEN
from utils.logger import logger

app     = typer.Typer(help="Phase 2: Model merging")
console = Console()


# ─────────────────────────────────────────────
# 1. Architecture Introspection
# ─────────────────────────────────────────────

@dataclass
class LayerNode:
    """One node in the architecture tree."""
    name:     str
    kind:     str          # "embedding", "attention", "mlp", "norm", "head", "other"
    shape:    tuple
    dtype:    str
    numel:    int
    children: list["LayerNode"] = field(default_factory=list)

    @property
    def params_m(self) -> float:
        return self.numel / 1e6

    def __repr__(self):
        return f"LayerNode({self.name}, {self.kind}, shape={self.shape}, {self.params_m:.2f}M)"


def _classify_layer(name: str) -> str:
    n = name.lower()
    if any(x in n for x in ["embed_tokens", "wte", "word_embeddings"]):
        return "embedding"
    if any(x in n for x in ["self_attn", "attention", "attn", "q_proj", "k_proj", "v_proj", "o_proj"]):
        return "attention"
    if any(x in n for x in ["mlp", "ffn", "feed_forward", "gate_proj", "up_proj", "down_proj", "fc1", "fc2"]):
        return "mlp"
    if any(x in n for x in ["norm", "layer_norm", "layernorm", "rmsnorm"]):
        return "norm"
    if any(x in n for x in ["lm_head", "embed_out", "output"]):
        return "head"
    return "other"


def introspect_architecture(model_or_state_dict, model_id: str = "") -> LayerNode:
    """
    Parse a model (or its state dict) into a DOM-tree-like LayerNode hierarchy.
    Works with nn.Module or a raw state dict.
    """
    if hasattr(model_or_state_dict, "named_parameters"):
        state = {n: p.data for n, p in model_or_state_dict.named_parameters()}
    else:
        state = model_or_state_dict   # already a state dict

    root     = LayerNode(model_id or "model", "root", (), "", sum(p.numel() for p in state.values()))
    # Group by top-level prefix (e.g. "model.layers.0", "model.layers.1", ...)
    groups: dict[str, list[tuple[str, torch.Tensor]]] = {}
    for full_name, tensor in state.items():
        top = full_name.split(".")[0]
        groups.setdefault(top, []).append((full_name, tensor))

    for top_key, params in sorted(groups.items()):
        total_numel = sum(t.numel() for _, t in params)
        group_node  = LayerNode(
            name   = top_key,
            kind   = _classify_layer(top_key),
            shape  = (),
            dtype  = "",
            numel  = total_numel,
        )
        for full_name, tensor in params:
            group_node.children.append(LayerNode(
                name  = full_name,
                kind  = _classify_layer(full_name),
                shape = tuple(tensor.shape),
                dtype = str(tensor.dtype).replace("torch.", ""),
                numel = tensor.numel(),
            ))
        root.children.append(group_node)

    return root


def print_architecture_tree(root: LayerNode, max_depth: int = 3) -> None:
    """Render the LayerNode tree with rich."""
    KIND_COLOR = {
        "embedding": "cyan", "attention": "magenta", "mlp": "green",
        "norm": "yellow", "head": "red", "other": "dim", "root": "bold white",
    }

    def _add(tree_node, layer: LayerNode, depth: int):
        if depth > max_depth:
            return
        color  = KIND_COLOR.get(layer.kind, "white")
        label  = (f"[{color}]{layer.name}[/{color}] "
                  f"[dim]{layer.kind}[/dim] "
                  f"[green]{layer.params_m:.2f}M[/green]")
        if layer.shape:
            label += f" [dim]{layer.shape}[/dim]"
        child_tree = tree_node.add(label)
        for child in layer.children:
            _add(child_tree, child, depth + 1)

    rich_tree = Tree(f"[bold]{root.name}[/bold] — {root.numel/1e9:.2f}B params")
    for child in root.children:
        _add(rich_tree, child, 1)
    console.print(rich_tree)


def get_layer_groups(root: LayerNode) -> dict[str, list[str]]:
    """Return {kind: [param_names]} mapping for targeted merging."""
    groups: dict[str, list[str]] = {}
    def _walk(node: LayerNode):
        if node.shape:   # leaf
            groups.setdefault(node.kind, []).append(node.name)
        for c in node.children:
            _walk(c)
    _walk(root)
    return groups


# ─────────────────────────────────────────────
# 2. mergekit YAML generation
# ─────────────────────────────────────────────

def _mergekit_yaml(strategy: str, models: list[str], weights: Optional[list[float]] = None,
                   density: float = 0.7, base_model: Optional[str] = None) -> dict:
    """Generate a mergekit-compatible merge config dict."""
    if weights is None:
        weights = [1.0 / len(models)] * len(models)

    model_specs = [{"model": m, "parameters": {"weight": w}} for m, w in zip(models, weights)]
    if base_model:
        model_specs[0]["model"] = base_model   # first slot = base for TIES/DARE

    cfg: dict[str, Any] = {
        "models": model_specs,
        "merge_method": strategy,
        "dtype": "bfloat16",
        "tokenizer_source": "union",
    }

    if strategy in ("ties", "dare_ties"):
        cfg["parameters"] = {"density": density, "normalize": True}
    elif strategy == "slerp":
        cfg["parameters"] = {"t": weights[1] if len(weights) > 1 else 0.5}
    elif strategy == "task_arithmetic":
        cfg["parameters"] = {"scaling_coefficient": 0.5}

    return cfg


def run_mergekit(
    strategy:    str,
    models:      list[str],
    output_dir:  Path,
    weights:     Optional[list[float]] = None,
    density:     float = 0.7,
    base_model:  Optional[str] = None,
) -> Path:
    """
    Write a mergekit YAML, then invoke mergekit-merge via subprocess.
    Requires: pip install mergekit
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_dict = _mergekit_yaml(strategy, models, weights, density, base_model)
    cfg_path = output_dir / "merge_config.yml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)
    logger.info(f"[Merge] mergekit config → {cfg_path}")

    cmd = [
        "mergekit-merge", str(cfg_path), str(output_dir),
        "--cuda", "--lazy-unpickle",
    ]
    if HF_TOKEN:
        cmd += ["--token", HF_TOKEN]

    logger.info(f"[Merge] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"mergekit failed:\n{result.stderr}")
        raise RuntimeError("mergekit-merge failed")
    logger.success(f"[Merge] Merged model saved → {output_dir}")
    return output_dir


# ─────────────────────────────────────────────
# 3. Direct torch merging (custom strategies)
# ─────────────────────────────────────────────

class TorchMerger:
    """
    Pure-PyTorch implementations of:
      • SLERP         — spherical linear interpolation
      • Task Arithmetic — delta-weight addition
      • TIES           — trim, elect sign, merge
      • Breadcrumbs    — intersection (consensus) merging
    All operate on loaded state dicts to avoid repeated model loading.
    """

    # ── helpers ──

    @staticmethod
    def _load_state(model_id_or_path: str) -> dict[str, torch.Tensor]:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        import torch
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            quantization_config=bnb,
            device_map="cpu",   # load to CPU for merging
            token=HF_TOKEN or None,
            trust_remote_code=True,
        )
        return {k: v.clone().float() for k, v in model.state_dict().items()}

    @staticmethod
    def _slerp_tensors(t0: torch.Tensor, t1: torch.Tensor, alpha: float) -> torch.Tensor:
        """SLERP between two flat parameter vectors."""
        orig_shape = t0.shape
        v0 = t0.flatten().double()
        v1 = t1.flatten().double()
        n0, n1 = v0.norm(), v1.norm()
        if n0 < 1e-8 or n1 < 1e-8:
            return (1 - alpha) * t0 + alpha * t1
        u0, u1 = v0 / n0, v1 / n1
        dot = torch.clamp((u0 * u1).sum(), -1.0, 1.0)
        theta = torch.acos(dot)
        if theta.abs() < 1e-6:
            return (1 - alpha) * t0 + alpha * t1
        sin_theta = torch.sin(theta)
        result = (torch.sin((1 - alpha) * theta) / sin_theta * v0
                + torch.sin(alpha * theta)       / sin_theta * v1)
        return result.float().reshape(orig_shape)

    # ── public merge methods ──

    def slerp(self, model_a: str, model_b: str, alpha: float = 0.5) -> dict[str, torch.Tensor]:
        """SLERP: interpolate every matching parameter tensor."""
        logger.info(f"[SLERP] Loading models...")
        sd_a = self._load_state(model_a)
        sd_b = self._load_state(model_b)
        merged = {}
        for key in sd_a:
            if key in sd_b and sd_a[key].shape == sd_b[key].shape:
                if sd_a[key].is_floating_point():
                    merged[key] = self._slerp_tensors(sd_a[key], sd_b[key], alpha)
                else:
                    merged[key] = sd_a[key]   # keep base for non-float (e.g. int8)
            else:
                merged[key] = sd_a[key]
        logger.success("[SLERP] Merge complete")
        return merged

    def task_arithmetic(
        self,
        base_model:  str,
        models:      list[str],
        scaling:     float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """
        Task Arithmetic: merged = base + Σ scaling * (model_i - base)
        Union of capabilities.
        """
        logger.info("[TaskArithmetic] Loading base...")
        base_sd = self._load_state(base_model)
        delta_sum: dict[str, torch.Tensor] = {}

        for mid in models:
            logger.info(f"[TaskArithmetic] Computing delta: {mid}")
            ft_sd = self._load_state(mid)
            for key in base_sd:
                if key in ft_sd and base_sd[key].shape == ft_sd[key].shape:
                    delta = ft_sd[key].float() - base_sd[key].float()
                    delta_sum[key] = delta_sum.get(key, torch.zeros_like(delta)) + delta

        merged = {}
        for key in base_sd:
            if key in delta_sum:
                merged[key] = base_sd[key].float() + scaling * delta_sum[key]
            else:
                merged[key] = base_sd[key]
        logger.success("[TaskArithmetic] Merge complete")
        return merged

    def ties(
        self,
        base_model:   str,
        models:       list[str],
        density:      float = 0.7,
        scaling:      float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """
        TIES (Trim, Elect Sign, Merge):
        1. Compute deltas from base.
        2. Trim lowest-magnitude changes per model (keep top `density` fraction).
        3. Elect dominant sign per parameter.
        4. Merge only parameters that agree with elected sign.
        """
        logger.info("[TIES] Loading base...")
        base_sd = self._load_state(base_model)
        all_deltas: list[dict[str, torch.Tensor]] = []

        for mid in models:
            logger.info(f"[TIES] Computing delta: {mid}")
            ft_sd = self._load_state(mid)
            delta = {}
            for key in base_sd:
                if key in ft_sd and base_sd[key].shape == ft_sd[key].shape:
                    d = ft_sd[key].float() - base_sd[key].float()
                    # Trim: zero out smallest (1 - density) fraction
                    flat    = d.abs().flatten()
                    k       = max(1, int(flat.numel() * density))
                    thresh  = flat.kthvalue(flat.numel() - k + 1).values
                    d[d.abs() < thresh] = 0.0
                    delta[key] = d
            all_deltas.append(delta)

        merged = {}
        for key in base_sd:
            stacked = torch.stack([
                d[key] for d in all_deltas if key in d
            ], dim=0)  # (n_models, *shape)

            # Elect sign by majority
            sign_sum = stacked.sign().sum(dim=0)
            elected  = sign_sum.sign()                 # +1 / -1 / 0
            elected[elected == 0] = 1                  # break ties

            # Mask: keep only params that agree with elected sign
            mask   = (stacked.sign() == elected.unsqueeze(0)).float()
            merged_delta = (stacked * mask).sum(dim=0) / (mask.sum(dim=0).clamp(min=1))
            merged[key]  = base_sd[key].float() + scaling * merged_delta

        logger.success("[TIES] Merge complete")
        return merged

    def breadcrumbs(
        self,
        base_model:  str,
        models:      list[str],
        density:     float = 0.7,
        epsilon:     float = 0.01,   # consensus threshold
    ) -> dict[str, torch.Tensor]:
        """
        Breadcrumbs (Intersection / Conservative merge):
        Only update parameters where ALL models agree on direction.
        Produces a safer, more conservative merged model.
        """
        logger.info("[Breadcrumbs] Loading base...")
        base_sd = self._load_state(base_model)
        all_deltas: list[dict[str, torch.Tensor]] = []

        for mid in models:
            logger.info(f"[Breadcrumbs] Computing delta: {mid}")
            ft_sd = self._load_state(mid)
            delta = {}
            for key in base_sd:
                if key in ft_sd and base_sd[key].shape == ft_sd[key].shape:
                    delta[key] = ft_sd[key].float() - base_sd[key].float()
            all_deltas.append(delta)

        merged = {}
        for key in base_sd:
            deltas_for_key = [d[key] for d in all_deltas if key in d]
            if not deltas_for_key:
                merged[key] = base_sd[key]
                continue

            stacked = torch.stack(deltas_for_key, dim=0)
            # Consensus: all models must agree on sign (intersection)
            signs   = stacked.sign()
            consensus = (signs == signs[0]).all(dim=0)   # agree with first model
            avg_delta = stacked.mean(dim=0)
            avg_delta[~consensus] = 0.0                  # zero out disagreements
            avg_delta[avg_delta.abs() < epsilon] = 0.0   # trim noise

            merged[key] = base_sd[key].float() + avg_delta

        logger.success("[Breadcrumbs] Merge complete")
        return merged

    def save(self, state_dict: dict[str, torch.Tensor], output_dir: Path,
             base_model_id: str) -> Path:
        """Save merged state dict as a HF model."""
        from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[Merge] Saving to {output_dir} ...")
        # Load skeleton (config + tokenizer) from base, apply our weights
        config = AutoConfig.from_pretrained(base_model_id, token=HF_TOKEN or None, trust_remote_code=True)
        tok    = AutoTokenizer.from_pretrained(base_model_id, token=HF_TOKEN or None, trust_remote_code=True)
        model  = AutoModelForCausalLM.from_config(config)
        # Convert back to bf16 for storage
        merged_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
        model.load_state_dict(merged_bf16, strict=False)
        model.save_pretrained(output_dir)
        tok.save_pretrained(output_dir)
        logger.success(f"[Merge] Model saved → {output_dir}")
        return output_dir


# ─────────────────────────────────────────────
# 4. High-level pipeline entry point
# ─────────────────────────────────────────────

def merge_models(
    strategy:   str,
    models:     list[str],
    output_dir: Optional[Path]  = None,
    base_model: Optional[str]   = None,
    density:    float           = 0.7,
    alpha:      float           = 0.5,
    use_mergekit: bool          = True,
) -> Path:
    """
    Unified entry point for all merge strategies.
    strategy ∈ {slerp, ties, dare_ties, task_arithmetic, breadcrumbs}

    use_mergekit=True  → generates YAML and calls mergekit-merge (recommended for prod)
    use_mergekit=False → uses TorchMerger (no mergekit dependency needed)
    """
    tag        = "_".join(m.split("/")[-1] for m in models[:2])
    out        = output_dir or (MERGES_DIR / f"{strategy}_{tag}")

    if use_mergekit and strategy in ("slerp", "ties", "dare_ties", "task_arithmetic"):
        return run_mergekit(strategy, models, out, density=density, base_model=base_model)

    # Torch-native path
    merger = TorchMerger()
    base   = base_model or models[0]

    if strategy == "slerp":
        sd = merger.slerp(models[0], models[1], alpha=alpha)
    elif strategy == "task_arithmetic":
        sd = merger.task_arithmetic(base, models[1:], scaling=alpha)
    elif strategy == "ties":
        sd = merger.ties(base, models[1:], density=density, scaling=alpha)
    elif strategy == "breadcrumbs":
        sd = merger.breadcrumbs(base, models[1:], density=density)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return merger.save(sd, out, base)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

@app.command()
def run(
    strategy:     str        = typer.Argument(..., help="slerp|ties|dare_ties|task_arithmetic|breadcrumbs"),
    models:       list[str]  = typer.Option(..., "--model", "-m", help="Model IDs (repeat flag)"),
    output:       Path       = typer.Option(None, "--output", "-o"),
    base:         str        = typer.Option(None, "--base", "-b", help="Base model for delta strategies"),
    density:      float      = typer.Option(0.7, help="TIES/Breadcrumbs density"),
    alpha:        float      = typer.Option(0.5, help="SLERP/TaskArith interpolation weight"),
    torch_only:   bool       = typer.Option(False, "--torch-only", help="Skip mergekit, use TorchMerger"),
    introspect:   str        = typer.Option(None, "--introspect", help="Print architecture tree for a model ID"),
):
    if introspect:
        from transformers import AutoModelForCausalLM
        logger.info(f"Loading {introspect} for introspection...")
        m    = AutoModelForCausalLM.from_pretrained(introspect, device_map="cpu", token=HF_TOKEN or None)
        root = introspect_architecture(m, introspect)
        print_architecture_tree(root)
        return

    out = merge_models(
        strategy=strategy, models=models, output_dir=output,
        base_model=base, density=density, alpha=alpha,
        use_mergekit=not torch_only,
    )
    console.print(f"[green]✓ Merged model ready at: {out}[/green]")


if __name__ == "__main__":
    app()
