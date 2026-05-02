"""
Phase 1 — Model Discovery & Selection
======================================
Scans Hugging Face Hub for candidate models per category,
filters by quality signals, runs lightweight benchmark evals,
and returns a ranked shortlist ready for merging.

Usage:
    python -m phase1_discovery.discover --category code --top-k 5
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import typer
from huggingface_hub import HfApi, ModelFilter, ModelCard
from huggingface_hub.utils import RepositoryNotFoundError
from rich.console import Console
from rich.table import Table
from rich.progress import track

from configs.settings import (
    HF_MODEL_CATEGORIES, TOP_K_CANDIDATES,
    MIN_DOWNLOADS, MIN_LIKES, CFG, EVAL_DIR, HF_TOKEN
)
from utils.logger import logger

app     = typer.Typer(help="Phase 1: Model discovery & selection")
console = Console()
api     = HfApi(token=HF_TOKEN or None)


# ─────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────

@dataclass
class ModelCandidate:
    model_id:       str
    category:       str
    downloads:      int
    likes:          int
    params_b:       float          # billions
    pipeline_tag:   str
    tags:           list[str]
    score:          float = 0.0    # composite ranking score
    card_quality:   float = 0.0    # 0–1, presence of key card sections
    notes:          str = ""


# ─────────────────────────────────────────────
# Hub scanning
# ─────────────────────────────────────────────

def _estimate_params(model_info) -> float:
    """
    Estimate parameter count (billions) from safetensors metadata
    or fallback heuristics from the model card tags.
    """
    # Try safetensors index
    try:
        siblings = {s.rfilename for s in (model_info.siblings or [])}
        if "model.safetensors.index.json" in siblings:
            # Has sharded weights — large model, guess from tag
            pass
    except Exception:
        pass

    # Parse from tags: "7b", "7B", "mistral-7b" → 7.0
    for tag in (model_info.tags or []):
        tag = tag.lower()
        for suffix in ["b", "b-instruct", "b-chat"]:
            idx = tag.rfind(suffix)
            if idx > 0:
                candidate = tag[:idx].split("-")[-1]
                try:
                    return float(candidate)
                except ValueError:
                    pass
    # Fallback: parse from model_id
    mid = model_info.modelId.lower()
    for tok in mid.replace("-", " ").replace("_", " ").split():
        if tok.endswith("b"):
            try:
                return float(tok[:-1])
            except ValueError:
                pass
    return 0.0   # unknown


def _score_model_card(model_id: str) -> float:
    """Return 0–1 based on card completeness."""
    key_sections = ["## Model Details", "## Intended Uses", "## Training",
                    "## Evaluation", "## Limitations"]
    try:
        card = ModelCard.load(model_id, token=HF_TOKEN or None)
        text = card.content
        hits = sum(1 for s in key_sections if s.lower() in text.lower())
        return hits / len(key_sections)
    except Exception:
        return 0.0


def _composite_score(m: ModelCandidate) -> float:
    """
    Weighted composite score for ranking.
    Weights tuned for 7B-scale selection:
      - downloads (log-scaled)  : 40%
      - likes (log-scaled)      : 20%
      - card quality            : 20%
      - param fit (closer to 7B): 20%
    """
    import math
    dl_score    = math.log10(max(m.downloads, 1)) / 8        # normalize to ~0-1
    like_score  = math.log10(max(m.likes, 1)) / 5
    card_score  = m.card_quality
    param_score = max(0.0, 1.0 - abs(m.params_b - 7.0) / 7.0) if m.params_b else 0.3
    return 0.4 * dl_score + 0.2 * like_score + 0.2 * card_score + 0.2 * param_score


def scan_hub(
    category: str,
    keywords: list[str],
    max_results: int = 50,
    max_params_b: float = 10.0,
) -> list[ModelCandidate]:
    """Query HF Hub and return filtered ModelCandidate list."""
    candidates: list[ModelCandidate] = []
    seen: set[str] = set()

    for keyword in keywords:
        logger.info(f"[Discovery] Searching: '{keyword}' (category={category})")
        try:
            results = api.list_models(
                search=keyword,
                filter=ModelFilter(task="text-generation"),
                sort="downloads",
                direction=-1,
                limit=max_results,
                cardData=True,
                fetch_config=True,
            )
        except Exception as e:
            logger.warning(f"Hub query failed for '{keyword}': {e}")
            continue

        for info in results:
            mid = info.modelId
            if mid in seen:
                continue
            seen.add(mid)

            # Basic quality gates
            dl    = info.downloads or 0
            likes = info.likes or 0
            if dl < MIN_DOWNLOADS or likes < MIN_LIKES:
                continue

            params = _estimate_params(info)
            if params > max_params_b and params != 0.0:
                continue

            c = ModelCandidate(
                model_id     = mid,
                category     = category,
                downloads    = dl,
                likes        = likes,
                params_b     = params,
                pipeline_tag = info.pipeline_tag or "text-generation",
                tags         = list(info.tags or []),
            )
            candidates.append(c)

    logger.info(f"[Discovery] {len(candidates)} candidates before scoring")
    return candidates


# ─────────────────────────────────────────────
# Lightweight benchmark (perplexity proxy)
# ─────────────────────────────────────────────

PROBE_TEXTS = [
    "The transformer architecture consists of",
    "To implement a binary search tree in Python,",
    "The causes of World War II include",
    "Recent advances in large language models show",
]

def _quick_perplexity(model_id: str) -> Optional[float]:
    """
    Load model in 4-bit, compute mean perplexity on probe texts.
    Returns None on OOM / load failure.
    Lower perplexity → better language model.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tok = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN or None, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_cfg,
            device_map="auto",
            token=HF_TOKEN or None,
            trust_remote_code=True,
        )
        model.eval()

        import math
        ppls = []
        with torch.no_grad():
            for text in PROBE_TEXTS:
                ids = tok(text, return_tensors="pt").input_ids.to(model.device)
                loss = model(ids, labels=ids).loss
                ppls.append(math.exp(loss.item()))

        del model
        torch.cuda.empty_cache()
        return sum(ppls) / len(ppls)

    except Exception as e:
        logger.warning(f"Perplexity probe failed for {model_id}: {e}")
        return None


# ─────────────────────────────────────────────
# Main discovery pipeline
# ─────────────────────────────────────────────

def discover(
    category: str,
    top_k: int = TOP_K_CANDIDATES,
    run_perplexity: bool = False,
    save: bool = True,
) -> list[ModelCandidate]:
    """
    Full discovery pipeline for one category.
    Returns ranked list of top_k ModelCandidates.
    """
    keywords = HF_MODEL_CATEGORIES.get(category)
    if not keywords:
        raise ValueError(f"Unknown category '{category}'. Choose from: {list(HF_MODEL_CATEGORIES)}")

    # 1. Scan Hub
    candidates = scan_hub(category, keywords, max_params_b=CFG["max_model_params_b"])
    if not candidates:
        logger.warning(f"No candidates found for '{category}'")
        return []

    # 2. Score model cards (parallelizable but kept sequential for simplicity)
    logger.info("[Discovery] Scoring model cards...")
    for c in track(candidates, description="Scoring cards"):
        c.card_quality = _score_model_card(c.model_id)
        c.score        = _composite_score(c)
        time.sleep(0.1)   # gentle rate-limiting

    # 3. Optional: lightweight perplexity probe on top 2×top_k
    if run_perplexity:
        pre_ranked = sorted(candidates, key=lambda x: x.score, reverse=True)[:top_k * 2]
        logger.info(f"[Discovery] Running perplexity probes on {len(pre_ranked)} models...")
        for c in pre_ranked:
            ppl = _quick_perplexity(c.model_id)
            if ppl is not None:
                # Adjust score: lower PPL is better, scale inversely
                ppl_bonus = max(0, 1.0 - (ppl - 5) / 100)   # PPL 5→bonus 1, PPL 105→0
                c.score   = 0.7 * c.score + 0.3 * ppl_bonus
                c.notes   = f"ppl={ppl:.1f}"

    # 4. Final ranking
    ranked = sorted(candidates, key=lambda x: x.score, reverse=True)[:top_k]

    # 5. Display
    _print_table(ranked, category)

    # 6. Persist
    if save:
        out = EVAL_DIR / f"discovery_{category}.json"
        with open(out, "w") as f:
            json.dump([asdict(c) for c in ranked], f, indent=2)
        logger.info(f"[Discovery] Results saved → {out}")

    return ranked


def _print_table(candidates: list[ModelCandidate], category: str) -> None:
    table = Table(title=f"Top candidates — {category}", show_lines=True)
    table.add_column("Rank", style="dim", width=5)
    table.add_column("Model ID", style="cyan")
    table.add_column("Params (B)", justify="right")
    table.add_column("Downloads", justify="right")
    table.add_column("Likes", justify="right")
    table.add_column("Card", justify="right")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Notes")

    for i, c in enumerate(candidates, 1):
        table.add_row(
            str(i),
            c.model_id,
            f"{c.params_b:.1f}" if c.params_b else "?",
            f"{c.downloads:,}",
            str(c.likes),
            f"{c.card_quality:.2f}",
            f"{c.score:.3f}",
            c.notes,
        )
    console.print(table)


def discover_all(top_k: int = TOP_K_CANDIDATES, run_perplexity: bool = False) -> dict[str, list[ModelCandidate]]:
    """Discover models for every configured category."""
    results = {}
    for cat in HF_MODEL_CATEGORIES:
        logger.info(f"\n{'='*60}\nDiscovering: {cat}\n{'='*60}")
        results[cat] = discover(cat, top_k=top_k, run_perplexity=run_perplexity)
    return results


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

@app.command()
def run(
    category:        str  = typer.Argument(..., help=f"Category: {list(HF_MODEL_CATEGORIES.keys())}"),
    top_k:           int  = typer.Option(TOP_K_CANDIDATES, help="Models to return"),
    perplexity:      bool = typer.Option(False, "--perplexity/--no-perplexity", help="Run quick perplexity probe"),
    all_categories:  bool = typer.Option(False, "--all", help="Run for all categories"),
):
    if all_categories:
        discover_all(top_k=top_k, run_perplexity=perplexity)
    else:
        discover(category, top_k=top_k, run_perplexity=perplexity)


if __name__ == "__main__":
    app()
