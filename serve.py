"""
Phase 5 — Inference Optimization & MLOps
==========================================
• vLLM high-throughput inference server
• MLflow / W&B experiment tracking
• HF Hub model deployment with model cards
• Benchmark comparison and reporting

Usage:
    # Start inference server
    python -m phase5_mlops.serve --model ./merged --port 8000

    # Track experiment
    python -m phase5_mlops.serve track --run-name my-merge --rouge 0.42 --judge 7.1

    # Deploy to HF Hub
    python -m phase5_mlops.serve deploy --model ./merged --repo my-org/my-model
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from configs.settings import (
    WANDB_PROJECT, MLFLOW_URI, HF_ORG, HF_TOKEN,
    VLLM_TENSOR_PARALLEL, VLLM_GPU_MEMORY_UTIL, VLLM_MAX_MODEL_LEN,
    EVAL_DIR
)
from utils.logger import logger

app     = typer.Typer(help="Phase 5: Inference & MLOps")
console = Console()


# ─────────────────────────────────────────────
# 1. vLLM Inference Server
# ─────────────────────────────────────────────

@dataclass
class GenerationRequest:
    prompt:        str
    max_new_tokens: int   = 256
    temperature:   float = 0.7
    top_p:         float = 0.95
    stop:          list[str] = None


@dataclass
class GenerationResponse:
    text:         str
    tokens_used:  int
    latency_ms:   float
    model_id:     str


class VLLMServer:
    """
    Wraps vLLM's AsyncLLMEngine for high-throughput offline + online inference.
    Supports: continuous batching, PagedAttention, tensor parallelism.
    """

    def __init__(
        self,
        model_id:          str,
        tensor_parallel:   int   = VLLM_TENSOR_PARALLEL,
        gpu_memory_util:   float = VLLM_GPU_MEMORY_UTIL,
        max_model_len:     int   = VLLM_MAX_MODEL_LEN,
        quantization:      Optional[str] = None,   # "awq", "gptq", "squeezellm", None
        dtype:             str   = "bfloat16",
    ):
        self.model_id = model_id
        self._engine  = None
        self._engine_args = dict(
            model              = model_id,
            tensor_parallel_size = tensor_parallel,
            gpu_memory_utilization = gpu_memory_util,
            max_model_len      = max_model_len,
            quantization       = quantization,
            dtype              = dtype,
            trust_remote_code  = True,
            tokenizer          = model_id,
        )

    def _load(self):
        if self._engine is None:
            from vllm import LLM, SamplingParams
            logger.info(f"[vLLM] Loading: {self.model_id}")
            self._llm    = LLM(**self._engine_args)
            self._SamplingParams = SamplingParams
            logger.success("[vLLM] Engine ready")

    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResponse]:
        """Batch inference — all requests processed together with continuous batching."""
        self._load()
        from vllm import SamplingParams

        prompts = [r.prompt for r in requests]
        params  = [
            SamplingParams(
                temperature  = r.temperature,
                top_p        = r.top_p,
                max_tokens   = r.max_new_tokens,
                stop         = r.stop or [],
            )
            for r in requests
        ]

        t0      = time.perf_counter()
        outputs = self._llm.generate(prompts, params)
        elapsed = (time.perf_counter() - t0) * 1000

        responses = []
        for out in outputs:
            text  = out.outputs[0].text
            toks  = len(out.outputs[0].token_ids)
            responses.append(GenerationResponse(
                text        = text,
                tokens_used = toks,
                latency_ms  = elapsed / len(outputs),
                model_id    = self.model_id,
            ))
        return responses

    def generate_one(self, prompt: str, **kwargs) -> GenerationResponse:
        return self.generate([GenerationRequest(prompt=prompt, **kwargs)])[0]

    def benchmark_throughput(self, n_requests: int = 50) -> dict:
        """Measure tokens/sec across n_requests synthetic requests."""
        prompts = [
            f"Explain the concept of {topic} in detail."
            for topic in (["machine learning", "quantum computing", "climate change",
                           "evolution", "blockchain"] * 10)[:n_requests]
        ]
        reqs = [GenerationRequest(p, max_new_tokens=128) for p in prompts]

        t0        = time.perf_counter()
        responses = self.generate(reqs)
        elapsed   = time.perf_counter() - t0

        total_toks = sum(r.tokens_used for r in responses)
        tps        = total_toks / elapsed

        result = {
            "n_requests":      n_requests,
            "total_tokens":    total_toks,
            "elapsed_s":       round(elapsed, 2),
            "tokens_per_sec":  round(tps, 1),
            "avg_latency_ms":  round(elapsed * 1000 / n_requests, 1),
        }
        logger.info(f"[vLLM] Throughput: {tps:.0f} tokens/s, avg latency: {result['avg_latency_ms']}ms")
        return result

    def start_api_server(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Launch vLLM's built-in OpenAI-compatible API server as a subprocess.
        Compatible with any OpenAI SDK client.
        """
        import subprocess, sys
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model",              self.model_id,
            "--host",               host,
            "--port",               str(port),
            "--tensor-parallel-size", str(self._engine_args["tensor_parallel_size"]),
            "--gpu-memory-utilization", str(self._engine_args["gpu_memory_utilization"]),
            "--max-model-len",      str(self._engine_args["max_model_len"]),
            "--dtype",              self._engine_args["dtype"],
            "--trust-remote-code",
        ]
        if HF_TOKEN:
            cmd += ["--token", HF_TOKEN]

        logger.info(f"[vLLM] Starting API server at http://{host}:{port}")
        logger.info(f"  → OpenAI-compatible: POST http://{host}:{port}/v1/chat/completions")
        subprocess.run(cmd)


# ─────────────────────────────────────────────
# 2. Experiment Tracking
# ─────────────────────────────────────────────

@dataclass
class ExperimentMetrics:
    run_name:           str
    model_id:           str
    merge_strategy:     str = ""
    base_models:        list[str] = None
    avg_rouge1:         float = 0.0
    avg_rouge2:         float = 0.0
    avg_rougeL:         float = 0.0
    avg_bertscore:      float = 0.0
    avg_faithfulness:   float = 0.0
    hallucination_rate: float = 0.0
    avg_judge_score:    float = 0.0
    tokens_per_sec:     float = 0.0
    gap_categories:     list[str] = None
    notes:              str = ""


class ExperimentTracker:
    """Unified tracker: logs to W&B + MLflow simultaneously."""

    def __init__(self, use_wandb: bool = True, use_mlflow: bool = True):
        self.use_wandb  = use_wandb
        self.use_mlflow = use_mlflow

    def log(self, metrics: ExperimentMetrics, artifacts: Optional[list[Path]] = None):
        if self.use_wandb:
            self._log_wandb(metrics, artifacts)
        if self.use_mlflow:
            self._log_mlflow(metrics, artifacts)

    def _log_wandb(self, m: ExperimentMetrics, artifacts: Optional[list[Path]]):
        try:
            import wandb
            run = wandb.init(
                project = WANDB_PROJECT,
                name    = m.run_name,
                config  = {
                    "model_id":      m.model_id,
                    "merge_strategy": m.merge_strategy,
                    "base_models":   m.base_models or [],
                },
            )
            run.log({
                "eval/rouge1":           m.avg_rouge1,
                "eval/rouge2":           m.avg_rouge2,
                "eval/rougeL":           m.avg_rougeL,
                "eval/bertscore":        m.avg_bertscore,
                "eval/faithfulness":     m.avg_faithfulness,
                "eval/hallucination_rate": m.hallucination_rate,
                "eval/judge_score":      m.avg_judge_score,
                "perf/tokens_per_sec":   m.tokens_per_sec,
            })
            if artifacts:
                for path in artifacts:
                    run.save(str(path))
            run.finish()
            logger.info(f"[W&B] Logged run: {m.run_name}")
        except Exception as e:
            logger.warning(f"W&B logging failed: {e}")

    def _log_mlflow(self, m: ExperimentMetrics, artifacts: Optional[list[Path]]):
        try:
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment(WANDB_PROJECT)
            with mlflow.start_run(run_name=m.run_name):
                mlflow.set_tags({
                    "model_id":       m.model_id,
                    "merge_strategy": m.merge_strategy,
                    "base_models":    str(m.base_models),
                    "gaps":           str(m.gap_categories),
                })
                mlflow.log_metrics({
                    "rouge1":           m.avg_rouge1,
                    "rouge2":           m.avg_rouge2,
                    "rougeL":           m.avg_rougeL,
                    "bertscore":        m.avg_bertscore,
                    "faithfulness":     m.avg_faithfulness,
                    "hallucination_rate": m.hallucination_rate,
                    "judge_score":      m.avg_judge_score,
                    "tokens_per_sec":   m.tokens_per_sec,
                })
                if artifacts:
                    for path in artifacts:
                        mlflow.log_artifact(str(path))
            logger.info(f"[MLflow] Logged run: {m.run_name}")
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")


# ─────────────────────────────────────────────
# 3. Model Card Generator
# ─────────────────────────────────────────────

MODEL_CARD_TEMPLATE = """\
---
language:
- en
tags:
- llm-pipeline
- merged-model
- peft
license: apache-2.0
pipeline_tag: text-generation
---

# {model_name}

Produced by the **LLM Pipeline** — automated discovery, merging, evaluation, and fine-tuning.

## Model Details

| Property       | Value |
|----------------|-------|
| Base models    | {base_models} |
| Merge strategy | `{merge_strategy}` |
| Fine-tuning    | LoRA/QLoRA (PEFT) |
| Scale          | 7B parameters |
| Precision      | bfloat16 |

## Evaluation Results

| Metric             | Score |
|--------------------|-------|
| ROUGE-1            | {rouge1:.3f} |
| ROUGE-2            | {rouge2:.3f} |
| ROUGE-L            | {rougeL:.3f} |
| BERTScore F1       | {bertscore:.3f} |
| Faithfulness       | {faithfulness:.3f} |
| Hallucination rate | {halluc:.1%} |
| Judge Score        | {judge:.1f}/10 |

## Training Details

### Knowledge Gaps Addressed
{gap_categories}

### Merge Configuration
- **Strategy**: {merge_strategy}
- **Density** (TIES/Breadcrumbs): 0.7
- **Alpha**: 0.5

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("{repo_id}", trust_remote_code=True)
tok   = AutoTokenizer.from_pretrained("{repo_id}")

inputs = tok("Your prompt here", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=256)
print(tok.decode(output[0], skip_special_tokens=True))
```

## Limitations
- Evaluated primarily on English text
- May hallucinate on out-of-training-distribution topics
- 4-bit quantization used during evaluation (not deployment)

## Citation
```
@misc{{llm-pipeline-2024,
  title={{Automated LLM Discovery, Merging, and Fine-Tuning Pipeline}},
  year={{2024}}
}}
```
"""


def generate_model_card(metrics: ExperimentMetrics, repo_id: str) -> str:
    model_name   = repo_id.split("/")[-1]
    base_models  = ", ".join(f"`{m}`" for m in (metrics.base_models or []))
    gap_cats     = "\n".join(f"- {g}" for g in (metrics.gap_categories or [])) or "- None detected"

    return MODEL_CARD_TEMPLATE.format(
        model_name      = model_name,
        base_models     = base_models,
        merge_strategy  = metrics.merge_strategy,
        rouge1          = metrics.avg_rouge1,
        rouge2          = metrics.avg_rouge2,
        rougeL          = metrics.avg_rougeL,
        bertscore       = metrics.avg_bertscore,
        faithfulness    = metrics.avg_faithfulness,
        halluc          = metrics.hallucination_rate,
        judge           = metrics.avg_judge_score,
        gap_categories  = gap_cats,
        repo_id         = repo_id,
    )


# ─────────────────────────────────────────────
# 4. HF Hub Deployment
# ─────────────────────────────────────────────

def deploy_to_hub(
    model_path:  str,
    repo_id:     str,
    metrics:     Optional[ExperimentMetrics] = None,
    private:     bool = False,
    commit_msg:  str  = "Upload merged model",
) -> str:
    """
    Push model + tokenizer + model card to HF Hub.
    Returns the repo URL.
    """
    from huggingface_hub import HfApi, create_repo

    if not HF_TOKEN:
        raise EnvironmentError("Set HF_TOKEN env var before deploying")

    api = HfApi(token=HF_TOKEN)

    # Create repo if needed
    url = create_repo(repo_id, token=HF_TOKEN, private=private, exist_ok=True)
    logger.info(f"[Hub] Repo: {url}")

    # Write model card
    if metrics:
        card_text = generate_model_card(metrics, repo_id)
        card_path = Path(model_path) / "README.md"
        with open(card_path, "w") as f:
            f.write(card_text)
        logger.info("[Hub] Model card written")

    # Upload
    logger.info(f"[Hub] Uploading {model_path} → {repo_id} ...")
    api.upload_folder(
        folder_path  = model_path,
        repo_id      = repo_id,
        commit_message = commit_msg,
        ignore_patterns = ["*.log", "__pycache__"],
    )
    logger.success(f"[Hub] Deployed: https://huggingface.co/{repo_id}")
    return f"https://huggingface.co/{repo_id}"


# ─────────────────────────────────────────────
# 5. Pipeline Summary Report
# ─────────────────────────────────────────────

def print_pipeline_summary(
    eval_dir: Path = EVAL_DIR,
    top_n:    int  = 10,
) -> None:
    """Load all saved eval JSONs and print a ranked leaderboard."""
    eval_files = sorted(eval_dir.glob("eval_*.json"))
    if not eval_files:
        console.print("[yellow]No evaluation results found.[/yellow]")
        return

    rows = []
    for ef in eval_files:
        try:
            with open(ef) as f:
                d = json.load(f)
            rows.append(d)
        except Exception:
            continue

    rows.sort(key=lambda x: x.get("avg_rouge1", 0), reverse=True)

    table = Table(title="Pipeline Leaderboard", show_lines=True)
    table.add_column("Rank",          width=5,  style="dim")
    table.add_column("Model",         style="cyan")
    table.add_column("Dataset",       style="dim")
    table.add_column("ROUGE-1",       justify="right", style="green")
    table.add_column("BERTScore",     justify="right")
    table.add_column("Faithfulness",  justify="right")
    table.add_column("Judge",         justify="right")
    table.add_column("Halluc %",      justify="right", style="red")

    for i, r in enumerate(rows[:top_n], 1):
        table.add_row(
            str(i),
            r.get("model_id", "")[-40:],
            r.get("dataset", ""),
            f"{r.get('avg_rouge1',0):.3f}",
            f"{r.get('avg_bertscore',0):.3f}",
            f"{r.get('avg_faithfulness',0):.3f}",
            f"{r.get('avg_judge_score',0):.1f}",
            f"{r.get('hallucination_rate',0)*100:.1f}%",
        )
    console.print(table)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

@app.command("serve")
def serve(
    model:  str  = typer.Argument(..., help="Model ID or local path"),
    port:   int  = typer.Option(8000),
    host:   str  = typer.Option("0.0.0.0"),
    bench:  bool = typer.Option(False, "--bench", help="Run throughput benchmark then exit"),
):
    server = VLLMServer(model)
    if bench:
        result = server.benchmark_throughput()
        console.print(Panel(str(result), title="Throughput Benchmark"))
        return
    server.start_api_server(host=host, port=port)


@app.command("track")
def track_cmd(
    run_name:    str   = typer.Argument(...),
    model:       str   = typer.Option(..., "--model"),
    strategy:    str   = typer.Option(""),
    rouge1:      float = typer.Option(0.0),
    bertscore:   float = typer.Option(0.0),
    judge:       float = typer.Option(0.0),
    tps:         float = typer.Option(0.0),
    wandb_flag:  bool  = typer.Option(True,  "--wandb/--no-wandb"),
    mlflow_flag: bool  = typer.Option(True,  "--mlflow/--no-mlflow"),
):
    metrics = ExperimentMetrics(
        run_name       = run_name,
        model_id       = model,
        merge_strategy = strategy,
        avg_rouge1     = rouge1,
        avg_bertscore  = bertscore,
        avg_judge_score = judge,
        tokens_per_sec = tps,
    )
    tracker = ExperimentTracker(use_wandb=wandb_flag, use_mlflow=mlflow_flag)
    tracker.log(metrics)
    console.print(f"[green]✓ Logged experiment: {run_name}[/green]")


@app.command("deploy")
def deploy_cmd(
    model:   str  = typer.Argument(..., help="Local model path"),
    repo:    str  = typer.Option(..., "--repo", help="HF repo: username/model-name"),
    private: bool = typer.Option(False),
):
    deploy_to_hub(model, repo, private=private)


@app.command("leaderboard")
def leaderboard():
    print_pipeline_summary()


if __name__ == "__main__":
    app()
