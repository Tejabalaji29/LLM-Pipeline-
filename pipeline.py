"""
Master Pipeline Orchestrator
==============================
Runs all 5 phases end-to-end or individually.

Usage:
    # Full pipeline for 'code' category
    python pipeline.py run --category code

    # Just discover + merge
    python pipeline.py run --category reasoning --skip-ft --skip-serve

    # Iterative improvement loop
    python pipeline.py run --category medical --loop --max-iter 3
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from configs.settings import (
    FT_BASE_MODEL, TOP_K_CANDIDATES, MERGES_DIR, ADAPTERS_DIR, HF_ORG
)
from utils.logger import logger

app     = typer.Typer(help="LLM Pipeline — full orchestrator")
console = Console()


def _banner(text: str):
    console.print(Rule(f"[bold cyan]{text}[/bold cyan]"))


@app.command()
def run(
    category:      str  = typer.Argument("reasoning", help="Model category to target"),
    top_k:         int  = typer.Option(TOP_K_CANDIDATES, help="Candidates per category"),
    strategy:      str  = typer.Option("ties",  help="Merge strategy: slerp|ties|dare_ties|task_arithmetic|breadcrumbs"),
    base_model:    str  = typer.Option(FT_BASE_MODEL, help="Base model for fine-tuning"),
    n_eval:        int  = typer.Option(100,  help="Eval samples"),
    n_syn:         int  = typer.Option(50,   help="Synthetic samples per gap"),
    ft_epochs:     int  = typer.Option(2,    help="Fine-tuning epochs"),
    loop:          bool = typer.Option(False, "--loop", help="Enable iterative improvement loop"),
    max_iter:      int  = typer.Option(3,    help="Max loop iterations"),
    skip_discover: bool = typer.Option(False, "--skip-discover"),
    skip_merge:    bool = typer.Option(False, "--skip-merge"),
    skip_eval:     bool = typer.Option(False, "--skip-eval"),
    skip_ft:       bool = typer.Option(False, "--skip-ft"),
    skip_serve:    bool = typer.Option(True,  "--skip-serve/--serve"),  # off by default
    deploy:        bool = typer.Option(False, "--deploy"),
    hf_repo:       str  = typer.Option("",   "--repo"),
    use_wandb:     bool = typer.Option(False, "--wandb"),
    use_mergekit:  bool = typer.Option(False, "--mergekit/--no-mergekit"),
):
    console.print(Panel(
        f"[bold]LLM Pipeline[/bold]\n"
        f"Category: [cyan]{category}[/cyan]  |  Strategy: [magenta]{strategy}[/magenta]  |  "
        f"Base: [green]{base_model.split('/')[-1]}[/green]",
        title="Starting",
    ))

    # ──────────────────────────────
    # Phase 1: Discovery
    # ──────────────────────────────
    candidates = []
    if not skip_discover:
        _banner("Phase 1 — Discovery")
        from phase1_discovery.discover import discover
        candidates = discover(category, top_k=top_k)
        if not candidates:
            logger.error("No candidates found. Exiting.")
            raise typer.Exit(1)
        console.print(f"[green]✓ Found {len(candidates)} candidates[/green]")

    # ──────────────────────────────
    # Phase 2: Merging
    # ──────────────────────────────
    merged_path: Optional[Path] = None
    if not skip_merge and candidates:
        _banner("Phase 2 — Merging")
        from phase2_merging.merge import merge_models
        model_ids    = [c.model_id for c in candidates[:3]]   # merge top-3
        merged_path  = merge_models(
            strategy     = strategy,
            models       = model_ids,
            base_model   = model_ids[0],
            use_mergekit = use_mergekit,
        )
        console.print(f"[green]✓ Merged → {merged_path}[/green]")

    eval_model = str(merged_path) if merged_path else base_model

    # ──────────────────────────────
    # Phase 3: Evaluation
    # ──────────────────────────────
    eval_result = None
    if not skip_eval:
        _banner("Phase 3 — Evaluation")
        from phase3_evaluation.evaluate import evaluate, load_squad
        samples     = load_squad(n_eval)
        eval_result = evaluate(eval_model, samples, category, run_judge=True)
        console.print(
            f"[green]✓ ROUGE-1: {eval_result.avg_rouge1:.3f} | "
            f"BERTScore: {eval_result.avg_bertscore:.3f} | "
            f"Judge: {eval_result.avg_judge_score:.1f}/10[/green]"
        )

    # ──────────────────────────────
    # Phase 4: Fine-tuning
    # ──────────────────────────────
    best_adapter: Optional[Path] = None
    if not skip_ft:
        _banner("Phase 4 — Fine-Tuning")
        from phase4_finetuning.finetune import improvement_loop, fine_tune, generate_synthetic_data, format_as_hf_dataset
        from phase3_evaluation.evaluate import load_squad

        if loop:
            # Full iterative loop
            best_adapter = improvement_loop(
                base_model_id   = eval_model,
                eval_samples_fn = lambda: load_squad(min(50, n_eval)),
                max_iterations  = max_iter,
                n_syn_per_gap   = n_syn,
                use_wandb       = use_wandb,
            )
        elif eval_result and eval_result.gap_categories:
            # One-shot: target detected gaps
            from configs.settings import ADAPTERS_DIR
            syn     = generate_synthetic_data(eval_result.gap_categories, n_per_gap=n_syn)
            dataset = format_as_hf_dataset(syn)
            adapter_dir = ADAPTERS_DIR / f"{category}_{strategy}"
            best_adapter = fine_tune(
                base_model_id = eval_model,
                dataset       = dataset,
                output_dir    = adapter_dir,
                run_name      = f"{category}-{strategy}",
                epochs        = ft_epochs,
                use_wandb     = use_wandb,
            )
        else:
            logger.info("[FT] No gaps detected, skipping fine-tuning")

        if best_adapter:
            console.print(f"[green]✓ Adapter → {best_adapter}[/green]")

    # ──────────────────────────────
    # Phase 5a: MLOps tracking
    # ──────────────────────────────
    _banner("Phase 5 — MLOps")
    if eval_result:
        from phase5_mlops.serve import ExperimentTracker, ExperimentMetrics
        metrics = ExperimentMetrics(
            run_name        = f"{category}-{strategy}",
            model_id        = eval_model,
            merge_strategy  = strategy,
            base_models     = [c.model_id for c in candidates] if candidates else [],
            avg_rouge1      = eval_result.avg_rouge1,
            avg_rouge2      = eval_result.avg_rouge2,
            avg_rougeL      = eval_result.avg_rougeL,
            avg_bertscore   = eval_result.avg_bertscore,
            avg_faithfulness = eval_result.avg_faithfulness,
            hallucination_rate = eval_result.hallucination_rate,
            avg_judge_score = eval_result.avg_judge_score,
            gap_categories  = eval_result.gap_categories,
        )
        tracker = ExperimentTracker(use_wandb=use_wandb, use_mlflow=True)
        tracker.log(metrics)
        console.print("[green]✓ Experiment tracked[/green]")

    # ──────────────────────────────
    # Phase 5b: HF Hub deploy
    # ──────────────────────────────
    if deploy:
        from phase5_mlops.serve import deploy_to_hub, ExperimentMetrics
        repo = hf_repo or f"{HF_ORG}/{category}-{strategy}-7b"
        deploy_to_hub(
            model_path = str(merged_path or base_model),
            repo_id    = repo,
            metrics    = metrics if eval_result else None,
        )

    # ──────────────────────────────
    # Phase 5c: vLLM server
    # ──────────────────────────────
    if not skip_serve:
        _banner("Phase 5 — Inference Server")
        final_model = str(best_adapter or merged_path or base_model)
        from phase5_mlops.serve import VLLMServer
        server = VLLMServer(final_model)
        server.start_api_server()

    # ──────────────────────────────
    # Summary
    # ──────────────────────────────
    _banner("Pipeline Complete")
    from phase5_mlops.serve import print_pipeline_summary
    print_pipeline_summary()


@app.command("leaderboard")
def leaderboard():
    """Print the evaluation leaderboard from saved results."""
    from phase5_mlops.serve import print_pipeline_summary
    print_pipeline_summary()


@app.command("introspect")
def introspect(model: str = typer.Argument(..., help="Model ID to introspect")):
    """Print DOM-tree-style architecture of a model."""
    from phase2_merging.merge import introspect_architecture, print_architecture_tree
    from transformers import AutoModelForCausalLM
    from configs.settings import HF_TOKEN
    logger.info(f"Loading {model}...")
    m    = AutoModelForCausalLM.from_pretrained(model, device_map="cpu", token=HF_TOKEN or None)
    root = introspect_architecture(m, model)
    print_architecture_tree(root)


if __name__ == "__main__":
    app()
