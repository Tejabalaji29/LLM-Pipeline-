"""
Phase 4 — Efficient Fine-Tuning + Synthetic Data Generation
=============================================================
• Synthetic data generation from gap categories (LLM-driven)
• LoRA / QLoRA fine-tuning using PEFT + TRL SFTTrainer
• Delta adapter extraction (merge-ready LoRA weights)
• Iterative improvement loop: eval → gap detect → generate → fine-tune → re-eval

Usage:
    python -m phase4_finetuning.finetune --base mistralai/Mistral-7B-v0.3 --gaps factual_recall
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import typer
from rich.console import Console
from rich.progress import track

from configs.settings import (
    FT_BASE_MODEL, FT_EPOCHS, FT_LR, FT_WARMUP_RATIO, FT_SAVE_STEPS,
    CFG, ADAPTERS_DIR, DATA_DIR, HF_TOKEN
)
from utils.logger import logger

app     = typer.Typer(help="Phase 4: Fine-tuning & synthetic data generation")
console = Console()


# ─────────────────────────────────────────────
# 1. Synthetic Data Generation
# ─────────────────────────────────────────────

GAP_PROMPTS: dict[str, str] = {
    "factual_recall": """Generate {n} high-quality QA pairs testing factual recall.
Format each as JSON: {{"context": "...", "question": "...", "answer": "..."}}
Focus on: historical dates, scientific facts, geography, key figures.
Return a JSON array only.""",

    "multi_step_reasoning": """Generate {n} QA pairs requiring multi-step reasoning.
Format each as JSON: {{"context": "...", "question": "...", "answer": "..."}}
Each answer must show intermediate reasoning steps.
Return a JSON array only.""",

    "numerical": """Generate {n} QA pairs involving numerical calculations or statistics.
Format each as JSON: {{"context": "...", "question": "...", "answer": "..."}}
Include percentages, comparisons, and mathematical relationships.
Return a JSON array only.""",

    "code_generation": """Generate {n} coding QA pairs.
Format each as JSON: {{"context": "function specification", "question": "implementation task", "answer": "working code"}}
Cover: Python functions, algorithms, data structures.
Return a JSON array only.""",

    "summarization": """Generate {n} summarization QA pairs.
Format each as JSON: {{"context": "long passage", "question": "summarize this", "answer": "concise summary"}}
Vary context length 200-800 words.
Return a JSON array only.""",
}


@dataclass
class SyntheticSample:
    context:  str
    question: str
    answer:   str
    gap_cat:  str


def _parse_json_array(text: str) -> list[dict]:
    """Extract first valid JSON array from LLM output."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find JSON array with regex
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []


def generate_synthetic_data(
    gap_categories: list[str],
    n_per_gap:      int = 50,
    generator_model: str = FT_BASE_MODEL,
) -> list[SyntheticSample]:
    """
    Use a capable LLM to generate targeted synthetic training data
    for each detected knowledge gap.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    logger.info(f"[SynData] Generating data for gaps: {gap_categories}")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(generator_model, token=HF_TOKEN or None, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        generator_model,
        quantization_config=bnb,
        device_map="auto",
        token=HF_TOKEN or None,
        trust_remote_code=True,
    )
    model.eval()

    all_samples: list[SyntheticSample] = []

    for gap in gap_categories:
        prompt_template = GAP_PROMPTS.get(gap, GAP_PROMPTS["factual_recall"])
        prompt = prompt_template.format(n=n_per_gap)

        logger.info(f"[SynData] Generating {n_per_gap} samples for: {gap}")
        inputs  = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
            )
        new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
        raw     = tok.decode(new_ids, skip_special_tokens=True)
        items   = _parse_json_array(raw)

        for item in items:
            try:
                all_samples.append(SyntheticSample(
                    context  = str(item.get("context", "")),
                    question = str(item.get("question", "")),
                    answer   = str(item.get("answer", "")),
                    gap_cat  = gap,
                ))
            except Exception:
                continue

        logger.info(f"[SynData] Got {len(items)} valid samples for '{gap}'")

    del model
    torch.cuda.empty_cache()

    # Save to disk
    out_path = DATA_DIR / "synthetic_data.jsonl"
    with open(out_path, "w") as f:
        for s in all_samples:
            f.write(json.dumps({"context": s.context, "question": s.question,
                                "answer": s.answer, "gap_cat": s.gap_cat}) + "\n")
    logger.success(f"[SynData] {len(all_samples)} samples saved → {out_path}")
    return all_samples


# ─────────────────────────────────────────────
# 2. Dataset formatting
# ─────────────────────────────────────────────

CHAT_TEMPLATE = """<s>[INST] Context: {context}

Question: {question} [/INST] {answer}</s>"""


def format_as_hf_dataset(samples: list[SyntheticSample]):
    """Convert SyntheticSample list → HF Dataset with text column."""
    from datasets import Dataset
    rows = [{
        "text": CHAT_TEMPLATE.format(
            context=s.context, question=s.question, answer=s.answer
        ),
        "gap_cat": s.gap_cat,
    } for s in samples if s.context and s.question and s.answer]
    return Dataset.from_list(rows)


def load_jsonl_dataset(path: str):
    """Load saved synthetic JSONL as HF Dataset."""
    from datasets import load_dataset
    return load_dataset("json", data_files=path, split="train")


# ─────────────────────────────────────────────
# 3. LoRA / QLoRA configuration
# ─────────────────────────────────────────────

def build_lora_config(
    r:           int   = CFG["lora_r"],
    alpha:       int   = CFG["lora_alpha"],
    dropout:     float = CFG["lora_dropout"],
    target_modules: Optional[list[str]] = None,
):
    """
    Build LoraConfig. Target modules auto-detected for common architectures
    (Mistral, LLaMA, Qwen), or pass custom list.
    """
    from peft import LoraConfig, TaskType

    if target_modules is None:
        # Standard attention + MLP projections for 7B-class LLaMA/Mistral
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    return LoraConfig(
        task_type       = TaskType.CAUSAL_LM,
        r               = r,
        lora_alpha      = alpha,
        lora_dropout    = dropout,
        target_modules  = target_modules,
        bias            = "none",
        inference_mode  = False,
    )


def build_bnb_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit               = CFG["load_in_4bit"],
        bnb_4bit_quant_type        = "nf4",
        bnb_4bit_compute_dtype     = torch.bfloat16,
        bnb_4bit_use_double_quant  = True,
    )


# ─────────────────────────────────────────────
# 4. SFT Trainer (TRL)
# ─────────────────────────────────────────────

def fine_tune(
    base_model_id: str,
    dataset,                          # HF Dataset
    output_dir:    Path,
    run_name:      str = "lora-ft",
    epochs:        int   = FT_EPOCHS,
    lr:            float = FT_LR,
    lora_config          = None,
    use_wandb:     bool  = False,
) -> Path:
    """
    QLoRA fine-tuning with TRL SFTTrainer.
    Saves merged adapter + tokenizer to output_dir.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[FT] Loading base model: {base_model_id}")
    bnb = build_bnb_config()
    tok = AutoTokenizer.from_pretrained(base_model_id, token=HF_TOKEN or None, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config   = bnb,
        device_map            = "auto",
        token                 = HF_TOKEN or None,
        trust_remote_code     = True,
        torch_dtype           = torch.bfloat16,
        use_cache             = False,   # required for gradient checkpointing
    )
    model = prepare_model_for_kbit_training(model)

    if lora_config is None:
        lora_config = build_lora_config()

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir                = str(output_dir),
        num_train_epochs          = epochs,
        per_device_train_batch_size = CFG["per_device_train_batch_size"],
        gradient_accumulation_steps = CFG["gradient_accumulation_steps"],
        gradient_checkpointing    = True,
        optim                     = "paged_adamw_32bit",
        learning_rate             = lr,
        weight_decay              = 0.001,
        warmup_ratio              = FT_WARMUP_RATIO,
        lr_scheduler_type         = "cosine",
        fp16                      = False,
        bf16                      = True,
        logging_steps             = 10,
        save_steps                = FT_SAVE_STEPS,
        save_total_limit          = 2,
        report_to                 = "wandb" if use_wandb else "none",
        run_name                  = run_name,
        dataloader_num_workers    = 4,
        group_by_length           = True,
    )

    # Response-only training: only compute loss on assistant turn
    response_template = " [/INST]"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template, tokenizer=tok
    )

    trainer = SFTTrainer(
        model             = model,
        train_dataset     = dataset,
        args              = training_args,
        tokenizer         = tok,
        data_collator     = collator,
        dataset_text_field = "text",
        max_seq_length    = CFG["max_seq_length"],
        packing           = True,    # pack short sequences for efficiency
    )

    logger.info("[FT] Starting training...")
    trainer.train()
    trainer.save_model(str(output_dir / "adapter"))
    tok.save_pretrained(str(output_dir / "adapter"))

    logger.success(f"[FT] Adapter saved → {output_dir / 'adapter'}")
    return output_dir / "adapter"


# ─────────────────────────────────────────────
# 5. Delta adapter extraction
# ─────────────────────────────────────────────

def extract_delta_adapter(
    base_model_id: str,
    finetuned_path: str,
    output_dir:    Path,
) -> Path:
    """
    Extract the LoRA delta weights as a standalone adapter.
    These can be merged back into any compatible base using mergekit.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("[Delta] Loading base model for delta extraction...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, device_map="cpu", torch_dtype=torch.float16,
        token=HF_TOKEN or None, trust_remote_code=True,
    )
    peft_model = PeftModel.from_pretrained(base, finetuned_path)

    logger.info("[Delta] Merging and unloading LoRA weights...")
    merged = peft_model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir))
    AutoTokenizer.from_pretrained(base_model_id, token=HF_TOKEN or None).save_pretrained(str(output_dir))
    logger.success(f"[Delta] Merged adapter model saved → {output_dir}")
    return output_dir


# ─────────────────────────────────────────────
# 6. Iterative improvement loop
# ─────────────────────────────────────────────

def improvement_loop(
    base_model_id:    str,
    eval_samples_fn,               # callable → list[Sample]
    max_iterations:   int   = 3,
    target_rouge:     float = 0.45,
    n_syn_per_gap:    int   = 50,
    use_wandb:        bool  = False,
) -> Path:
    """
    The core improvement loop:
      eval → detect gaps → generate data → fine-tune → eval → repeat

    Returns path to the final best adapter.
    """
    from phase3_evaluation.evaluate import evaluate, EvalResult

    best_model  = base_model_id
    best_rouge  = 0.0
    best_adapter: Optional[Path] = None
    history: list[EvalResult] = []

    for iteration in range(1, max_iterations + 1):
        logger.info(f"\n{'='*60}\n[Loop] Iteration {iteration}/{max_iterations}\n{'='*60}")

        # 1. Evaluate current best model
        samples = eval_samples_fn()
        result  = evaluate(best_model, samples, f"iter_{iteration}", run_judge=False)
        history.append(result)

        current_rouge = result.avg_rouge1
        logger.info(f"[Loop] ROUGE-1: {current_rouge:.3f} | gaps: {result.gap_categories}")

        # 2. Check convergence
        if current_rouge >= target_rouge:
            logger.success(f"[Loop] Target reached ({current_rouge:.3f} ≥ {target_rouge}). Stopping.")
            break
        if not result.gap_categories:
            logger.info("[Loop] No gaps detected. Stopping.")
            break

        # 3. Generate synthetic data for detected gaps
        syn_samples = generate_synthetic_data(result.gap_categories, n_per_gap=n_syn_per_gap)
        if not syn_samples:
            logger.warning("[Loop] No synthetic samples generated. Stopping.")
            break

        dataset = format_as_hf_dataset(syn_samples)

        # 4. Fine-tune
        adapter_dir = ADAPTERS_DIR / f"iter_{iteration}"
        fine_tune(
            base_model_id = best_model,
            dataset       = dataset,
            output_dir    = adapter_dir,
            run_name      = f"iter-{iteration}",
            use_wandb     = use_wandb,
        )

        # 5. Extract merged model for next iteration
        merged_dir  = adapter_dir / "merged"
        best_model  = str(extract_delta_adapter(best_model, str(adapter_dir / "adapter"), merged_dir))
        best_rouge  = current_rouge
        best_adapter= adapter_dir / "adapter"

        logger.info(f"[Loop] Iteration {iteration} done. Next model: {best_model}")

    logger.success(f"[Loop] Finished. Best ROUGE-1: {max(r.avg_rouge1 for r in history):.3f}")
    return best_adapter or Path(base_model_id)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

@app.command()
def run(
    base:        str        = typer.Option(FT_BASE_MODEL, "--base", help="Base model ID"),
    gaps:        list[str]  = typer.Option([], "--gap", "-g", help="Gap categories (repeat flag)"),
    data_path:   str        = typer.Option(None, help="Existing synthetic JSONL path"),
    n_syn:       int        = typer.Option(50,   help="Synthetic samples per gap"),
    output:      Path       = typer.Option(ADAPTERS_DIR / "run", "--output", "-o"),
    epochs:      int        = typer.Option(FT_EPOCHS),
    loop:        bool       = typer.Option(False, "--loop", help="Run iterative improvement loop"),
    max_iter:    int        = typer.Option(3,     help="Max iterations (--loop mode)"),
    wandb:       bool       = typer.Option(False, "--wandb"),
):
    if loop:
        from phase3_evaluation.evaluate import load_squad
        improvement_loop(
            base_model_id  = base,
            eval_samples_fn = lambda: load_squad(50),
            max_iterations = max_iter,
            use_wandb      = wandb,
        )
        return

    # One-shot fine-tune
    if data_path:
        dataset = load_jsonl_dataset(data_path)
    elif gaps:
        syn = generate_synthetic_data(list(gaps), n_per_gap=n_syn)
        dataset = format_as_hf_dataset(syn)
    else:
        raise typer.BadParameter("Provide --gap or --data-path")

    fine_tune(base, dataset, output, epochs=epochs, use_wandb=wandb)


if __name__ == "__main__":
    app()
