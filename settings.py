"""
Global configuration for the LLM Pipeline.
Edit these values to match your environment.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
ROOT          = Path(__file__).parent.parent
MODELS_DIR    = ROOT / "artifacts" / "models"
MERGES_DIR    = ROOT / "artifacts" / "merges"
ADAPTERS_DIR  = ROOT / "artifacts" / "adapters"
EVAL_DIR      = ROOT / "artifacts" / "evaluations"
DATA_DIR      = ROOT / "artifacts" / "data"
LOGS_DIR      = ROOT / "logs"

for _d in [MODELS_DIR, MERGES_DIR, ADAPTERS_DIR, EVAL_DIR, DATA_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Scale: medium (7B, single A100/H100)
# ─────────────────────────────────────────────
SCALE = "medium"

SCALE_DEFAULTS = {
    "medium": {
        "max_model_params_b": 10,      # max billions of params to consider
        "dtype": "bfloat16",
        "device_map": "auto",
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "max_seq_length": 2048,
        "load_in_4bit": True,
    }
}

CFG = SCALE_DEFAULTS[SCALE]

# ─────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────
HF_MODEL_CATEGORIES = {
    "code":       ["starcoder", "codellama", "deepseek-coder", "qwen2.5-coder"],
    "reasoning":  ["mistral", "llama", "qwen", "gemma"],
    "chat":       ["zephyr", "openchat", "neural-chat"],
    "medical":    ["meditron", "biomedlm", "llama-med"],
    "multilingual":["aya", "bloomz", "xglm"],
}

TOP_K_CANDIDATES   = 5      # per category
MIN_DOWNLOADS      = 1_000  # filter noise
MIN_LIKES          = 10

# ─────────────────────────────────────────────
# Merging
# ─────────────────────────────────────────────
MERGE_STRATEGIES = ["slerp", "ties", "dare_ties", "task_arithmetic", "breadcrumbs"]

# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
EVAL_METRICS  = ["rouge", "bertscore", "faithfulness", "llm_judge"]
JUDGE_MODEL   = "mistralai/Mistral-7B-Instruct-v0.3"   # swap for a stronger model
ROUGE_TYPES   = ["rouge1", "rouge2", "rougeL"]

# ─────────────────────────────────────────────
# Fine-tuning
# ─────────────────────────────────────────────
FT_BASE_MODEL = "mistralai/Mistral-7B-v0.3"
FT_EPOCHS     = 3
FT_LR         = 2e-4
FT_WARMUP_RATIO = 0.03
FT_SAVE_STEPS   = 100

# ─────────────────────────────────────────────
# Inference (vLLM)
# ─────────────────────────────────────────────
VLLM_TENSOR_PARALLEL = 1       # GPUs
VLLM_GPU_MEMORY_UTIL = 0.90
VLLM_MAX_MODEL_LEN   = 4096

# ─────────────────────────────────────────────
# MLOps
# ─────────────────────────────────────────────
WANDB_PROJECT  = "llm-pipeline"
MLFLOW_URI     = "sqlite:///mlflow.db"
HF_ORG         = ""           # your HF username/org for pushing models

# ─────────────────────────────────────────────
# Tokens (set via env vars, never hardcode)
# ─────────────────────────────────────────────
import os
HF_TOKEN    = os.getenv("HF_TOKEN", "")
WANDB_TOKEN = os.getenv("WANDB_API_KEY", "")
