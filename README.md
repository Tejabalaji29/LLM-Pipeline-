# LLM-Pipeline-
# 🤖 LLM Pipeline

> An end-to-end automated pipeline for **discovering**, **merging**, **evaluating**, and **fine-tuning** open-source LLMs — with full MLOps integration.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![HF Hub](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co)
[![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-tracking-orange)](https://wandb.ai)

---

## 🗺️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Phase 1 — Discovery    │  Scan HF Hub → filter → rank  │
├─────────────────────────────────────────────────────────┤
│  Phase 2 — Merging      │  SLERP · TIES · DARE · TA     │
├─────────────────────────────────────────────────────────┤
│  Phase 3 — Evaluation   │  ROUGE · BERTScore · Judge    │
├─────────────────────────────────────────────────────────┤
│  Phase 4 — Fine-Tuning  │  LoRA/QLoRA · Synthetic Data  │
├─────────────────────────────────────────────────────────┤
│  Phase 5 — MLOps        │  vLLM · W&B · MLflow · HF Hub │
└─────────────────────────────────────────────────────────┘
            ↑__________________________|
                 Iterative improvement loop
```

---

## ✨ Features

### Phase 1 — Model Discovery
- Automated HF Hub crawler with category-based keyword search
- Quality filtering: downloads, likes, parameter count, model card completeness
- Optional lightweight perplexity probe for fast quality estimation
- Composite scoring and ranked shortlist output

### Phase 2 — Model Composition
- **Union merging** (capability aggregation):
  - SLERP — spherical linear interpolation
  - TIES — trim, elect sign, merge
  - DARE-TIES — dropout + TIES
  - Task Arithmetic — delta-weight addition
- **Intersection merging** (conservative):
  - Breadcrumbs — consensus-only parameter updates
- DOM-tree-style architecture introspection (layers, attention heads, MLP blocks)
- mergekit integration + pure-PyTorch fallback

### Phase 3 — Evaluation Framework
- ROUGE (rouge1, rouge2, rougeL)
- BERTScore (semantic similarity)
- Faithfulness + hallucination detection (NLI-based)
- LLM-as-Judge scoring (0–10)
- Multi-model side-by-side comparison
- **Knowledge gap detector** → feeds Phase 4

### Phase 4 — Efficient Fine-Tuning
- QLoRA (4-bit NF4 quantization + LoRA adapters)
- Response-only training (loss on assistant turns only)
- Synthetic data generation per detected gap category
- Delta adapter extraction (merge-ready weights)
- **Iterative improvement loop**: eval → gap detect → generate → train → repeat

### Phase 5 — MLOps
- vLLM inference with PagedAttention (OpenAI-compatible API)
- Throughput benchmarking
- Dual tracking: Weights & Biases + MLflow
- Auto-generated model cards
- One-command HF Hub deployment

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/llm-pipeline.git
cd llm-pipeline
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
export HF_TOKEN="hf_..."          # Hugging Face token
export WANDB_API_KEY="..."        # W&B token (optional)
```

### 3. Run the full pipeline

```bash
# Full pipeline for reasoning models
python pipeline.py run reasoning

# With iterative improvement loop
python pipeline.py run code --loop --max-iter 3

# Custom merge strategy
python pipeline.py run medical --strategy breadcrumbs --top-k 3
```

---

## 📖 Usage — Individual Phases

### Phase 1: Discovery
```bash
python -m phase1_discovery.discover run reasoning --top-k 5
python -m phase1_discovery.discover run code --perplexity  # adds perplexity probe
python -m phase1_discovery.discover run --all              # all categories
```

### Phase 2: Merging
```bash
# TIES merge (recommended for union)
python -m phase2_merging.merge run ties \
  --model mistralai/Mistral-7B-v0.3 \
  --model teknium/OpenHermes-2.5-Mistral-7B \
  --base mistralai/Mistral-7B-v0.3

# SLERP interpolation
python -m phase2_merging.merge run slerp \
  --model model_a --model model_b --alpha 0.6

# Breadcrumbs (conservative / intersection)
python -m phase2_merging.merge run breadcrumbs \
  --model base --model ft_a --model ft_b --density 0.7

# Inspect architecture (DOM-tree view)
python -m phase2_merging.merge run ties \
  --introspect mistralai/Mistral-7B-v0.3
```

### Phase 3: Evaluation
```bash
# Evaluate on SQuAD v2
python -m phase3_evaluation.evaluate run ./merged_model --dataset squad --n-samples 200

# Compare multiple models
python -m phase3_evaluation.evaluate run model_a \
  --compare model_b --compare model_c

# Disable LLM judge (faster)
python -m phase3_evaluation.evaluate run ./merged --no-judge
```

### Phase 4: Fine-Tuning
```bash
# Fine-tune targeting specific gaps
python -m phase4_finetuning.finetune run \
  --base mistralai/Mistral-7B-v0.3 \
  --gap factual_recall --gap numerical \
  --n-syn 100 --output ./adapters/run1

# Use existing synthetic data
python -m phase4_finetuning.finetune run \
  --base mistralai/Mistral-7B-v0.3 \
  --data-path ./artifacts/data/synthetic_data.jsonl

# Iterative loop
python -m phase4_finetuning.finetune run --loop --max-iter 3
```

### Phase 5: Inference & MLOps
```bash
# Start vLLM server (OpenAI-compatible)
python -m phase5_mlops.serve serve ./merged_model --port 8000

# Benchmark throughput
python -m phase5_mlops.serve serve ./merged_model --bench

# Track experiment
python -m phase5_mlops.serve track my-run \
  --model ./merged --strategy ties \
  --rouge1 0.42 --bertscore 0.71 --judge 7.3

# Deploy to HF Hub
python -m phase5_mlops.serve deploy ./merged_model \
  --repo your-username/my-merged-7b

# View leaderboard
python -m phase5_mlops.serve leaderboard
```

---

## 📁 Project Structure

```
llm-pipeline/
├── pipeline.py                  # Master orchestrator
├── requirements.txt
├── configs/
│   └── settings.py              # All config: paths, scale, hyperparams
├── utils/
│   └── logger.py                # Centralized logging
├── phase1_discovery/
│   └── discover.py              # HF Hub crawler + ranking
├── phase2_merging/
│   └── merge.py                 # Merging + architecture introspection
├── phase3_evaluation/
│   └── evaluate.py              # Multi-metric eval + gap detection
├── phase4_finetuning/
│   └── finetune.py              # QLoRA + synthetic data + loop
├── phase5_mlops/
│   └── serve.py                 # vLLM + W&B + MLflow + HF deploy
└── artifacts/                   # Auto-created at runtime
    ├── models/
    ├── merges/
    ├── adapters/
    ├── evaluations/
    └── data/
```

---

## ⚙️ Configuration

Edit `configs/settings.py` to customize:

```python
# Scale preset (currently: medium = 7B, single A100)
SCALE = "medium"

# Categories and keywords for discovery
HF_MODEL_CATEGORIES = {
    "code":      ["starcoder", "codellama", "deepseek-coder"],
    "reasoning": ["mistral", "llama", "qwen"],
    ...
}

# Fine-tuning defaults
FT_BASE_MODEL = "mistralai/Mistral-7B-v0.3"
FT_EPOCHS     = 3
FT_LR         = 2e-4

# vLLM
VLLM_GPU_MEMORY_UTIL = 0.90
VLLM_MAX_MODEL_LEN   = 4096
```

---

## 🧩 Supported Merge Strategies

| Strategy | Type | Best For |
|---|---|---|
| `slerp` | Union | Two-model smooth interpolation |
| `ties` | Union | Multi-model, removes conflicting deltas |
| `dare_ties` | Union | Aggressive sparsification before TIES |
| `task_arithmetic` | Union | Adding task-specific capabilities |
| `breadcrumbs` | Intersection | Conservative, safety-preserving merge |

---

## 📊 Evaluation Metrics

| Metric | Tool | Threshold |
|---|---|---|
| ROUGE-1/2/L | `rouge-score` | ≥ 0.30 |
| BERTScore F1 | `bert-score` | ≥ 0.50 |
| Faithfulness | `cross-encoder/nli-deberta-v3-small` | ≥ 0.50 |
| Hallucination | Heuristic + NLI | < 10% |
| Judge Score | LLM-as-Judge | ≥ 5.0/10 |

---

## 🔄 Iterative Improvement Loop

```
┌─ Evaluate model ──────────────────────────────────┐
│   ROUGE / BERTScore / Judge / Faithfulness         │
└──────────────────┬────────────────────────────────┘
                   │ gaps detected?
                   ▼
┌─ Detect knowledge gaps ───────────────────────────┐
│   factual_recall / numerical / code / reasoning    │
└──────────────────┬────────────────────────────────┘
                   │
                   ▼
┌─ Generate synthetic data ─────────────────────────┐
│   LLM generates targeted QA pairs per gap          │
└──────────────────┬────────────────────────────────┘
                   │
                   ▼
┌─ QLoRA fine-tune ─────────────────────────────────┐
│   Response-only loss, 4-bit NF4, paged_adamw       │
└──────────────────┬────────────────────────────────┘
                   │
                   └──────────────► repeat until target ROUGE or max_iter
```

---

## 🛠️ Hardware Requirements

| Scale | GPU | RAM | Notes |
|---|---|---|---|
| Small (1–3B) | Any CUDA GPU | 16GB | CPU possible but slow |
| **Medium (7B)** | **A100 / H100 40GB** | **32GB** | **Recommended** |
| Large (13B+) | 2× A100 80GB | 64GB | Set `tensor_parallel=2` |

---

## 📦 Key Dependencies

- [`transformers`](https://github.com/huggingface/transformers) — model loading
- [`peft`](https://github.com/huggingface/peft) — LoRA/QLoRA adapters
- [`trl`](https://github.com/huggingface/trl) — SFTTrainer
- [`mergekit`](https://github.com/arcee-ai/mergekit) — TIES, DARE, SLERP
- [`vllm`](https://github.com/vllm-project/vllm) — high-throughput inference
- [`bert-score`](https://github.com/Tiiiger/bert_score) — semantic evaluation
- [`wandb`](https://wandb.ai) + [`mlflow`](https://mlflow.org) — experiment tracking

---

## 📄 License

[Apache 2.0](LICENSE)

---

## 🙏 Acknowledgements

- [mergekit](https://github.com/arcee-ai/mergekit) by Arcee AI
- [TIES-Merging](https://arxiv.org/abs/2306.01708) — Yadav et al., 2023
- [DARE](https://arxiv.org/abs/2311.03099) — Yu et al., 2023
- [Task Arithmetic](https://arxiv.org/abs/2212.04089) — Ilharco et al., 2023
- [QLoRA](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
