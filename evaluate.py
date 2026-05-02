"""
Phase 3 — Evaluation Framework
================================
Multi-metric evaluation pipeline:
  • ROUGE (rouge1, rouge2, rougeL)
  • BERTScore
  • Faithfulness / hallucination detection
  • LLM-as-Judge (model comparison)
  • Knowledge gap detector (feeds Phase 4)

Usage:
    python -m phase3_evaluation.evaluate --model ./merged --dataset squad --output ./evals
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import track

from configs.settings import (
    EVAL_DIR, ROUGE_TYPES, JUDGE_MODEL, CFG, HF_TOKEN
)
from utils.logger import logger

app     = typer.Typer(help="Phase 3: Evaluation framework")
console = Console()


# ─────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────

@dataclass
class Sample:
    id:        str
    context:   str
    question:  str
    reference: str          # ground-truth answer
    prediction: str = ""   # filled during eval


@dataclass
class SampleScore:
    sample_id:    str
    rouge1:       float = 0.0
    rouge2:       float = 0.0
    rougeL:       float = 0.0
    bertscore_f1: float = 0.0
    faithfulness: float = 0.0   # 0–1
    hallucinated: bool  = False
    judge_score:  float = 0.0   # 0–10, from LLM-Judge
    gap_signals:  list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    model_id:       str
    dataset:        str
    n_samples:      int
    avg_rouge1:     float = 0.0
    avg_rouge2:     float = 0.0
    avg_rougeL:     float = 0.0
    avg_bertscore:  float = 0.0
    avg_faithfulness: float = 0.0
    hallucination_rate: float = 0.0
    avg_judge_score: float = 0.0
    gap_categories: list[str] = field(default_factory=list)
    sample_scores:  list[SampleScore] = field(default_factory=list)


# ─────────────────────────────────────────────
# 1. ROUGE scorer
# ─────────────────────────────────────────────

class RougeScorer:
    def __init__(self):
        from rouge_score import rouge_scorer as rs
        self._scorer = rs.RougeScorer(ROUGE_TYPES, use_stemmer=True)

    def score(self, prediction: str, reference: str) -> dict[str, float]:
        scores = self._scorer.score(reference, prediction)
        return {k: scores[k].fmeasure for k in ROUGE_TYPES}


# ─────────────────────────────────────────────
# 2. BERTScore
# ─────────────────────────────────────────────

class BertScorer:
    def __init__(self, lang: str = "en", device: Optional[str] = None):
        self.lang   = lang
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None   # lazy load

    def _load(self):
        if self._model is None:
            from bert_score import BERTScorer
            self._model = BERTScorer(lang=self.lang, device=self.device, rescale_with_baseline=True)

    def score_batch(self, predictions: list[str], references: list[str]) -> list[float]:
        self._load()
        _, _, f1 = self._model.score(predictions, references)
        return f1.tolist()


# ─────────────────────────────────────────────
# 3. Faithfulness / Hallucination Detector
# ─────────────────────────────────────────────

HALLUCINATION_PATTERNS = [
    r"\b(definitely|certainly|always|never|everyone|nobody)\b",
    r"\b(100%|absolutely|guaranteed)\b",
    r"\bin \d{4},? .{0,30} (invented|discovered|created|founded)\b",
]

class FaithfulnessChecker:
    """
    Lightweight faithfulness checker using:
      1. Heuristic hallucination patterns
      2. NLI-based entailment (context → prediction)
    """

    def __init__(self):
        self._nli = None

    def _load_nli(self):
        if self._nli is None:
            from transformers import pipeline
            self._nli = pipeline(
                "text-classification",
                model="cross-encoder/nli-deberta-v3-small",
                device=0 if torch.cuda.is_available() else -1,
            )

    def _heuristic_hallucination(self, prediction: str) -> bool:
        return any(re.search(p, prediction, re.IGNORECASE) for p in HALLUCINATION_PATTERNS)

    def check(self, context: str, prediction: str) -> tuple[float, bool]:
        """
        Returns (faithfulness_score 0-1, is_hallucinated).
        faithfulness_score = NLI entailment probability.
        """
        hallucinated = self._heuristic_hallucination(prediction)
        try:
            self._load_nli()
            pair   = f"{context[:512]} [SEP] {prediction[:256]}"
            result = self._nli(pair, truncation=True, max_length=512)[0]
            label  = result["label"].upper()
            score  = result["score"]
            faith  = score if label == "ENTAILMENT" else (1.0 - score if label == "CONTRADICTION" else 0.5)
            if faith < 0.3:
                hallucinated = True
        except Exception as e:
            logger.warning(f"NLI check failed: {e}")
            faith = 0.5
        return faith, hallucinated

    def check_batch(self, contexts: list[str], predictions: list[str]) -> list[tuple[float, bool]]:
        return [self.check(c, p) for c, p in zip(contexts, predictions)]


# ─────────────────────────────────────────────
# 4. LLM-as-Judge
# ─────────────────────────────────────────────

JUDGE_PROMPT = """You are an expert evaluator. Score the following answer on a scale of 0-10.

Question: {question}
Reference Answer: {reference}
Model Answer: {prediction}

Evaluation criteria:
- Accuracy (does it match the reference?)
- Completeness (does it cover all key points?)
- Clarity (is it well-expressed?)
- Faithfulness (no hallucinations?)

Respond ONLY with: SCORE: <integer 0-10>\nREASON: <one sentence>
"""

class LLMJudge:
    def __init__(self, judge_model_id: str = JUDGE_MODEL):
        self.model_id = judge_model_id
        self._pipe    = None

    def _load(self):
        if self._pipe is None:
            from transformers import pipeline, BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            self._pipe = pipeline(
                "text-generation",
                model=self.model_id,
                model_kwargs={"quantization_config": bnb},
                device_map="auto",
                token=HF_TOKEN or None,
                trust_remote_code=True,
            )

    def score(self, question: str, reference: str, prediction: str) -> tuple[float, str]:
        """Returns (score 0-10, reason string)."""
        try:
            self._load()
            prompt = JUDGE_PROMPT.format(
                question=question, reference=reference, prediction=prediction
            )
            out    = self._pipe(prompt, max_new_tokens=64, do_sample=False)[0]["generated_text"]
            # Parse response
            score_match  = re.search(r"SCORE:\s*(\d+)", out)
            reason_match = re.search(r"REASON:\s*(.+)", out)
            score  = int(score_match.group(1)) if score_match else 5
            reason = reason_match.group(1).strip() if reason_match else ""
            return min(10.0, max(0.0, float(score))), reason
        except Exception as e:
            logger.warning(f"Judge scoring failed: {e}")
            return 5.0, ""

    def compare_models(
        self,
        question: str, reference: str,
        pred_a: str, pred_b: str,
        model_a_name: str = "A", model_b_name: str = "B",
    ) -> dict:
        score_a, _ = self.score(question, reference, pred_a)
        score_b, _ = self.score(question, reference, pred_b)
        winner     = model_a_name if score_a >= score_b else model_b_name
        return {
            "winner": winner,
            model_a_name: score_a,
            model_b_name: score_b,
            "delta": abs(score_a - score_b),
        }


# ─────────────────────────────────────────────
# 5. Knowledge Gap Detector
# ─────────────────────────────────────────────

GAP_THRESHOLDS = {
    "rouge":       0.30,
    "bertscore":   0.50,
    "faithfulness": 0.50,
    "judge":       5.0,
}

GAP_CATEGORY_KEYWORDS = {
    "factual_recall":    ["who", "when", "where", "what year", "which country"],
    "multi_step_reasoning": ["therefore", "because", "if", "then", "step"],
    "numerical":         ["how many", "calculate", "percent", "number", "count"],
    "code_generation":   ["write a function", "implement", "code", "python", "sql"],
    "summarization":     ["summarize", "summary", "main points", "briefly"],
}


def detect_gaps(sample_scores: list[SampleScore], samples: list[Sample]) -> list[str]:
    """
    Analyze per-sample scores to identify systematic weaknesses.
    Returns list of gap category labels.
    """
    gap_counts: dict[str, int] = {}
    failing: list[int] = []

    for i, ss in enumerate(sample_scores):
        is_failing = (
            ss.rouge1     < GAP_THRESHOLDS["rouge"]       or
            ss.bertscore_f1 < GAP_THRESHOLDS["bertscore"] or
            ss.faithfulness < GAP_THRESHOLDS["faithfulness"] or
            ss.judge_score  < GAP_THRESHOLDS["judge"]     or
            ss.hallucinated
        )
        if is_failing:
            failing.append(i)

    for idx in failing:
        question = (samples[idx].question + " " + samples[idx].context).lower()
        for cat, keywords in GAP_CATEGORY_KEYWORDS.items():
            if any(kw in question for kw in keywords):
                gap_counts[cat] = gap_counts.get(cat, 0) + 1

    # Return categories with ≥2 failures
    gaps = [cat for cat, cnt in gap_counts.items() if cnt >= 2]
    logger.info(f"[GapDetect] Identified gaps: {gaps or ['none']}")
    return gaps


# ─────────────────────────────────────────────
# 6. Inference helper
# ─────────────────────────────────────────────

def run_inference(
    model_id: str,
    samples:  list[Sample],
    batch_size: int = 4,
) -> list[Sample]:
    """Fill sample.prediction for each sample using the target model."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN or None, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        token=HF_TOKEN or None,
        trust_remote_code=True,
    )
    model.eval()

    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        prompts = [
            f"Context: {s.context}\nQuestion: {s.question}\nAnswer:"
            for s in batch
        ]
        inputs = tok(prompts, return_tensors="pt", padding=True,
                     truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        for j, sample in enumerate(batch):
            new_tokens = out_ids[j][inputs["input_ids"].shape[1]:]
            sample.prediction = tok.decode(new_tokens, skip_special_tokens=True).strip()

    del model
    torch.cuda.empty_cache()
    return samples


# ─────────────────────────────────────────────
# 7. Master evaluation pipeline
# ─────────────────────────────────────────────

def evaluate(
    model_id:      str,
    samples:       list[Sample],
    dataset_name:  str = "custom",
    run_judge:     bool = True,
    save:          bool = True,
) -> EvalResult:
    """
    Full evaluation pipeline for one model on a dataset.
    """
    n = len(samples)
    logger.info(f"[Eval] Evaluating '{model_id}' on {n} samples...")

    # 1. Generate predictions
    samples = run_inference(model_id, samples)

    # 2. Initialize scorers
    rouge    = RougeScorer()
    bert     = BertScorer()
    faith    = FaithfulnessChecker()
    judge    = LLMJudge() if run_judge else None

    # 3. Score each sample
    predictions = [s.prediction for s in samples]
    references  = [s.reference  for s in samples]

    # BERTScore in batch
    bs_f1s = bert.score_batch(predictions, references)

    sample_scores: list[SampleScore] = []

    for i, s in enumerate(track(samples, description="Scoring")):
        rscores    = rouge.score(s.prediction, s.reference)
        faith_sc, hallu = faith.check(s.context, s.prediction)
        j_score    = 5.0
        if judge:
            j_score, _ = judge.score(s.question, s.reference, s.prediction)

        ss = SampleScore(
            sample_id    = s.id,
            rouge1       = rscores["rouge1"],
            rouge2       = rscores["rouge2"],
            rougeL       = rscores["rougeL"],
            bertscore_f1 = bs_f1s[i],
            faithfulness = faith_sc,
            hallucinated = hallu,
            judge_score  = j_score,
        )
        sample_scores.append(ss)

    # 4. Aggregate
    def avg(lst): return sum(lst) / len(lst) if lst else 0.0

    result = EvalResult(
        model_id           = model_id,
        dataset            = dataset_name,
        n_samples          = n,
        avg_rouge1         = avg([s.rouge1       for s in sample_scores]),
        avg_rouge2         = avg([s.rouge2       for s in sample_scores]),
        avg_rougeL         = avg([s.rougeL       for s in sample_scores]),
        avg_bertscore      = avg([s.bertscore_f1 for s in sample_scores]),
        avg_faithfulness   = avg([s.faithfulness for s in sample_scores]),
        hallucination_rate = avg([float(s.hallucinated) for s in sample_scores]),
        avg_judge_score    = avg([s.judge_score  for s in sample_scores]),
        gap_categories     = detect_gaps(sample_scores, samples),
        sample_scores      = sample_scores,
    )

    # 5. Display
    _print_eval_table(result)

    # 6. Save
    if save:
        out = EVAL_DIR / f"eval_{model_id.replace('/', '_')}_{dataset_name}.json"
        with open(out, "w") as f:
            json.dump(asdict(result), f, indent=2)
        logger.info(f"[Eval] Results saved → {out}")

    return result


def compare_models(
    model_ids:    list[str],
    samples:      list[Sample],
    dataset_name: str = "custom",
) -> list[EvalResult]:
    """Evaluate multiple models and print a side-by-side comparison."""
    results = [evaluate(mid, samples, dataset_name) for mid in model_ids]
    _print_comparison_table(results)
    return results


def _print_eval_table(r: EvalResult) -> None:
    table = Table(title=f"Evaluation — {r.model_id}", show_lines=True)
    table.add_column("Metric",     style="cyan")
    table.add_column("Score",      justify="right", style="green")
    table.add_column("Threshold",  justify="right", style="dim")

    rows = [
        ("ROUGE-1",          f"{r.avg_rouge1:.3f}",          f"{GAP_THRESHOLDS['rouge']:.2f}"),
        ("ROUGE-2",          f"{r.avg_rouge2:.3f}",          "—"),
        ("ROUGE-L",          f"{r.avg_rougeL:.3f}",          "—"),
        ("BERTScore F1",     f"{r.avg_bertscore:.3f}",       f"{GAP_THRESHOLDS['bertscore']:.2f}"),
        ("Faithfulness",     f"{r.avg_faithfulness:.3f}",    f"{GAP_THRESHOLDS['faithfulness']:.2f}"),
        ("Hallucination %",  f"{r.hallucination_rate*100:.1f}%", "< 10%"),
        ("Judge Score",      f"{r.avg_judge_score:.1f}/10",  f"{GAP_THRESHOLDS['judge']:.0f}/10"),
    ]
    for row in rows:
        table.add_row(*row)
    console.print(table)
    if r.gap_categories:
        console.print(f"[yellow]⚠ Gaps detected:[/yellow] {', '.join(r.gap_categories)}")


def _print_comparison_table(results: list[EvalResult]) -> None:
    table = Table(title="Model Comparison", show_lines=True)
    table.add_column("Metric", style="cyan")
    for r in results:
        table.add_column(r.model_id.split("/")[-1], justify="right")
    metrics = [
        ("ROUGE-1",     lambda r: f"{r.avg_rouge1:.3f}"),
        ("BERTScore",   lambda r: f"{r.avg_bertscore:.3f}"),
        ("Faithfulness",lambda r: f"{r.avg_faithfulness:.3f}"),
        ("Halluc. %",   lambda r: f"{r.hallucination_rate*100:.1f}%"),
        ("Judge",       lambda r: f"{r.avg_judge_score:.1f}"),
    ]
    for label, fn in metrics:
        table.add_row(label, *[fn(r) for r in results])
    console.print(table)


# ─────────────────────────────────────────────
# Dataset loaders
# ─────────────────────────────────────────────

def load_squad(n: int = 100) -> list[Sample]:
    """Load SQuAD v2 validation set as Sample list."""
    from datasets import load_dataset
    ds = load_dataset("squad_v2", split=f"validation[:{n}]")
    samples = []
    for i, row in enumerate(ds):
        if not row["answers"]["text"]:
            continue   # skip unanswerable
        samples.append(Sample(
            id        = str(i),
            context   = row["context"],
            question  = row["question"],
            reference = row["answers"]["text"][0],
        ))
    return samples[:n]


def load_custom(path: str) -> list[Sample]:
    """Load custom JSONL dataset: {id, context, question, reference}"""
    with open(path) as f:
        return [Sample(**json.loads(line)) for line in f]


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

@app.command()
def run(
    model:       str   = typer.Argument(..., help="Model ID or local path"),
    dataset:     str   = typer.Option("squad", help="squad | custom"),
    data_path:   str   = typer.Option(None,    help="Path to custom JSONL (if dataset=custom)"),
    n_samples:   int   = typer.Option(100,     help="Number of samples to evaluate"),
    judge:       bool  = typer.Option(True,    "--judge/--no-judge"),
    compare:     list[str] = typer.Option([], "--compare", help="Additional models for comparison"),
):
    if dataset == "squad":
        samples = load_squad(n_samples)
    elif dataset == "custom" and data_path:
        samples = load_custom(data_path)[:n_samples]
    else:
        raise typer.BadParameter("Provide --data-path for custom dataset")

    if compare:
        compare_models([model] + list(compare), samples, dataset)
    else:
        evaluate(model, samples, dataset, run_judge=judge)


if __name__ == "__main__":
    app()
