import math
from rouge_score import rouge_scorer
from bert_score import score as bert_score


def _extract_logprob(entry) -> float | None:
    """Accept a dict, an object with .logprob, or a plain float."""
    if isinstance(entry, dict):
        return entry.get("logprob")
    if hasattr(entry, "logprob"):
        return entry.logprob
    try:
        return float(entry)
    except (TypeError, ValueError):
        return None


def compute_perplexity(logprobs: list) -> float:
    if not logprobs:
        return float("nan")

    log_probs = [v for entry in logprobs if (v := _extract_logprob(entry)) is not None]
    if not log_probs:
        return float("nan")

    avg_neg_log_prob = -sum(log_probs) / len(log_probs)
    return math.exp(avg_neg_log_prob)


def compute_perplexity_hf(model, input_ids, attention_mask=None) -> float:
    """Перплексия на HF causal-LM: exp(loss) при labels=input_ids."""
    import torch
    with torch.no_grad(), torch.autocast("cuda"):
        out = model(input_ids, attention_mask=attention_mask, labels=input_ids)
    return math.exp(out.loss.item())


def compute_rouge_l(hypothesis: str, reference: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = scorer.score(reference, hypothesis)
    return scores["rougeL"].fmeasure


def compute_bert_score(hypothesis: str, reference: str, lang: str = "ru") -> float:
    """F1 BERTScore между hypothesis и reference."""
    _, _, f1 = bert_score([hypothesis], [reference], lang=lang, verbose=False)
    return f1.item()
