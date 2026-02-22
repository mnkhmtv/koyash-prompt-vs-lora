import math
from rouge_score import rouge_scorer


def compute_perplexity(logprobs: list[dict]) -> float:
    if not logprobs:
        return float("nan")

    log_probs = [entry["logprob"] for entry in logprobs if entry["logprob"] is not None]
    if not log_probs:
        return float("nan")

    avg_neg_log_prob = -sum(log_probs) / len(log_probs)
    return math.exp(avg_neg_log_prob)


def compute_rouge_l(hypothesis: str, reference: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = scorer.score(reference, hypothesis)
    return scores["rougeL"].fmeasure
