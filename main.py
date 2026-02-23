import os
import pandas as pd
from tqdm import tqdm

from src.inference.baseline import BaselineKoyashLLM
from src.inference.finetuned import FinetunedKoyashLLM
from src.eval.metrics import compute_perplexity, compute_rouge_l

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED    = os.path.join(BASE_DIR, "data", "preprocessed")
EVAL_CSV        = os.path.join(PREPROCESSED, "metrics_eval.csv")
SYSTEM_PROMPT   = os.path.join(PREPROCESSED, "system_prompt.txt")
OUTPUT_CSV      = "metrics_results.csv"

def load_system_prompt() -> str:
    try:
        with open(SYSTEM_PROMPT, encoding="utf-8") as f:
            return f.read()
    except:
        return "Ты Koyash Ассистент"


def build_models(system_prompt: str, temperature: float = 0.3) -> list:
    return [
        BaselineKoyashLLM(system_prompt=system_prompt, temperature=temperature),
        # FinetunedKoyashLLM(system_prompt=system_prompt, temperature=temperature), # пока не работает
    ]


def evaluate_model(model, eval_df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc=model.name, leave=False):
        sample_id   = row["id"]
        user_prompt = row["prompt"]
        reference   = row["response"]

        response = model.get_response(user_prompt)
        answer   = model.get_answer(response)

        raw_logprobs = response.get("logprobs") or []
        perplexity = compute_perplexity(raw_logprobs)

        rouge_l = compute_rouge_l(answer, reference)

        print(f"[{model.name}] id={sample_id}: PPL={perplexity:.2f}, ROUGE-L={rouge_l:.4f}")

        records.append(
            {
                "id":         sample_id,
                "model_name": model.name,
                "response":   answer,
                "perplexity": perplexity,
                "rouge_l":    rouge_l,
            }
        )

    return pd.DataFrame(records)

def main():
    print(f"Loading eval dataset from: {EVAL_CSV}")
    eval_df = pd.read_csv(EVAL_CSV, skipinitialspace=True)

    system_prompt = load_system_prompt()

    models  = build_models(system_prompt)
    results = []

    for model in models:
        print(f"Evaluating {model.name}")
        df = evaluate_model(model, eval_df)
        results.append(df)

    final_df = pd.concat(results, ignore_index=True)

    final_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
