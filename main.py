import os
import math
import signal
import pandas as pd

from src.inference.baseline import BaselineKoyashLLM
from src.inference.finetuned import FinetunedKoyashLLM
from src.eval.metrics import compute_perplexity, compute_rouge_l, compute_bert_score

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
PREPROCESSED    = os.path.join(BASE_DIR, "data", "preprocessed")
EVAL_CSV        = os.path.join(PREPROCESSED, "metrics_eval.csv")
SYSTEM_PROMPT   = os.path.join(PREPROCESSED, "system_prompt.txt")
OUTPUT_CSV      = "metrics_results.csv"
TIMEOUT_SEC     = 180  # 3 минуты


def load_system_prompt() -> str:
    try:
        with open(SYSTEM_PROMPT, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "Ты Koyash Ассистент"


def build_models(system_prompt: str, temperature: float = 0.3) -> list:
    return [
        BaselineKoyashLLM(system_prompt=system_prompt, temperature=temperature),
        # FinetunedKoyashLLM(system_prompt=system_prompt, temperature=temperature),
    ]


def load_done_ids(output_csv: str, model_name: str) -> set:
    """Возвращает set id, которые уже записаны для данной модели."""
    if not os.path.exists(output_csv):
        return set()
    df = pd.read_csv(output_csv)
    return set(df.loc[df["model_name"] == model_name, "id"].astype(str))


def append_row(output_csv: str, row: dict) -> None:
    """Дописывает одну строку в CSV (создаёт файл если нет)."""
    df = pd.DataFrame([row])
    write_header = not os.path.exists(output_csv)
    df.to_csv(output_csv, mode="a", header=write_header, index=False, encoding="utf-8")


def evaluate_model(model, eval_df: pd.DataFrame, output_csv: str) -> None:
    done_ids = load_done_ids(output_csv, model.name)
    total    = len(eval_df)

    for i, (_, row) in enumerate(eval_df.iterrows(), 1):
        sample_id   = str(row["id"])
        user_prompt = row["prompt"]
        reference   = str(row.get("response", ""))

        if sample_id in done_ids:
            print(f"[{model.name}] ({i}/{total}) id={sample_id} — пропуск (уже есть)", flush=True)
            continue

        print(f"[{model.name}] ({i}/{total}) id={sample_id}...", flush=True)

        # ── таймаут через SIGALRM (Unix) ──────────────────────────────
        def _timeout_handler(signum, frame):
            raise TimeoutError()

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(TIMEOUT_SEC)

        try:
            response   = model.get_response(user_prompt)
            answer     = model.get_answer(response)
            logprobs   = response.get("logprobs") or []
            perplexity = compute_perplexity(logprobs)
            rouge_l    = compute_rouge_l(answer, reference) if reference else float("nan")
            bert_s     = compute_bert_score(answer, reference) if reference else float("nan")
            print(f"  ✓ PPL={perplexity:.2f}, ROUGE-L={rouge_l:.4f}, BERTScore={bert_s:.4f}", flush=True)

        except TimeoutError:
            print(f"  ✗ TIMEOUT после {TIMEOUT_SEC}s — пропускаем", flush=True)
            answer, perplexity, rouge_l, bert_s = "TIMEOUT", float("nan"), float("nan"), float("nan")

        except Exception as e:
            print(f"  ✗ Ошибка: {e} — пропускаем", flush=True)
            answer, perplexity, rouge_l, bert_s = "ERROR", float("nan"), float("nan"), float("nan")

        finally:
            signal.alarm(0)  # сбросить таймер

        append_row(output_csv, {
            "id":         sample_id,
            "model_name": model.name,
            "response":   answer,
            "perplexity": perplexity,
            "rouge_l":    rouge_l,
            "bert_score": bert_s,
        })


def main():
    print(f"Loading eval dataset from: {EVAL_CSV}")
    eval_df = pd.read_csv(EVAL_CSV, skipinitialspace=True)
    print(f"Total samples: {len(eval_df)}")

    system_prompt = load_system_prompt()
    models        = build_models(system_prompt)

    for model_idx, model in enumerate(models, 1):
        print(f"\n[{model_idx}/{len(models)}] Model: {model.name}")
        evaluate_model(model, eval_df, OUTPUT_CSV)
        print(f"  Done: {model.name}")

    print(f"\nAll results in: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
