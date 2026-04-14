"""merges LoRA adapter weights into the base model and saves the result as a new checkpoint"""
import shutil
import tempfile
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR   = Path(__file__).resolve().parents[1]
MODEL_DIR  = BASE_DIR / "model"
MERGED_DIR = MODEL_DIR / "merged"


MERGED_DIR.mkdir(parents=True, exist_ok=True)

with tempfile.TemporaryDirectory() as tmp:
    base_only = Path(tmp)
    for src in MODEL_DIR.iterdir():
        if src.is_dir() or src.name.startswith("adapter_"):
            continue
        (base_only / src.name).symlink_to(src)

    base = AutoModelForCausalLM.from_pretrained(
        base_only,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(base, MODEL_DIR)
    model = model.merge_and_unload()
    model.save_pretrained(MERGED_DIR, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MERGED_DIR)

for fname in ("chat_template.jinja", "generation_config.json"):
    src = MODEL_DIR / fname
    if src.exists():
        shutil.copy2(src, MERGED_DIR / fname)

