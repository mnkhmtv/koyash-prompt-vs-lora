#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MERGED="$ROOT/model/merged"
OUT="$ROOT/model/koyash-f16.gguf"
LLAMA="$ROOT/vendor/llama.cpp"

if [ ! -d "$LLAMA" ]; then
  mkdir -p "$ROOT/vendor"
  git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA"
fi

"$ROOT/.venv/bin/pip" install -q -r "$LLAMA/requirements/requirements-convert_hf_to_gguf.txt"

echo "Converting HF -> GGUF fp16"
"$ROOT/.venv/bin/python" "$LLAMA/convert_hf_to_gguf.py" "$MERGED" \
  --outfile "$OUT" --outtype f16

echo "Done: $OUT"
