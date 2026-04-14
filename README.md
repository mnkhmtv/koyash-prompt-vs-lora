# koyash-prompt-vs-lora

Prompt engineering vs LoRA fine-tuning on Qwen3-4B (Koyash cosmetics domain).

## For InnoDataHub

```bash
git clone https://github.com/mnkhmtv/koyash-prompt-vs-lora.git
cd koyash-prompt-vs-lora
```

```bash
pip install -q "transformers>=4.51,<4.56" "peft>=0.13,<0.15" "trl>=0.12,<0.14" "bitsandbytes>=0.43.3,<0.45" "datasets>=2.20,<3.0" "accelerate>=1.3,<2" rouge-score bert-score
```

## Setup

```bash
brew install ollama
ollama serve &
ollama pull qwen3:4b
```

Download `koyash-f16.gguf` and `Modelfile` from [Google Drive](/https://drive.google.com/file/d/1ix1cIcDmcao6KcGTHwf1DrtFNHc-y2Fj/view?usp=sharing) into `model/`, then:

```bash
cd model && ollama create koyash-finetuned -f Modelfile && cd ..
```

## Run

```bash
make run
make interface
```
