# koyash-prompt-vs-lora

to launch you have to install ollama and launch it:
```bash
brew install ollama
ollama serve
ollama pull qwen2.5:32b
```

to test the interface, run:
```bash
make interface
```

to evaluate the models, run:
```bash
make run
```
it creates `metrics_results.csv`, that can be used to aggregate results and analyze them.
