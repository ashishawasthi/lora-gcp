# lora-gcp

LoRA / QLoRA fine-tuning experiments, sized to run on free / cheap cloud GPUs (Colab T4, GCP spot, etc.).

## Notebooks

| Notebook | Base model | Dataset | GPU | Notes |
| --- | --- | --- | --- | --- |
| [qlora_t4_llama.ipynb](qlora_t4_llama.ipynb) | `unsloth/Llama-3.2-3B-Instruct` (4-bit) | `yahma/alpaca-cleaned` (1k slice, wrapped as `{"response": ...}` JSON) | Colab T4 (16 GB) | QLoRA run with Unsloth + TRL `SFTTrainer` that teaches the model to emit JSON. ~15 min, trains ~0.75% of params, ~109 MB adapter. BEFORE = prose, AFTER = `{"response": "..."}`. |

More notebooks (different base models, datasets, ranks, target modules) will be added here as experiments accumulate.

## Running

Each notebook is self-contained. Open in Colab, set `Runtime → Change runtime type → T4 GPU`, and run top-to-bottom. The first cell installs its own dependencies.

## Layout conventions

- One notebook per experiment, named `qlora_<gpu>_<model>.ipynb` or `lora_<gpu>_<model>.ipynb`.
- Each notebook ends by saving the trained adapter under `lora_model/` (gitignored).
- Things-to-try notes live in the last cell of each notebook, not in this README.
