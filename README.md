# lora-gcp

LoRA / QLoRA fine-tuning experiments, sized to run on free / cheap cloud GPUs (Colab T4, GCP spot, etc.).

## Notebooks

| Notebook | Base model | Dataset | GPU | Notes |
| --- | --- | --- | --- | --- |
| [qlora_t4_llama.ipynb](qlora_t4_llama.ipynb) | `unsloth/Llama-3.2-3B-Instruct` (4-bit) | `yahma/alpaca-cleaned` (1k slice, wrapped as `{"response": ...}` JSON) | Colab T4 (16 GB) | QLoRA run with Unsloth + TRL `SFTTrainer` that teaches the model to emit JSON. ~15 min, trains ~0.75% of params, ~109 MB adapter. BEFORE = prose, AFTER = `{"response": "..."}`. |
| ⚠ [qlora_t4_gemma.ipynb](qlora_t4_gemma.ipynb) | `unsloth/gemma-2-2b-it` (4-bit) | `yahma/alpaca-cleaned` (1k slice, wrapped as `{"response": ...}` JSON) | Colab T4 (16 GB) | **Known-broken on T4** - Gemma 2's attention logit soft-capping is numerically unstable in fp16, and T4 has no bf16, so inference produces token salad. Kept as a documented failure case; use [qlora_gemma4.ipynb](qlora_gemma4.ipynb) for a working Gemma fine-tune. (9B also OOMs on T4: 256k-vocab fp16 embedding table is ~2.4 GB on top of the 4-bit weights.) |
| [qlora_gemma4.ipynb](qlora_gemma4.ipynb) | `unsloth/gemma-4-E4B-it` (4-bit) | `yahma/alpaca-cleaned` (1k slice, wrapped as `{"response": ...}` JSON) | Colab T4 (16 GB) baseline; bf16-capable (A100/H100/L4, TPU) for speed | Same JSON-output task as the Llama notebook, on Gemma 4 E4B (~4.5B effective). Uses Unsloth's `FastModel` API because Gemma 4 is multimodal (text + image + audio); we LoRA the language tower only via `finetune_vision_layers=False`. No fp16 stability tweaks needed - Gemma 4 dropped Gemma 2's soft-capping. |

More notebooks (different base models, datasets, ranks, target modules) will be added here as experiments accumulate.

## Running

Each notebook is self-contained. Open in Colab, set `Runtime → Change runtime type → T4 GPU`, and run top-to-bottom. The first cell installs its own dependencies.

## Layout conventions

- One notebook per experiment, named `qlora_<gpu>_<model>.ipynb` or `lora_<gpu>_<model>.ipynb`.
- Each notebook ends by saving the trained adapter under `lora_model/` (gitignored).
- Things-to-try notes live in the last cell of each notebook, not in this README.
