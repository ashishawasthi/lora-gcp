# lora-gcp

LoRA / QLoRA fine-tuning experiments, sized to run on free / cheap cloud GPUs (Colab T4, GCP spot, etc.).

## Notebooks

| Notebook | Base model | Dataset | GPU | Notes |
| --- | --- | --- | --- | --- |
| [qlora_t4_llama.ipynb](qlora_t4_llama.ipynb) | `unsloth/Llama-3.2-3B-Instruct` (4-bit) | `yahma/alpaca-cleaned` (1k slice, wrapped as `{"response": ...}` JSON) | Colab T4 (16 GB) | QLoRA run with Unsloth + TRL `SFTTrainer` that teaches the model to emit JSON. ~15 min, trains ~0.75% of params, ~109 MB adapter. BEFORE = prose, AFTER = `{"response": "..."}`. |
| ⚠ [qlora_t4_gemma.ipynb](qlora_t4_gemma.ipynb) | `unsloth/gemma-2-2b-it` (4-bit) | `yahma/alpaca-cleaned` (1k slice, wrapped as `{"response": ...}` JSON) | Colab T4 (16 GB) | **Known-broken on T4** - Gemma 2's attention logit soft-capping is numerically unstable in fp16, and T4 has no bf16, so inference produces token salad. Kept as a documented failure case; use [qlora_gemma4.ipynb](qlora_gemma4.ipynb) for a working Gemma fine-tune. (9B also OOMs on T4: 256k-vocab fp16 embedding table is ~2.4 GB on top of the 4-bit weights.) |
| [qlora_gemma4.ipynb](qlora_gemma4.ipynb) | `unsloth/gemma-4-E4B-it` (4-bit) | `yahma/alpaca-cleaned` (1k slice, wrapped as `{"response": ...}` JSON) | Colab **L4 (24 GB)** baseline; T4 (16 GB) **cannot fit E4B** - swap `model_name` to `gemma-4-E2B-it` if you're stuck on T4. A100/H100 also work. | JSON-output task on Gemma 4 E4B (~4.5B effective). Uses Unsloth's `FastModel` API because Gemma 4 is multimodal (text + image + audio); we LoRA the language tower only via `finetune_vision_layers=False`. See [gemma4-oom-mitigations.md](gemma4-oom-mitigations.md) for the memory budget across GPU classes and why T4 doesn't fit. |

More notebooks (different base models, datasets, ranks, target modules) will be added here as experiments accumulate.

See [gemma4-oom-mitigations.md](gemma4-oom-mitigations.md) for a breakdown of every memory-saving lever pulled in the Gemma 4 notebook to fit it on a free T4 (memory saved per measure + tradeoffs).

## Running

Each notebook is self-contained. Open in Colab, set `Runtime → Change runtime type → T4 GPU`, and run top-to-bottom. The first cell installs its own dependencies.

## Layout conventions

- One notebook per experiment, named `qlora_<gpu>_<model>.ipynb` or `lora_<gpu>_<model>.ipynb`.
- Each notebook ends by saving the trained adapter under `lora_model/` (gitignored).
- Things-to-try notes live in the last cell of each notebook, not in this README.
