# Gemma 4 E4B on T4 - OOM mitigations

This notebook ([qlora_gemma4.ipynb](qlora_gemma4.ipynb)) trains Gemma 4 E4B (~8B total params, ~4.5B effective) with QLoRA on a free Colab T4 (14.56 GB VRAM). Every memory lever this stack offers is pulled - removing any single one of the high-impact measures will OOM.

## Why the budget is tight

Gemma 4 E4B's static footprint dominates VRAM before activations even start, because the Per-Layer Embeddings (PLE) cannot be 4-bit quantized:

| Component                                                | Approx size | 4-bit-able? |
| -------------------------------------------------------- | ----------- | ----------- |
| 4.5B linear params (NF4 4-bit)                           | ~2.5 GB     | yes         |
| ~3.5B PLE + 262k-vocab main embedding (fp16)             | ~7 GB       | **no** (`bnb` only quantizes `nn.Linear`, not `nn.Embedding`) |
| LoRA adapter + grads + adamw_8bit state (36.7M trainable) | ~370 MB    | n/a         |
| CUDA / cuBLAS / bnb dequant workspaces                   | ~2 GB       | n/a         |
| Activations (bs=1, seq=1024, gradient-checkpointed)      | ~0.5-1 GB   | n/a         |
| **Total**                                                | **~12-13 GB** of 14.56 GB | |

The fp16 PLE column is what makes Gemma 4 E4B specifically hard on T4. A normal-vocab 4B Llama would have ~1.5 GB total embeddings; E4B has ~7 GB.

## Mitigations applied (ordered by memory saved)

| #  | Measure                                                          | Notebook location                       | Memory saved (vs not doing it) | Tradeoff                                                                                                  |
| -- | ---------------------------------------------------------------- | --------------------------------------- | ------------------------------ | --------------------------------------------------------------------------------------------------------- |
| 1  | **LoRA** (vs full fine-tune)                                     | cell 6 (`FastModel.get_peft_model`)     | ~30-50 GB                       | only 0.46% of params train; long-tail tasks may need higher rank or full fine-tune                        |
| 2  | **NF4 4-bit quantization of linears** (`load_in_4bit=True`)      | cell 4 (`FastModel.from_pretrained`)    | ~7.5 GB (9 GB fp16 → 2.5 GB NF4) | bnb dequant kernel runs in every forward; slight quality loss vs fp16                                     |
| 3  | **Gradient checkpointing** (`use_gradient_checkpointing="unsloth"`) | cell 6                                | ~1-2 GB on activations          | ~30% slower per step - activations are recomputed during backward                                         |
| 4  | **`MAX_SEQ_LENGTH = 1024`** (down from 2048)                     | cell 4                                  | ~0.5-1 GB on activations        | examples >1024 tokens get truncated - some alpaca outputs are longer and may lose the closing JSON brace |
| 5  | **`per_device_train_batch_size = 1`** (down from 2)              | cell 12                                 | ~0.5-1 GB on activations        | each optimizer step sees less data per device; wall-clock per epoch is slower                            |
| 6  | **`gradient_accumulation_steps = 8`** (up from 4)                | cell 12                                 | 0 directly                      | compensates for `bs=1` to keep effective batch = 8; not memory itself but the price of #5                |
| 7  | **`gc.collect() + torch.cuda.empty_cache()` before model load**  | cell 4                                  | variable - reclaims whatever previous failed loads left behind (can be GBs in a long Colab session) | ~50-100 ms pause; harmless                                                                                |
| 8  | **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**           | cell 4 (set BEFORE `import torch`)      | ~200-500 MB on allocator fragmentation | slightly different allocator behavior; rare edge cases with `torch.compile`                            |
| 9  | **`adamw_8bit` optimizer** (vs fp32 AdamW)                       | cell 12 (`optim="adamw_8bit"`)          | ~220 MB on optimizer state      | minor numerical accuracy hit; rare instability on very small LoRA ranks                                   |
| 10 | **`finetune_vision_layers = False`** (text-only LoRA)            | cell 6                                  | ~100-200 MB on vision LoRA + grads + opt state | model can no longer be fine-tuned on image inputs without re-attaching vision LoRA                       |
| 11 | **`device_map = {"": 0}`**                                       | cell 4                                  | 0 directly                      | forces everything on GPU 0 - prevents the fp16 embedding from being silently CPU-offloaded (which causes a `cpu/cuda` device-mismatch at inference, not an OOM, but blocks the whole pipeline) |
| 12 | **Model choice: E4B** (vs 26B-A4B / 31B)                         | cell 4 (`model_name`)                   | ~10+ GB vs the next size up     | smaller model, lower ceiling on capability                                                                |
| 13 | **`lora_dropout = 0, bias = "none"`**                            | cell 6                                  | <50 MB (no dropout mask buffers, no bias grads) | less regularization; LoRA can't adapt biases. Mostly a speed knob, listed for completeness               |
| 14 | **`packing = False`**                                            | cell 12                                 | 0 (slight padding waste)        | Unsloth's gradient checkpointing path is incompatible with packing - this is a constraint, not a savings  |

## If these aren't enough

If the notebook still OOMs after all of the above, the remaining knobs in order of severity:

- **Drop to `unsloth/gemma-4-E2B-it`**: E2B's PLE is ~half the size of E4B's (~3.5 GB instead of ~7 GB), reclaiming several GB. Smallest code change, smallest capability loss.
- **Restrict LoRA to attention only** (`finetune_mlp_modules = False`): cuts trainable params and grads roughly in half (~18M instead of ~36M). Quality on instruction-following can suffer noticeably.
- **Drop `MAX_SEQ_LENGTH` to 512**: another ~500 MB on activations, but now ~half of alpaca examples get truncated.
- **Move to a 24 GB+ GPU (A10, L4, RTX 4090, A100)**: undoes the need for #4, #5, #6 and you can run `bs=2, seq=2048` cleanly.

## Source data

The memory estimates are derived from:
- 4-bit NF4 cost: ~0.55 bytes/param including per-block fp16 scales (block size 64)
- fp16 embeddings: vocab × hidden × 2 bytes, no quantization
- LoRA: `r × (in_features + out_features)` per targeted projection, ×2 bytes (fp16)
- `adamw_8bit` state: ~6 bytes per trainable param (m + v at 8-bit + fp32 master)
- Activations: linear in `batch × seq_len × hidden × num_layers`, divided by ~5-10× under Unsloth gradient checkpointing
- CUDA/bnb overhead measured empirically from `torch.cuda.max_memory_reserved()` in cell 13

For a broader walk-through of what loads to GPU memory during a LoRA training run, see the analysis notes in [`/Users/ashishawasthi/.claude/plans/explain-what-all-gets-twinkling-turtle.md`](/Users/ashishawasthi/.claude/plans/explain-what-all-gets-twinkling-turtle.md) (this session's planning artifact).
