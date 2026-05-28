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
| Activations (bs=1, seq=512, gradient-checkpointed)       | ~0.3-0.5 GB | n/a         |
| **Total**                                                | **~12-12.5 GB** of 14.56 GB | |

The fp16 PLE column is what makes Gemma 4 E4B specifically hard on T4. A normal-vocab 4B Llama would have ~1.5 GB total embeddings; E4B has ~7 GB.

## Mitigations applied (ordered by memory saved)

| #  | Measure                                                          | Notebook location                       | Memory saved (vs not doing it) | Tradeoff                                                                                                  |
| -- | ---------------------------------------------------------------- | --------------------------------------- | ------------------------------ | --------------------------------------------------------------------------------------------------------- |
| 1  | **LoRA** (vs full fine-tune)                                     | cell 6 (`FastModel.get_peft_model`)     | ~30-50 GB                       | only 0.46% of params train; long-tail tasks may need higher rank or full fine-tune                        |
| 2  | **NF4 4-bit quantization of linears** (`load_in_4bit=True`)      | cell 4 (`FastModel.from_pretrained`)    | ~7.5 GB (9 GB fp16 → 2.5 GB NF4) | bnb dequant kernel runs in every forward; slight quality loss vs fp16                                     |
| 3  | **Gradient checkpointing** (`use_gradient_checkpointing="unsloth"`) | cell 6                                | ~1-2 GB on activations          | ~30% slower per step - activations are recomputed during backward                                         |
| 4  | **`MAX_SEQ_LENGTH = 512`** (down from 2048)                      | cell 4                                  | ~1-1.5 GB on activations        | examples >512 tokens get truncated - a meaningful share of alpaca outputs are longer and may lose the closing JSON brace |
| 5  | **`per_device_train_batch_size = 1`** (down from 2)              | cell 12                                 | ~0.5-1 GB on activations        | each optimizer step sees less data per device; wall-clock per epoch is slower                            |
| 6  | **`gradient_accumulation_steps = 8`** (up from 4)                | cell 12                                 | 0 directly                      | compensates for `bs=1` to keep effective batch = 8; not memory itself but the price of #5                |
| 7  | **`gc.collect() + torch.cuda.empty_cache()` before model load**  | cell 4                                  | variable - reclaims whatever previous failed loads left behind (can be GBs in a long Colab session) | ~50-100 ms pause; harmless                                                                                |
| 8  | **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**           | cell 4 (set BEFORE `import torch`)      | ~200-500 MB on allocator fragmentation | slightly different allocator behavior; rare edge cases with `torch.compile`                            |
| 9  | **`paged_adamw_8bit` optimizer** (vs fp32 AdamW)                 | cell 12 (`optim="paged_adamw_8bit"`)    | ~370 MB on optimizer state (220 MB from 8-bit quant + ~150 MB from CPU-paging the fp32 master weights) | minor numerical accuracy hit; ~10-15% wall-clock slowdown from CPU↔GPU paging traffic                     |
| 10 | **`finetune_vision_layers = False`** (text-only LoRA)            | cell 6                                  | ~100-200 MB on vision LoRA + grads + opt state | model can no longer be fine-tuned on image inputs without re-attaching vision LoRA                       |
| 11 | **`device_map = {"": 0}`**                                       | cell 4                                  | 0 directly                      | forces everything on GPU 0 - prevents the fp16 embedding from being silently CPU-offloaded (which causes a `cpu/cuda` device-mismatch at inference, not an OOM, but blocks the whole pipeline) |
| 12 | **Model choice: E4B** (vs 26B-A4B / 31B)                         | cell 4 (`model_name`)                   | ~10+ GB vs the next size up     | smaller model, lower ceiling on capability                                                                |
| 13 | **`lora_dropout = 0, bias = "none"`**                            | cell 6                                  | <50 MB (no dropout mask buffers, no bias grads) | less regularization; LoRA can't adapt biases. Mostly a speed knob, listed for completeness               |
| 14 | **`packing = False`**                                            | cell 12                                 | 0 (slight padding waste)        | Unsloth's gradient checkpointing path is incompatible with packing - this is a constraint, not a savings  |

## If these aren't enough

If the notebook still OOMs after all of the above, the remaining knobs in order of severity:

- **Drop to `unsloth/gemma-4-E2B-it`**: E2B's PLE is ~half the size of E4B's (~3.5 GB instead of ~7 GB), reclaiming several GB. Smallest code change, smallest capability loss.
- **Restrict LoRA to attention only** (`finetune_mlp_modules = False`): cuts trainable params and grads roughly in half (~18M instead of ~36M). Quality on instruction-following can suffer noticeably.
- **Drop `MAX_SEQ_LENGTH` to 256**: another ~150 MB on activations, but most alpaca examples get truncated mid-output - the JSON brace usually survives the user turn but the assistant turn may not.
- **Move to a 24 GB+ GPU (A10, L4, RTX 4090, A100)**: undoes the need for #4, #5, #6 and you can run `bs=2, seq=2048` cleanly.

## Techniques deliberately NOT used (and why)

A reasonable practitioner will ask "what about ZeRO? FSDP? FA2?" These are popular memory-saving levers that are either incompatible with our setup, redundant given what's already applied, or only meaningful at scales we're not operating at. Listed here for completeness so the choice is explicit rather than accidental.

| Category               | Technique                                                  | Why not used here                                                                                                                                                          |
| ---------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Multi-GPU parallelism  | Distributed Data Parallel (DDP)                            | PyTorch's standard multi-GPU replication: each GPU holds a **full** copy of the model and processes different batches; gradients are all-reduced across ranks (and bucketed for communication efficiency). Needs ≥2 GPUs, and provides **zero memory savings** even when available - it only scales throughput. Useless for our OOM problem regardless of GPU count. |
| Multi-GPU parallelism  | DeepSpeed ZeRO Stage 1/2/3                                 | Needs ≥2 GPUs; with N=1 the sharding stages collapse to no-ops. ZeRO-3 also can't cleanly shard bnb 4-bit tensors (in-place quantization breaks the sharding contract).    |
| Multi-GPU parallelism  | PyTorch FSDP / FSDP buckets (Fully Sharded Data Parallel)  | PyTorch-native equivalent of ZeRO-3 - shards model parameters into communication "buckets" (`auto_wrap_policy` groups params; each bucket is all-gathered just-in-time for forward/backward and freed after). With N=1 GPU there's no peer rank to bucket *across*, so the all-gather is a self-loop and FSDP collapses to a no-op. (Bucket sizing only matters in multi-GPU setups, where it trades comm latency vs. memory peak.) |
| Multi-GPU parallelism  | Pipeline parallelism                                       | Splits the model's layers across GPUs; needs ≥2 GPUs.                                                                                                                      |
| Multi-GPU parallelism  | Tensor parallelism                                         | Splits individual layer matmuls across GPUs; needs ≥2 GPUs.                                                                                                                |
| Offloading             | DeepSpeed ZeRO-Offload                                     | The single-GPU ZeRO variant (offloads optimizer state to CPU RAM). Our `adamw_8bit` state is only ~220 MB - marginal savings at ~30% wall-clock cost.                       |
| Offloading             | DeepSpeed ZeRO-Infinity                                    | Offloads to NVMe; complex setup, very slow; only meaningful for full fine-tuning of huge models.                                                                            |
| Offloading             | `paged_adamw_8bit` (bnb-native CPU paging)                 | Lightweight alternative to ZeRO-Offload, no DeepSpeed dependency. Listed under "If these aren't enough" as a tertiary fallback - not applied by default since we already fit. |
| Offloading             | HuggingFace Accelerate auto CPU offload (`device_map="auto"`) | **Explicitly disabled** via `device_map={"": 0}` (see mitigations table row 11) - auto-offloads our fp16 embedding to CPU and causes a `cpu/cuda` device-mismatch crash at inference. |
| Quantization           | AWQ / GPTQ                                                 | Inference-oriented 4-bit schemes; not the QLoRA fine-tuning standard (bnb NF4 is what `peft` and Unsloth integrate with).                                                   |
| Quantization           | 8-bit weight quantization (LLM.int8)                       | Doubles base-weight memory vs NF4 for negligible quality gain on a generative SFT task. Only worth using if you suspect 4-bit hurts quality - rarely the bottleneck.        |
| Quantization           | Activation quantization                                    | Experimental, slow, often unstable on consumer GPUs; no first-class Unsloth integration.                                                                                    |
| Adapter method         | DoRA (Weight-Decomposed LoRA)                              | Slightly better quality at slightly higher memory. Unsloth supports it via `use_dora=True`; skipped to preserve VRAM headroom on T4.                                        |
| Adapter method         | VeRA (Vector-based Random Matrix Adaptation)               | Smaller than LoRA but less mature; not in Unsloth's default path.                                                                                                           |
| Adapter method         | LoftQ initialization                                       | Improves quantized model initialization quality, doesn't save memory. `loftq_config = None` is the standard choice.                                                         |
| Adapter method         | LoRA-FA (frozen A matrix)                                  | Freezes the LoRA A matrix and only trains B - halves trainable params. Not standard in Unsloth and quality drop is usually noticeable.                                      |
| Attention / activation | Flash Attention 2                                          | Not supported on Turing (T4 = sm_75); needs sm_80+ (Ampere). Unsloth's own attention kernels cover this niche on T4. (Xformers 0.0.35 is what Unsloth uses as a fallback path on Turing.) |
| Attention / activation | Selective / Megatron-style activation recomputation        | Finer-grained than Unsloth's gradient checkpointing; marginal improvement; not exposed by Unsloth.                                                                          |
| Attention / activation | CPU activation offloading                                  | Trades CPU RAM + PCIe traffic for VRAM. Gradient checkpointing already saves 1-2 GB more cheaply, with no host-device traffic.                                              |

The pattern across this table: most of what's missing is either **for multi-GPU setups** (sharding/parallelism — irrelevant on single T4) or **redundant with what QLoRA already provides** (4-bit weights + 8-bit optimizer + LoRA + checkpointing covers ~95% of what ZeRO targets, via compression rather than sharding).

## Source data

The memory estimates are derived from:
- 4-bit NF4 cost: ~0.55 bytes/param including per-block fp16 scales (block size 64)
- fp16 embeddings: vocab × hidden × 2 bytes, no quantization
- LoRA: `r × (in_features + out_features)` per targeted projection, ×2 bytes (fp16)
- `adamw_8bit` state: ~6 bytes per trainable param (m + v at 8-bit + fp32 master)
- Activations: linear in `batch × seq_len × hidden × num_layers`, divided by ~5-10× under Unsloth gradient checkpointing
- CUDA/bnb overhead measured empirically from `torch.cuda.max_memory_reserved()` in cell 13

For a broader walk-through of what loads to GPU memory during a LoRA training run, see the analysis notes in [`/Users/ashishawasthi/.claude/plans/explain-what-all-gets-twinkling-turtle.md`](/Users/ashishawasthi/.claude/plans/explain-what-all-gets-twinkling-turtle.md) (this session's planning artifact).
