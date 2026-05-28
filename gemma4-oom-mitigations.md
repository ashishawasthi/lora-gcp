# Gemma 4 E4B memory budget (L4 default, T4 attempt history)

The notebook ([qlora_gemma4.ipynb](qlora_gemma4.ipynb)) targets **Colab L4 (24 GB VRAM, bf16-capable)** because the free **T4 (16 GB) cannot reliably fit Gemma 4 E4B for training**, even with every memory-saving lever applied. This doc covers:

1. Why T4 was abandoned (the hard limit).
2. The L4 budget the notebook actually runs against (loose - we have ~10 GB of headroom).
3. The full mitigation table - split into **always-on** (QLoRA core, applied everywhere) and **T4-only fallbacks** (currently NOT in the notebook; resurrect them if you must run on T4 + E2B).
4. Techniques deliberately NOT used (multi-GPU, alternative quantizations, etc.).

## Why T4 was abandoned

T4 has 14.56 GB usable. Gemma 4 E4B's **static** footprint alone reaches ~10.5 GB on T4:

| Component                                                 | Approx size | 4-bit-able? |
| --------------------------------------------------------- | ----------- | ----------- |
| 4.5B linear params (NF4 4-bit)                            | ~2.5 GB     | yes         |
| ~3.5B PLE + 262k-vocab main embedding (fp16)              | ~7 GB       | **no** (`bnb` only quantizes `nn.Linear`, not `nn.Embedding`) |
| LoRA adapter + grads + `paged_adamw_8bit` state (36.7M trainable) | ~0.4 GB | n/a |
| CUDA / cuBLAS / bnb dequant scratch                       | ~2 GB       | n/a         |
| **Static subtotal**                                       | **~10.5 GB** |             |

That leaves ~4 GB on T4 for the **transient** demand during forward/backward:

- bnb 4-bit dequant working tiles (sequence-*independent* — `seq=512` does not help): up to ~200 MB peak per layer for Gemma 4's 2560×10240 MLP linears.
- Activations (bs=1, seq=512, gradient-checkpointed): ~0.3-0.5 GB.
- Gradients + cuBLAS workspaces + first-step optimizer alloc: ~0.5 GB.
- PyTorch allocator fragmentation: 0.2-0.5 GB even with `expandable_segments`.

Combined transient: ~1.5 GB minimum, ~3 GB peak. Adding to the 10.5 GB static, the peak load lands in the **13-13.5 GB range** — within rounding of the 14.56 GB cap. Allocator fragmentation, kernel JIT buffers, or any unrelated allocation tips it over. Repeated empirical attempts:

| Attempt                                          | Reserved before training | OOM during step  | Headroom at OOM |
| ------------------------------------------------ | ------------------------ | ---------------- | --------------- |
| bs=2, seq=2048, adamw_8bit                       | -                        | yes              | 0 (basically)   |
| bs=1, seq=2048, adamw_8bit                       | -                        | yes              | ~2 MB free      |
| bs=1, seq=1024, adamw_8bit                       | -                        | yes              | ~2 MB free      |
| bs=1, seq=512, paged_adamw_8bit (most aggressive) | 10.45 GB               | yes              | 27 MB free      |

Each step shrank the activation column but ran into the unmovable bnb dequant + PLE static cost. **T4 simply has no margin for E4B + LoRA training.** The notebook now targets L4 instead.

If you must stay on T4: swap `model_name` in cell 4 to `unsloth/gemma-4-E2B-it` (E2B's PLE is ~3.5 GB vs E4B's ~7 GB), and you'll have ~3-4 GB of headroom back.

## The L4 budget (what the notebook now runs against)

L4 has 24 GB. Same static numbers as above plus the L4-baseline settings give:

| Component                                              | Approx size on L4 |
| ------------------------------------------------------ | ----------------- |
| Static (4-bit weights + fp16 PLE + LoRA + workspaces) | ~10.5 GB          |
| Activations (bs=2, seq=2048, checkpointed)            | ~1.5-2.5 GB       |
| bnb dequant peak                                      | ~0.5-1 GB         |
| Gradients + adamw_8bit + cuBLAS                       | ~0.5 GB           |
| **Total peak**                                        | **~13-14 GB**     |

**~10 GB of headroom** on L4. Could comfortably go `bs=4` or `seq=4096`. Code stays clean.

## Mitigations table

Split into two categories:

### Always-on (QLoRA core - applied on every GPU class)

| #  | Measure                                                            | Notebook location                       | Memory saved (vs not doing it) | Tradeoff |
| -- | ------------------------------------------------------------------ | --------------------------------------- | ------------------------------ | -------- |
| 1  | **LoRA** (vs full fine-tune)                                       | cell 6 (`FastModel.get_peft_model`)     | ~30-50 GB                       | only 0.46% of params train; long-tail tasks may need higher rank or full FT |
| 2  | **NF4 4-bit quantization of linears** (`load_in_4bit=True`)        | cell 4 (`FastModel.from_pretrained`)    | ~7.5 GB (9 GB fp16 → 2.5 GB NF4) | bnb dequant kernel runs each forward; slight quality loss vs fp16 |
| 3  | **Gradient checkpointing** (`use_gradient_checkpointing="unsloth"`) | cell 6                                | ~1-2 GB on activations          | ~30% slower per step - activations are recomputed during backward |
| 4  | **`adamw_8bit` optimizer**                                         | cell 12 (`optim="adamw_8bit"`)          | ~220 MB on optimizer state      | minor numerical accuracy hit |
| 5  | **`gc.collect() + torch.cuda.empty_cache()` before load**          | cell 4                                  | variable - reclaims leftover state from earlier failed loads | ~50-100 ms pause |
| 6  | **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**             | cell 4 (set BEFORE `import torch`)      | ~200-500 MB on allocator fragmentation | slightly different allocator behavior; rare edge cases with `torch.compile` |
| 7  | **`finetune_vision_layers = False`** (text-only LoRA)              | cell 6                                  | ~100-200 MB on vision LoRA + grads + opt state | model can't be fine-tuned on image inputs without re-attaching vision LoRA |
| 8  | **`device_map = {"": 0}`**                                         | cell 4                                  | 0 directly                      | pins all modules on GPU 0 - prevents the fp16 embedding from being silently CPU-offloaded (which causes a `cpu/cuda` device-mismatch at inference) |
| 9  | **Model choice: E4B** (vs 26B-A4B / 31B)                           | cell 4 (`model_name`)                   | ~10+ GB vs the next size up     | smaller model, lower capability ceiling |
| 10 | **`lora_dropout = 0, bias = "none"`**                              | cell 6                                  | <50 MB                          | less regularization; LoRA can't adapt biases. Mostly a speed knob, listed for completeness |

### T4-only fallbacks (NOT in the L4 notebook; reach for these only on T4 + E2B)

| #  | Measure                                          | Memory saved | Tradeoff                                                                                          |
| -- | ------------------------------------------------ | ------------ | ------------------------------------------------------------------------------------------------- |
| F1 | **`MAX_SEQ_LENGTH = 512`** (down from 2048)      | ~1-1.5 GB    | examples >512 tokens get truncated - a meaningful share of alpaca outputs are longer             |
| F2 | **`per_device_train_batch_size = 1`** (down from 2) | ~0.5-1 GB | each optimizer step sees less data per device; wall-clock per epoch is slower                    |
| F3 | **`gradient_accumulation_steps = 8`** (up from 4) | 0 directly  | compensates for F2 to keep effective batch = 8; the price of F2, not a savings                  |
| F4 | **`paged_adamw_8bit`** (vs `adamw_8bit`)         | ~150 MB      | ~10-15% wall-clock slowdown from CPU↔GPU paging traffic                                          |

Even with F1-F4 + E4B, T4 OOMs. F1-F4 fit only after also swapping the model to E2B.

## If even L4 isn't enough (unlikely)

Quick escalation if you push E4B to long contexts or large batches:

- **Drop to `unsloth/gemma-4-E2B-it`**: E2B's PLE is ~half the size (~3.5 GB instead of ~7 GB).
- **Restrict LoRA to attention only** (`finetune_mlp_modules = False`): cuts trainable params and grads roughly in half (~18M instead of ~36M). Quality on instruction-following can suffer noticeably.
- **Move to A100 (40 GB)**: undoes any need to compromise on batch / seq / optimizer.

## Techniques deliberately NOT used (and why)

A reasonable practitioner will ask "what about ZeRO? FSDP? DDP? FA2?" These are popular memory-saving levers that are either incompatible with our setup, redundant given what's already applied, or only meaningful at scales we're not operating at. Listed for completeness so the choice is explicit rather than accidental.

| Category               | Technique                                                  | Why not used here                                                                                                                                                          |
| ---------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Multi-GPU parallelism  | Distributed Data Parallel (DDP)                            | PyTorch's standard multi-GPU replication: each GPU holds a **full** copy of the model and processes different batches; gradients are all-reduced across ranks (and bucketed for communication efficiency). Needs ≥2 GPUs, and provides **zero memory savings** even when available - it only scales throughput. Useless for our OOM problem regardless of GPU count. |
| Multi-GPU parallelism  | DeepSpeed ZeRO Stage 1/2/3                                 | Needs ≥2 GPUs; with N=1 the sharding stages collapse to no-ops. ZeRO-3 also can't cleanly shard bnb 4-bit tensors (in-place quantization breaks the sharding contract).    |
| Multi-GPU parallelism  | PyTorch FSDP / FSDP buckets (Fully Sharded Data Parallel)  | PyTorch-native equivalent of ZeRO-3 - shards model parameters into communication "buckets" (`auto_wrap_policy` groups params; each bucket is all-gathered just-in-time for forward/backward and freed after). With N=1 GPU there's no peer rank to bucket *across*, so the all-gather is a self-loop and FSDP collapses to a no-op. (Bucket sizing only matters in multi-GPU setups, where it trades comm latency vs. memory peak.) |
| Multi-GPU parallelism  | Pipeline parallelism                                       | Splits the model's layers across GPUs; needs ≥2 GPUs.                                                                                                                      |
| Multi-GPU parallelism  | Tensor parallelism                                         | Splits individual layer matmuls across GPUs; needs ≥2 GPUs.                                                                                                                |
| Offloading             | DeepSpeed ZeRO-Offload                                     | The single-GPU ZeRO variant (offloads optimizer state to CPU RAM). Marginal savings vs `paged_adamw_8bit` at ~30% wall-clock cost; not worth the DeepSpeed dependency.    |
| Offloading             | DeepSpeed ZeRO-Infinity                                    | Offloads to NVMe; complex setup, very slow; only meaningful for full fine-tuning of huge models.                                                                            |
| Offloading             | HuggingFace Accelerate auto CPU offload (`device_map="auto"`) | **Explicitly disabled** via `device_map={"": 0}` (see always-on row 8) - auto-offloads our fp16 embedding to CPU and causes a `cpu/cuda` device-mismatch crash at inference. |
| Quantization           | AWQ / GPTQ                                                 | Inference-oriented 4-bit schemes; not the QLoRA fine-tuning standard (bnb NF4 is what `peft` and Unsloth integrate with).                                                   |
| Quantization           | 8-bit weight quantization (LLM.int8)                       | Doubles base-weight memory vs NF4 for negligible quality gain on a generative SFT task.                                                                                     |
| Quantization           | Activation quantization                                    | Experimental, slow, often unstable on consumer GPUs; no first-class Unsloth integration.                                                                                    |
| Adapter method         | DoRA (Weight-Decomposed LoRA)                              | Slightly better quality at slightly higher memory. Unsloth supports it via `use_dora=True`; skipped to preserve VRAM headroom.                                              |
| Adapter method         | VeRA (Vector-based Random Matrix Adaptation)               | Smaller than LoRA but less mature; not in Unsloth's default path.                                                                                                           |
| Adapter method         | LoftQ initialization                                       | Improves quantized model initialization quality, doesn't save memory. `loftq_config = None` is the standard choice.                                                         |
| Adapter method         | LoRA-FA (frozen A matrix)                                  | Freezes the LoRA A matrix and only trains B - halves trainable params. Not standard in Unsloth and quality drop is usually noticeable.                                      |
| Attention / activation | Flash Attention 2                                          | Not supported on Turing (T4 = sm_75); needs sm_80+ (Ampere). Available on L4 (Ada, sm_89) - Unsloth picks it up automatically.                                              |
| Attention / activation | Selective / Megatron-style activation recomputation        | Finer-grained than Unsloth's gradient checkpointing; marginal improvement; not exposed by Unsloth.                                                                          |
| Attention / activation | CPU activation offloading                                  | Trades CPU RAM + PCIe traffic for VRAM. Gradient checkpointing already saves 1-2 GB more cheaply, with no host-device traffic.                                              |

The pattern across this table: most of what's missing is either **for multi-GPU setups** (sharding/parallelism — irrelevant on single-GPU) or **redundant with what QLoRA already provides** (4-bit weights + 8-bit optimizer + LoRA + checkpointing covers ~95% of what ZeRO targets, via compression rather than sharding).

## Source data

The memory estimates are derived from:
- 4-bit NF4 cost: ~0.55 bytes/param including per-block fp16 scales (block size 64)
- fp16 embeddings: vocab × hidden × 2 bytes, no quantization
- LoRA: `r × (in_features + out_features)` per targeted projection, ×2 bytes (fp16)
- `adamw_8bit` state: ~6 bytes per trainable param (m + v at 8-bit + fp32 master)
- Activations: linear in `batch × seq_len × hidden × num_layers`, divided by ~5-10× under Unsloth gradient checkpointing
- CUDA/bnb overhead measured empirically from `torch.cuda.max_memory_reserved()` in cell 13 and the OOM error reports

For a broader walk-through of what loads to GPU memory during a LoRA training run, see the analysis notes in [`/Users/ashishawasthi/.claude/plans/explain-what-all-gets-twinkling-turtle.md`](/Users/ashishawasthi/.claude/plans/explain-what-all-gets-twinkling-turtle.md) (this session's planning artifact).
