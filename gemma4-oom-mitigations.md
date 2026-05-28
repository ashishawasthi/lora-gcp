# Gemma 4 E4B memory budget (A100 default; T4 and L4 attempt history)

The notebook ([qlora_gemma4.ipynb](qlora_gemma4.ipynb)) targets **A100 (40 GB)** because both Colab T4 (16 GB) and Colab L4 (22 GB) **OOM for Gemma 4 E4B + LoRA training**, even with every memory-saving lever applied. This doc covers:

1. Why T4 and L4 were both abandoned (the hard limits, with empirical traces).
2. The A100 budget the notebook actually runs against (loose - plenty of headroom).
3. The full mitigation table - split into **always-on** (QLoRA core, applied everywhere) and **T4/L4-only fallbacks** (currently NOT in the notebook; resurrect them if you swap to E2B).
4. Techniques deliberately NOT used (multi-GPU, alternative quantizations, etc.).

For an interactive what-fits-where calculator, see [gemma4-memory-calculator.html](gemma4-memory-calculator.html).

## Why T4 and L4 were abandoned

Two different walls. Both end at OOM but for **different dominant components**.

### T4: PLE-dominated static footprint

T4 has 14.56 GB usable. Gemma 4 E4B's static footprint alone reaches ~10.5 GB:

| Component                                                 | Approx size | 4-bit-able? |
| --------------------------------------------------------- | ----------- | ----------- |
| 4.5B linear params (NF4 4-bit)                            | ~2.5 GB     | yes         |
| ~3.5B PLE + 262k-vocab main embedding (fp16)              | ~7 GB       | **no** (`bnb` only quantizes `nn.Linear`, not `nn.Embedding`) |
| LoRA adapter + grads + `paged_adamw_8bit` state           | ~0.4 GB     | n/a         |
| CUDA / cuBLAS / bnb dequant scratch                       | ~2 GB       | n/a         |
| **Static subtotal**                                       | **~10.5 GB** |             |

That leaves only ~4 GB for transient demand on T4 — not enough for the bnb dequant working tiles + activations + first-step optimizer alloc.

### L4: bf16 cross-entropy logits-dominated transient

L4 (Colab allocation is 22 GB, not always 24) has plenty of static room. The killer is the **cross-entropy loss computation** during the backward pass:

- Cross-entropy needs the full logits tensor: `bs × seq × vocab × 2 bytes` (bf16) = `2 × 1024 × 262144 × 2 ≈ 1 GB` for one forward pass.
- During backward, intermediates for `log_softmax`, the gradient w.r.t. logits, and the gradient w.r.t. embeddings all coexist briefly. Real peak is ~1.5-2 GB just for the loss + its backward.
- This is **sequence-dependent** but `vocab`-dominated. Halving `seq` only halves this term; the 262k-vocab factor doesn't budge.
- Length-sort doesn't help (the loss tensor size is determined by `max_seq_length`, not actual content length).

Combined with the ~10.5 GB static + activations + dequant + cuBLAS, L4 lands at ~22 GB peak — within rounding of its 22.03 GB cap.

### Empirical attempts

| Attempt                                                            | GPU class | Reserved before training | OOM at step | Headroom at OOM | Dominant cause of failure                  |
| ------------------------------------------------------------------ | --------- | ------------------------ | ----------- | --------------- | ------------------------------------------ |
| bs=2, seq=2048, adamw_8bit                                         | T4 16 GB  | -                        | step 0      | 0 (basically)   | PLE static + activations                   |
| bs=1, seq=2048, adamw_8bit                                         | T4 16 GB  | -                        | step 0      | ~2 MB free      | PLE static + dequant peak                  |
| bs=1, seq=1024, adamw_8bit                                         | T4 16 GB  | -                        | step 0      | ~2 MB free      | PLE static + dequant peak                  |
| bs=1, seq=512, paged_adamw_8bit (most aggressive)                  | T4 16 GB  | 10.45 GB                 | step 0      | 27 MB free      | PLE static + minimal activations           |
| bs=2, seq=2048, adamw_8bit, **no length sort**                     | L4 22 GB  | -                        | **step 9** (mid-training) | 49 MB free      | unsorted long-outlier batch activation spike |
| bs=2, seq=1024, adamw_8bit, **with length sort, all mitigations**  | L4 22 GB  | -                        | **step 2** (early)        | 63 MB free      | **bf16 cross-entropy logits over 262k vocab (522 MB transient)** - even shortest sorted batches hit this |
| bs=2, seq=1024, adamw_8bit, with length sort                       | A100 40 GB| -                        | **completes** | ~17 GB free    | n/a - fits comfortably                     |

**Conclusion:** T4 is unreachable for E4B without architecture-level intervention (would need to quantize the embeddings, which bnb doesn't support). L4 is closer but still loses to the loss-tensor transient peak; even chunked cross-entropy (`cut_cross_entropy`, if your Unsloth version exposes it) might bridge the gap, but stock SFTTrainer + Gemma 4 doesn't. **A100 40 GB is the smallest GPU class verified working** for E4B + LoRA on this notebook.

If you must stay on L4 or T4: swap `model_name` in cell 4 to `unsloth/gemma-4-E2B-it`. E2B's PLE is ~3.5 GB (vs E4B's ~7 GB) and its vocab embedding factor is identical so the loss tensor is the same ~1 GB, but everything else shrinks — comfortable on L4, fits T4 with the T4-only fallbacks below.

## The A100 budget (what the notebook now runs against)

A100 40 GB has 39.5 GB usable. The notebook's L4-era settings (bs=2, seq=1024, length-sort) leave generous headroom:

| Component                                              | Approx size on A100 |
| ------------------------------------------------------ | ------------------- |
| Static (4-bit weights + fp16 PLE + LoRA + workspaces) | ~10.5 GB            |
| Activations (bs=2, seq=1024, checkpointed)            | ~1-2 GB             |
| bnb dequant peak                                      | ~0.5-1 GB           |
| **Cross-entropy logits (bs × seq × 262k × 2 bytes)**  | **~1-2 GB**         |
| Gradients + adamw_8bit + cuBLAS                       | ~0.5 GB             |
| PyTorch allocator overhead                            | ~1 GB               |
| **Total peak**                                        | **~14-17 GB**       |

**~22-25 GB of headroom** on A100 40 GB. You could comfortably go `bs=4, seq=2048` (would need ~25 GB peak) or `bs=8, seq=2048` on A100 80 GB / H100.

## Mitigations table

Split into two categories: what runs on A100 by default, and what you'd add on smaller hardware (with E2B).

### Always-on (QLoRA core - applied on every GPU class)

| #  | Measure                                                            | Notebook location                       | Memory saved (vs not doing it) | Tradeoff |
| -- | ------------------------------------------------------------------ | --------------------------------------- | ------------------------------ | -------- |
| 1  | **LoRA** (vs full fine-tune)                                       | cell 6 (`FastModel.get_peft_model`)     | ~30-50 GB                       | only 0.46% of params train; long-tail tasks may need higher rank or full FT |
| 2  | **NF4 4-bit quantization of linears** (`load_in_4bit=True`)        | cell 4 (`FastModel.from_pretrained`)    | ~7.5 GB (9 GB fp16 → 2.5 GB NF4) | bnb dequant kernel runs each forward; slight quality loss vs fp16 |
| 3  | **Gradient checkpointing** (`use_gradient_checkpointing="unsloth"`) | cell 6                                | ~1-2 GB on activations          | ~30% slower per step - activations are recomputed during backward |
| 4  | **`adamw_8bit` optimizer**                                         | cell 12 (`optim="adamw_8bit"`)          | ~220 MB on optimizer state      | minor numerical accuracy hit |
| 5  | **`gc.collect() + torch.cuda.empty_cache()` before load**          | cell 4                                  | variable - reclaims leftover state from earlier failed loads | ~50-100 ms pause |
| 6  | **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**             | cell 4 (set BEFORE `import torch`)      | ~200-500 MB on allocator fragmentation | slightly different allocator behavior; rare edge cases with `torch.compile` |
| 6a | **`MAX_SEQ_LENGTH = 1024`** (not 2048)                             | cell 4                                  | ~half of activations and the bf16 loss tensor (~500 MB) vs seq=2048 | examples >1024 tokens truncate - rare in alpaca-cleaned but real for long-output rows. On A100 you can raise back to 2048 |
| 6b | **Pre-sort dataset by length** (replacement for `group_by_length`) | cell 8 (`dataset.sort` after `dataset.map`) | up to ~3 GB on worst-case batches; converts unpredictable mid-training OOMs into a clean step-0 fit check | samples train in length order which can mildly bias gradient noise per step. Implemented as a manual sort because transformers 5.x removed `group_by_length` from `TrainingArguments`. |
| 7  | **`finetune_vision_layers = False`** (text-only LoRA)              | cell 6                                  | ~100-200 MB on vision LoRA + grads + opt state | model can't be fine-tuned on image inputs without re-attaching vision LoRA |
| 8  | **`device_map = {"": 0}`**                                         | cell 4                                  | 0 directly                      | pins all modules on GPU 0 - prevents the fp16 embedding from being silently CPU-offloaded (which causes a `cpu/cuda` device-mismatch at inference) |
| 9  | **Model choice: E4B** (vs 26B-A4B / 31B)                           | cell 4 (`model_name`)                   | ~10+ GB vs the next size up     | smaller model, lower capability ceiling |
| 10 | **`lora_dropout = 0, bias = "none"`**                              | cell 6                                  | <50 MB                          | less regularization; LoRA can't adapt biases. Mostly a speed knob, listed for completeness |

### T4/L4-only fallbacks (NOT in the A100 notebook; reach for these only with E2B on smaller GPUs)

| #  | Measure                                          | Memory saved | Tradeoff                                                                                          |
| -- | ------------------------------------------------ | ------------ | ------------------------------------------------------------------------------------------------- |
| F1 | **`MAX_SEQ_LENGTH = 512`** (down from 1024)      | ~500 MB on activations + ~500 MB on loss tensor | examples >512 tokens get truncated - a meaningful share of alpaca outputs are longer |
| F2 | **`per_device_train_batch_size = 1`** (down from 2) | ~0.5-1 GB on activations + ~500 MB on loss tensor | each optimizer step sees less data per device; wall-clock per epoch is slower |
| F3 | **`gradient_accumulation_steps = 8`** (up from 4) | 0 directly  | compensates for F2 to keep effective batch = 8; the price of F2, not a savings                  |
| F4 | **`paged_adamw_8bit`** (vs `adamw_8bit`)         | ~150 MB      | ~10-15% wall-clock slowdown from CPU↔GPU paging traffic                                          |
| F5 | **Swap model to E2B**                            | ~3.5 GB (PLE)| smaller model, lower capability ceiling. Required for reliable L4 fit; needed alongside F1-F4 on T4 |
| F6 | **Enable `cut_cross_entropy`** (if Unsloth exposes it) | ~1 GB on bf16 loss tensor | chunks loss computation across vocab dim; speed impact minor. Might allow E4B on L4 directly - check your Unsloth version |

E4B fits A100 with **none of F1-F6**. E4B on L4 needs F6 (and maybe F1+F2); on T4 needs the full F1-F5. E2B fits L4 with no fallbacks and T4 with F1-F4.

## If even A100 isn't enough (unlikely)

Quick escalation if you push E4B to long contexts or large batches:

- **Drop to `unsloth/gemma-4-E2B-it`**: E2B's PLE is ~half the size (~3.5 GB instead of ~7 GB).
- **Restrict LoRA to attention only** (`finetune_mlp_modules = False`): cuts trainable params and grads roughly in half (~18M instead of ~36M). Quality on instruction-following can suffer noticeably.
- **Move to A100 80 GB or H100**: undoes any need to compromise on batch / seq / optimizer; comfortably runs `bs=8, seq=4096`.

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
| Attention / activation | Flash Attention 2                                          | Not supported on Turing (T4 = sm_75); needs sm_80+ (Ampere). Available on L4 (Ada, sm_89) and A100 (sm_80) - Unsloth picks it up automatically there.                       |
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
- **Cross-entropy logits: `batch × seq × vocab × 2 bytes` (bf16) - sequence-dependent but vocab-dominated; ~1 GB at bs=2, seq=1024, vocab=262144 (Gemma 4's vocab)**
- CUDA/bnb overhead measured empirically from `torch.cuda.max_memory_reserved()` in cell 13 and the OOM error reports
- T4 / L4 OOM data from the empirical attempts table above

For a broader walk-through of what loads to GPU memory during a LoRA training run, see the analysis notes in [`/Users/ashishawasthi/.claude/plans/explain-what-all-gets-twinkling-turtle.md`](/Users/ashishawasthi/.claude/plans/explain-what-all-gets-twinkling-turtle.md) (this session's planning artifact).
