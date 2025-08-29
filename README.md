# Diffusion Hypernetwork for Weight Generation — Technical Specification

## 1) **Project Goals & Constraints**

* **Objective:** Train a diffusion-based hypernetwork that generates full or low-rank delta weights for multiple target families.
* **Target families (v1 scope):**

  * **Image U-Nets:** 2D Conv U-Net (channels {64,128,256,512}, group norm, SiLU), latent-space U-Net (Stable-Diffusion-style block layout).
  * **Transformers:** ViT-B/16 encoder, GPT-Small decoder (≈124M; attn heads {8,12}).
  * **Classical CNNs:** ResNet-50 (BatchNorm), MobileNetV2 (BatchNorm).
* **Outputs supported:**

  * Full tensors matching the family schema.
  * LoRA/IA³ adapters per linear/conv weight as an option to cut compute and memory.
* **Primary metrics:**

  * **Downstream performance:** Top-1 val accuracy (imagenet-1k for vision), perplexity on held-out text for GPT-Small, mIoU on a small segmentation set for U-Net variant.
  * **Reconstruction:** MSE/PSNR of weights vs ground truth; layer-wise cosine similarity.
  * **Efficiency:** Inference steps ≤ 40 with target wall-clock ≤ 1.5× a standard DDIM run on A100 80GB.
* **Constraints:** Single-node 8×A100 80GB training, mixed precision, max model RAM footprint ≤ 60GB.
* **Format requirement:** **All input and output weights use `SafeTensor` (`.safetensors`)** with strict metadata schemas.

---

## 2) **Data Ingestion & Preprocessing**

* **Loading `.safetensors`:**

  * Use `safetensors` Python API with memory-mapping. For each file: read `metadata` block containing `{family, arch_version, tensor_names, dtypes, shapes, dataset_id, task_id, train_seed}`.
  * Disallow `torch.load` and any pickle deserialization. Fail closed on missing metadata.
* **Registry & shape validation:**

  * Maintain a **model registry** keyed by `(family, arch_version)` that enumerates ordered parameter specs:
    `name, shape, dtype, role{weight|bias|norm_w|norm_b|embed}, fan_in, fan_out`.
  * On load: verify exact set equality and per-tensor shape match; compute a stable **layout hash** to ensure architectural identity.
* **Normalization:**

  * For each tensor `W`:

    * Compute per-tensor mean `μ` and std `σ` over elements. Store `(μ, σ)` in sidecar stats.
    * Normalize to `\tilde{W} = (W - μ) / (σ + ε)`; for BatchNorm/LayerNorm scale/bias keep raw values but clip to `[-10,10]`.
    * For conv/linear **weights only**, additionally apply fan-in scaling to unit variance at activation: `\tilde{W} ← \tilde{W} / √fan_in`.
  * **Packing:** Flatten each tensor to 1-D, concatenate into a single vector `x ∈ ℝ^D` with an **offset index map** to enable scatter/gather back to shapes. Maintain per-block boundaries to preserve locality.
* **Outlier handling:** Robust clip weights at 5× MAD per block. Record mask of clipped elements for analysis.

---

## 3) **Conditioning & Embedding**

* **Architecture embedding:**

  * Serialize registry graph (modules as nodes; edges = parent/child). Encode with a small **graph encoder**: 3× GraphSAGE layers → mean-pool → 256-D vector `e_arch`.
  * Concatenate learned **role embeddings** for each block (conv, qkv, o, mlp-in, mlp-out, norm, embed).
* **Dataset & task embeddings:**

  * **Dataset ID:** Learned table `E_ds ∈ ℝ^{N_ds×128}` → 128-D `e_ds`.
  * **Task spec:**

    * Structured tokens: `{task_type, label_space, resolution, modality}` mapped to embeddings and summed → 128-D `e_task`.
    * Optional text descriptor routed through a frozen MiniLM or SBERT encoder to 256-D, then projected to 128-D if present.
* **Global conditioning token:** `c = LayerNorm([e_arch | e_ds | e_task])` → 512-D, then MLP to 256-D.
* **Time-step embedding:**

  * Use **log-SNR** parameterization `λ(t)` (EDM style). Embed with sinusoidal features + 2-layer MLP with SiLU to 256-D.
  * Conditioning modulation via **AdaLN**/FiLM in the backbone.

---

## 4) **Diffusion Pipeline**

* **Data scaling:** Operate on normalized packed vector `x0` with unit variance by construction.
* **Forward (noise) process:**

  * Continuous-time EDM with log-SNR schedule `λ(t) ∈ [λ_min, λ_max]` linearly interpolated in `t ∈ [0,1]`.
  * Convert to `(α_t, σ_t)` via `α_t = √(sigmoid(λ(t)))`, `σ_t = √(sigmoid(-λ(t)))`.
  * **Noise operator $\xi_t$:**

    $$
      x_t = \xi_t(x_0, \epsilon, t) = \alpha_t\,x_0 + \sigma_t\,\epsilon,\quad \epsilon \sim \mathcal{N}(0,I).
    $$
* **Prediction target:** **v-prediction** $v = \alpha_t \epsilon - \sigma_t x_0$ for improved stability on heavy-tailed weights.
* **Reverse sampler (inference):**

  * Default **DPM-Solver++(2M)**, 20–40 steps. Fallback **DDIM** for determinism, 50 steps.
  * Classifier-free guidance on conditioning token (drop prob 0.1) to improve task adherence.
  * Optional **x0-clamp per block** using recorded `(μ,σ)` to inverse-normalize safely.

---

## 5) **Weight-Generator Network Architecture**

* **Backbone:** **Perceiver-Transformer** over block tokens.

  * **Inputs:**

    * **Block tokens:** For each parameter block `b`, project its chunked noisy slice `x_t^{(b)}` (split into fixed 4k-length chunks) with linear → 256-D tokens.
    * **Condition tokens:** `c` and role/arch tokens.
    * **Time token:** `e_t`.
  * **Latents:** 512 latent vectors (dim 512). 12 cross-attn + self-attn layers, RoPE, SwiGLU MLPs, Pre-LN, dropout 0.1.
  * **Modulation:** AdaLN with `[e_t | c]` on all blocks.
  * **Locality bias:** Per-layer attention masks to prefer attention within same module; periodic global tokens every 64 chunks.
* **Heads:**

  * For each chunk token → linear head predicts **v** with the same shape as input chunk.
  * Merge chunk outputs back to block and then to the packed vector `\hat{v}`. Recover `\hat{x}_0` and `\hat{\epsilon}` as needed.
* **LoRA mode (optional):**

  * Predict rank-r factors `(A,B)` per linear/conv weight with small r∈{4,8}. Compose with base init `W = W_base + BA`.
  * Same diffusion on concatenated `[vec(A), vec(B)]`.

---

## 6) **Training Objectives**

* **Primary loss:**

  $$
    \mathcal{L} = \lambda_{\text{denoise}}\;\| \hat{v} - v \|_2^2
      + \lambda_{\text{down}}\;\mathbb{E}_{\mathcal{D}_\text{val}}\big[\mathcal{L}_{\text{task}}(f_{\theta_{\text{gen}}}(x); \text{batch})\big]
      + \lambda_{\text{reg}}\;\mathcal{R}.
  $$

  * `\mathcal{L}_{task}` = CE for classifiers, language modeling NLL for GPT-Small, Dice/BCE for segmentation.
  * `\mathcal{R}` = weight-norm penalty on generated tensors, spectral norm soft cap per block, and KL on log-scales to discourage extreme magnitudes.
* **Downstream coupling:**

  * Evaluate target model with **generated weights only**, no fine-tune, on a small fixed validation shard (e.g., 256 images or 2k tokens).
  * **Frequency:** Apply downstream term on **25%** of steps via gradient stop-gap to control cost.
  * **Tricks:** Freeze BatchNorm stats or recompute running stats once per sample; disable dropout.
* **Weights:** Start with `λ_denoise=1.0, λ_down=0.05, λ_reg=1e-4`. Tune by Pareto sweeps.

---

## 7) **Training Setup**

* **Datasets of weights:**

  * Curate \~50k checkpoints across families, tasks, seeds, and training stages. Deduplicate via layout hash + per-block cosine threshold. Split by architecture to test cross-family generalization.
* **Batching:** Pack `B=8–16` models per step; microbatch across devices. Randomize `t ~ U[0,1]`.
* **Optimizer:** AdamW, β=(0.9,0.95), weight decay 0.05, lr 2e-4 with cosine decay and 10k warmup.
* **Precision:** bfloat16 activations, fp32 master weights. Gradient checkpointing on attention + MLPs.
* **Runtime:** EMA of model params with decay 0.999.
* **Ablations planned:** v-pred vs ε-pred, EDM vs cosine schedule, LoRA rank, downstream frequency, Perceiver latents {256,512,1024}.

---

## 8) **Sampling & Output Reconstruction**

* After the reverse pass, compute `\hat{x}_0` from `\hat{v}` and `(α_t,σ_t)`.
* **Inverse normalization:** For each block: undo fan-in scaling and z-score using stored `(μ,σ)`.
* **Safety bounds:** Clip extreme values using percentile caps per role; validate finite values.
* **Reassemble tensors:** Scatter from packed vector using index map to exact shapes and dtypes.

---

## 9) **Evaluation Protocol**

* **Reconstruction quality:** MSE, PSNR, and layer-wise cosine on held-out checkpoints.
* **Downstream:**

  * Vision: Top-1 on ImageNet-1k val for ResNet-50/ViT-B.
  * Text: PPL on WikiText-103 subset for GPT-Small.
  * U-Net: mIoU on small segmentation val or FID on a 10k synthetic set if applicable.
* **Generalization:** Train on ResNet-50, test on MobileNetV2; train on ViT-B, test on ViT-S; measure delta vs Xavier/He init and vs matching real checkpoints.
* **Efficiency:** Steps vs accuracy curves for 10, 20, 40 sampler steps.

---

## 10) **Integration & SafeTensor I/O**

* **I/O guarantees:** All reads/writes via `safetensors`. No pickle.
* **Read path:**

  * `from safetensors.torch import load_file` with `device='cpu'`, `mmap=True`.
  * Validate registry, compute layout hash, build offset map, capture `(μ,σ)` stats.
* **Write path:**

  * Repack tensors to original shapes and dtypes.
  * `from safetensors.torch import save_file(state_dict, out_path, metadata=meta)` with metadata fields: `{family, arch_version, layout_hash, created_at, gen_model_commit, stats_version}`.
  * Atomic write: write to temp + fsync + rename.
* **Streaming:** For large tensors, stream pack/unpack by block to bounded CPU buffers; overlap CPU→GPU copy with compute using non-blocking transfers.
* **Corruption checks:** SHA256 over concatenated tensors stored in metadata. On read, recompute and verify.
* **Interoperability:** Provide adapters to and from PyTorch `state_dict` without `torch.save`: use direct assignment and `state_dict.copy_`.

---

## 11) **APIs & Schemas**

* **Model registry entry (YAML):**

  ```
  family: resnet50
  arch_version: v1
  tensors:
    - name: conv1.weight
      shape: [64, 3, 7, 7]
      dtype: float32
      role: weight
      fan_in: 147
      fan_out: 64
    - ...
  ```
* **Conditioning spec (JSON):**

  ```
  {
    "dataset_id": "imagenet1k",
    "task": {"type": "classification", "num_classes": 1000, "resolution": [224,224]},
    "family": "resnet50",
    "arch_version": "v1"
  }
  ```
* **Python façade:**

  * `generate_weights(cond: Dict, mode: {"full","lora"}, steps:int=30) -> Dict[str, Tensor]`
  * `train_step(batch_of_checkpoints, cond_batch) -> LossDict`

---

## 12) **Security, Reproducibility, Compliance**

* **Determinism:** Seed all RNGs; DDIM path for deterministic runs.
* **Provenance:** Store `git_commit`, dataset hashes, and sampler settings in SafeTensor metadata.
* **Privacy:** Strip any path or user info from metadata.
* **Licensing:** Ensure only redistributable checkpoints in the training corpus.

---

## 13) **Milestones**

1. **M0:** Registry + SafeTensor I/O + packing/unpacking validated on all families.
2. **M1:** Denoising-only training to convergence on ResNet-50 weights.
3. **M2:** Add downstream loss for ResNet-50 classification.
4. **M3:** Extend to ViT-B and GPT-Small; unify conditioning.
5. **M4:** LoRA mode + 20-step DPM-Solver++ parity with denoise-only baseline.
6. **M5:** Cross-family generalization and full eval suite.

---

## 14) **Default Hyperparameters (v1)**

* `λ_min=-13`, `λ_max=13`; ρ=7 for EDM noise scaling.
* Sampler steps: 30, DPM-Solver++(2M).
* Tokens: block chunk 4k, latent slots 512, depth 12, dim 512, heads 8, MLP ratio 4.
* Batch size: effective 64 chunks per GPU via gradient accumulation.

---

## 15) **Acceptance Criteria**

* SafeTensor-only I/O with schema validation and checksums.
* ≤40 reverse steps reach within **95%** of real-checkpoint downstream metrics on held-out tasks.
* Reconstruction cosine ≥ **0.98** median per block on validation checkpoints.
* End-to-end generation time ≤ **2×** loading a standard checkpoint.

---

## 16) **Notes on Extensions**

* Per-layer SDEs with layer-specific schedules.
* Mixture-of-experts heads per family.
* Distill the sampler to a small **rectified flow** network for 4–8 step inference later.
