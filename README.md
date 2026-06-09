# README

# Multimodal Composed Image Retrieval on FashionIQ

CLIP ViT-B/32 fine-tuned with LoRA + a custom DeepCrossAttentionCombiner for composed image retrieval (CIR) on the FashionIQ dataset. Given a reference garment image and a natural-language modification query, the system retrieves the closest matching target from a FAISS-indexed gallery.

---

## Reproducibility Scope

Full training reproduction is **not required** to verify results. Pre-trained weights and a pre-built FAISS index are hosted on Hugging Face and are loaded automatically at runtime. The notebook will skip training and go straight to evaluation and the interactive demo.

> **What you can reproduce:** validation Recall@{1,10,20,50,100} across all three FashionIQ categories (dress, shirt, toptee), plus the Gradio demo interface — all without any local training.

If you want to retrain from scratch, set `FORCE_RETRAIN = True` in Cell 3. A GPU runtime is required; full training on the default 10 epochs takes ~1–2 hours on a T4.

---

## Execution Environment

- **Platform:** Google Colab (tested on T4/A100 GPU runtimes)
- **Python:** 3.10+

Key library versions (installed automatically by Cell 1):

| Library | Version |
|---|---|
| `transformers` | 4.41.0 |
| `peft` | 0.11.1 |
| `faiss-cpu` | latest |
| `gradio` | latest |
| `torch` | Colab default (≥2.0) |
| `ftfy`, `pillow`, `tqdm` | latest |
| `huggingface_hub` | latest |

---

## How to Run

1. Open `clip_fine_tuned_on_fashioniq_final.ipynb` in Google Colab.
2. Make sure you have a **GPU runtime** enabled (`Runtime > Change runtime type > T4 GPU`).
3. Mount Google Drive when prompted (used for checkpoint caching only).
4. Run all cells top-to-bottom (`Runtime > Run all`).

The notebook will:
- Install dependencies (Cell 1)
- Pull the FashionIQ dataset zip from `berekety/fashion-iq-data` on Hugging Face (Cell 3)
- Load pre-trained LoRA-CLIP weights + combiner from `berekety/fashion-iq-lora-clip` (Cell 6) — **training is skipped**
- Run Recall@K evaluation across all three categories (Cell 7)
- Load the pre-built demo FAISS index (Cell 8)
- Launch a Gradio interface for interactive retrieval (Cell 9)

**Expected output (Cell 7):**
```
📊 CATEGORY-SPECIFIC PERFORMANCE BREAKDOWN:
🛍️ Category: DRESS (N queries)
   🎯 Recall@1   : x.xx%
   🎯 Recall@10  : x.xx%
   ...
🌍 GLOBAL PERFORMANCE SUMMARY
```

**Expected output (Cell 9):** A Gradio link (public or local) where you can upload a garment image and enter a text modifier to retrieve top-K matches from the gallery.

---

## Model & Dataset Links

| Resource | Link |
|---|---|
| Model weights + FAISS index | https://huggingface.co/berekety/fashion-iq-lora-clip |
| FashionIQ dataset (zipped) | https://huggingface.co/datasets/berekety/fashion-iq-data |

Both repos are set to public. If access fails, contact the team.

---

## Architecture Overview

- **Backbone:** `openai/clip-vit-base-patch32` with LoRA adapters injected into `q_proj`, `v_proj`, `k_proj`, `out_proj` (r=8, alpha=16)
- **Combiner:** `DeepCrossAttentionCombiner` — cross-attention between text tokens and reference image patches, gated MLP fusion, outputs a normalized 512-d composed query vector
- **Loss:** Global InfoNCE (temperature=0.07)
- **Retrieval:** FAISS `IndexFlatIP` over a gallery of ~50+ images (FashionIQ train+val)
- **Training:** AdamW (lr=5e-5), CosineAnnealingLR, gradient clipping (max_norm=1.0), batch_size=32

---

## AI Tools Used

- **Claude (Anthropic):** Used throughout for architecture design discussions, debugging training loop issues, and generating documentation. The combiner architecture and LoRA injection strategy were iterated with Claude's assistance.
- **ChatGPT:** Used for brainstorming ideas.
- **Gemnai:** Used to fix some Google colab runtime issue.


---


