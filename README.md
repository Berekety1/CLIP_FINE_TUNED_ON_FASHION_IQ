# Composed Image Retrieval on FashionIQ with LoRA-tuned CLIP

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Berekety1/CLIP_FINE_TUNED_ON_FASHION_IQ/blob/main/clip_fine_tuned_on_fashioniq.ipynb)

**Find a garment from a photo plus a description of what to change.** Give the model a reference image and a text such as *"is shorter and has no sleeves"*, and it retrieves the matching item from a gallery of 52,464 fashion images.

The model is CLIP ViT-B/32 fine-tuned with **LoRA** (0.65% of parameters trained), plus a custom **cross-attention combiner** that fuses the image and the text into one query vector. That vector is searched with **FAISS**.

## Results

Validation set of [FashionIQ](https://github.com/XiaoxiaoGuo/fashion-iq): 4,574 queries searched against 13,138 candidate images.

| Category | Queries | Recall@1 | Recall@10 | Recall@20 | Recall@50 | Recall@100 |
|---|---:|---:|---:|---:|---:|---:|
| Dress | 1,845 | 5.96% | 24.50% | 33.12% | 45.91% | 57.07% |
| Shirt | 1,172 | 7.68% | 24.32% | 32.94% | 48.29% | 59.81% |
| Top/tee | 1,557 | 9.06% | 28.52% | 37.57% | 52.34% | 63.33% |
| **All** | **4,574** | **7.46%** | **25.82%** | **34.59%** | **48.71%** | **59.90%** |

*Recall@K: the share of queries whose correct target image is among the top K results.*

## How it works

```
reference image ──► CLIP image encoder (LoRA) ──► image patch tokens ─┐
                                                                       ├─► cross-attention combiner ──► composed query (512-d)
modification text ─► CLIP text encoder (LoRA) ──► text tokens ─────────┘                                     │
                                                                                                              ▼
gallery images ────► CLIP image encoder (LoRA) ──► image embeddings ──► FAISS index ──► top-K nearest images
```

- **Backbone:** `openai/clip-vit-base-patch32`, with LoRA adapters (r=8, α=16) on the attention `q_proj`, `k_proj`, `v_proj` and `out_proj` layers. That is 983K trainable parameters out of 152M.
- **Combiner (`DeepCrossAttentionCombiner`):** the text tokens attend to the reference image's patches. A learned gate then mixes this attended summary with an MLP over the global image and text embeddings, giving a normalised 512-d query.
- **Training:** contrastive InfoNCE loss (temperature 0.07) between composed queries and target images, AdamW (lr 5e-5), cosine LR schedule, gradient clipping at 1.0, batch size 32, 10 epochs.
- **Search:** exact inner-product search with FAISS `IndexFlatIP`.

## Run it

Click **Open in Colab** above, switch to a GPU runtime (`Runtime → Change runtime type → T4 GPU`), and choose `Runtime → Run all`.

No training is needed. The trained weights and pre-built FAISS indexes are downloaded from Hugging Face, so the notebook goes straight to:

1. installing dependencies (cell 1)
2. downloading FashionIQ (cell 3)
3. loading the LoRA-CLIP weights and the combiner (cell 6)
4. evaluating Recall@K for each category (cell 7)
5. loading the demo index (cell 8)
6. launching a **Gradio demo**: upload a garment photo, type a change, and see the top matches (cell 9)

To retrain from scratch, set `FORCE_RETRAIN = True` in cell 3. Training takes about 1–2 hours on a T4.

**Tested with:** Python 3.10+, `transformers` 4.41.0, `peft` 0.11.1, PyTorch ≥ 2.0, plus `faiss-cpu`, `gradio`, `ftfy`, `pillow`, `tqdm` and `huggingface_hub` (all installed by cell 1).

## Model and data

| Resource | Link |
|---|---|
| Trained weights and FAISS indexes | [huggingface.co/berekety/fashion-iq-lora-clip](https://huggingface.co/berekety/fashion-iq-lora-clip) |
| FashionIQ dataset (zipped) | [huggingface.co/datasets/berekety/fashion-iq-data](https://huggingface.co/datasets/berekety/fashion-iq-data) |

FashionIQ: H. Wu et al., *Fashion IQ: A New Dataset Towards Retrieving Images by Natural Language Feedback*, CVPR 2021.
