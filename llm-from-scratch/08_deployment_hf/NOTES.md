# 08 — Deployment on Hugging Face

> **How this file works:**  
> Answer every question below *before* your deployment session.  
> Explain what you actually think is happening — use your own words, draw it out if needed, use an analogy, walk through an example. No length limit. The goal is that someone could read your answer and understand the concept without looking it up.

---

## Resources

**Read (no video needed — this is procedural):**
- [HF Hub — Upload a model](https://huggingface.co/docs/hub/models-uploading)
- [HF Hub — Model Cards guide](https://huggingface.co/docs/hub/model-cards)
- [HF `push_to_hub()` API reference](https://huggingface.co/docs/transformers/model_sharing)

**Pre-push checklist — complete every item before pushing:**

**Export & Loading**
- [ ] Weights saved in **safetensors format** (`model.save_pretrained(path, safe_serialization=True)`) — HF Hub prefers this over `.bin`; it loads faster and is safer to share
- [ ] `config.json` present alongside weights — required for `from_pretrained()` to reconstruct the model architecture on HF's side without your source code. Check that your model class writes it correctly via `save_pretrained()`, or write it manually with your architecture parameters
- [ ] Tokenizer saved alongside weights (`tokenizer.save_pretrained()`)
- [ ] Local load test: `AutoModel.from_pretrained(local_path)` runs without error before you push

**Model Card**
- [ ] `model_card.md` filled out: architecture (GPT-2-style, 120M params), dataset (TinyStories), training config (steps, LR, batch size), known limitations
- [ ] Inference example in the model card runs end-to-end without error
- [ ] `README.md` at repo root: how to load and run the model (copy-paste ready)

**Serving — choose one before you push:**
- [ ] **Inference API** (default, free) — HF runs your model on their servers automatically for any public repo. Use this if you just want a working demo endpoint with zero setup. Limitation: free tier is slow, especially for 120M unquantized.
- [ ] **Spaces** (Gradio/Streamlit app) — use this if you want a custom UI (a text box, generation controls, etc.). More setup, but gives you a shareable demo page. Use Spaces if Inference API output is too slow for your purposes.

**Inference speed — set this expectation now:**
- [ ] An **unquantized 120M model on HF's free Inference API will be slow** (~2–5 seconds per generation, sometimes more under load). This is normal. Your options:
  - Accept it — it works, it's just slow
  - Quantize (4-bit or 8-bit via `bitsandbytes`) to speed up inference — adds a dependency
  - Upgrade to HF Pro for faster inference hardware
  - Move to Spaces on a GPU instance (costs credits but fast)

**Cleanup**
- [ ] `checkpoints/` is **not** pushed — weights only, no training artifacts
- [ ] Repo visibility decision made (public or private) with a documented reason

---

## ⚠️ Required: Answer Before Your Session

**Q1. What is a model card and who is it actually for? What happens — concretely — if it's missing?**

*Your answer:*

---

**Q2. Why push weights to HF Hub instead of storing them in the git repo? What specific problem does this solve?**

*Your answer:*

---

**Q3. What is the minimum a stranger needs to run your model, with zero other context? List every piece.**

*Your answer:*

---

## Post-Session: Biggest Prediction Gap

> After deployment — what assumption were you most wrong about?

*Write it here after the session:*
