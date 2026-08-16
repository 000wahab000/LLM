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

- [ ] `model.save_pretrained()` tested locally — model loads back without error
- [ ] Tokenizer saved alongside weights (`tokenizer.save_pretrained()`)
- [ ] `model_card.md` filled out: architecture, dataset used, training config, known limitations
- [ ] Inference example in the model card runs end-to-end without error
- [ ] `README.md` at repo root: how to load and run the model (copy-paste ready)
- [ ] `checkpoints/` is **not** pushed — weights only, no training artifacts
- [ ] Repo visibility decision made (public or private) and documented with a reason

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
