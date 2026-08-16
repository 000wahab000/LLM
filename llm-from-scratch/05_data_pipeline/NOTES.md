# 05 — Data Pipeline

> **How this file works:**  
> Answer every question below *before* you open the notebook.  
> Your session = defending what you wrote here, not learning it cold.  
> Explain what you actually think is happening — use your own words, draw it out if needed, use an analogy, walk through an example. No length limit. The goal is that someone could read your answer and understand the concept without looking it up.

---

## Resources

**Reference — read the code, not a video:**
- [Karpathy — nanoGPT `data/` folder](https://github.com/karpathy/nanoGPT/tree/master/data) — look at `data/shakespeare_char/prepare.py` first (simple), then `data/openwebtext/prepare.py` (complex). The difference in complexity between the two is the lesson about what data preparation actually involves.

**Dataset decision — read this before the session, not after a failed run:**

You are training at ~120M parameters. At this scale the model has enough capacity to overfit narrow data and not enough to generalize on broad data. This means:

- **Use:** [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) — short stories, controlled vocabulary, completions are achievable and inspectable. A narrow, clean, single-domain corpus is the honest choice.
- **Do not use:** OpenWebText, FineWeb, or any broad web scrape. Your model will not produce coherent output on those datasets at 120M parameters. The run will cost you compute and teach you nothing except that you need more compute.

The dataset is not a detail. It is an architectural decision. Make it now.

---

## ⚠️ Required: Answer Before Your Session

**Q1. Why does data quality matter more than data quantity at small scale? Name one concrete failure mode.**

*Your answer:*

---

**Q2. Why must the tokenizer be fit on your specific training data, not imported pre-fit from somewhere else?**

*Your answer:*

---

**Q3. What happens if you feed the model training batches in strict sequential order, with no shuffling?**

*Your answer:*

---

**Q4. What is the difference between cleaning text before tokenization vs. after? Which matters more, and why?**

*Your answer:*

---

## Post-Session: Biggest Prediction Gap

> After running the notebook — what prediction were you most wrong about, and why?

*Write it here after the session:*
