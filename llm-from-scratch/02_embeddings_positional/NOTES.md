# 02 — Embeddings & Positional Encoding

> **How this file works:**  
> Answer every question below *before* you open the notebook.  
> Your session = defending what you wrote here, not learning it cold.  
> Explain what you actually think is happening — use your own words, draw it out if needed, use an analogy, walk through an example. No length limit. The goal is that someone could read your answer and understand the concept without looking it up.

---

## Resources

**Watch first:**
- [3Blue1Brown — But what is a GPT? Visual intro to transformers](https://www.youtube.com/watch?v=wjZofJX0v4M) — watch specifically for the embedding section. He covers the lookup table, the vector space idea, and why embeddings have geometric meaning. That's the intuition base for everything in this module.

**Sheets exercise (`sheets/`):**
Cosine similarity between hand-picked word vectors, computed by hand — including the `king − man + woman` analogy. See [`sheets/README.md`](sheets/README.md).

---

## ⚠️ Required: Answer Before Your Session

**Q1. Why is a one-hot vector a bad representation for tokens? What does an embedding fix?**

*Your answer:*

---

**Q2. If you removed positional encoding entirely, what would the model lose? Give a concrete example sentence where it would fail.**

*Your answer:*

---

**Q3. What would happen if two different positions shared an identical positional encoding vector?**

*Your answer:*

---

**Q4. Why use sinusoidal functions for positional encoding? What property do they have that learned positional encodings don't guarantee?**

*Your answer:*

---

**Q5. The embedding layer is just a lookup table. Why does it have trainable weights if it's "just" a lookup?**

*Your answer:*

---

## Post-Session: Biggest Prediction Gap

> After running the notebook — what prediction were you most wrong about, and why?

*Write it here after the session:*
