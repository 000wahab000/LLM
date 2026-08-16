# 07 — Sampling & Generation

> **How this file works:**  
> Answer every question below *before* you open the notebook.  
> Your session = defending what you wrote here, not learning it cold.  
> Explain what you actually think is happening — use your own words, draw it out if needed, use an analogy, walk through an example. No length limit. The goal is that someone could read your answer and understand the concept without looking it up.

---

## Resources

**Start here — interactive visual:**
- [Brendan Bycroft — LLM Visualizer](https://bbycroft.net/llm) — the best tool for building intuition on how sampling works at inference time. Use it before reading anything. Step through token generation manually and watch the probability distribution change.

**Then watch:**
- **Justin Angel's Workshop** — [link needed, ask instructor] — covers the implementation side of sampling (temperature, top-k, top-p in code). Good complement to Bycroft's visual.

---

## ⚠️ Required: Answer Before Your Session

**Q1. Temperature > 1 makes output more random. Explain *why* mechanically — what does it literally do to the logits?**

*Your answer:*

---

**Q2. What is wrong with greedy decoding (always pick the highest-probability token)? Give a concrete failure case.**

*Your answer:*

---

**Q3. What is the difference between top-k and top-p (nucleus) sampling? When would you prefer each?**

*Your answer:*

---

**Q4. If you set temperature = 0, what does generation become equivalent to? Is that always bad?**

*Your answer:*

---

## Post-Session: Biggest Prediction Gap

> After running the notebook — what prediction were you most wrong about, and why?

*Write it here after the session:*
