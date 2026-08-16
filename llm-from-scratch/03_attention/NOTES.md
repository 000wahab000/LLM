# 03 — Attention

> **How this file works:**  
> Answer every question below *before* you open the notebook.  
> Your session = defending what you wrote here, not learning it cold.  
> Explain what you actually think is happening — use your own words, draw it out if needed, use an analogy, walk through an example. No length limit. The goal is that someone could read your answer and understand the concept without looking it up.

---

## Resources

**Watch first — two videos, watch both now:**
- [3Blue1Brown — Attention in transformers, visually explained](https://www.youtube.com/watch?v=eMlx5fFNoYc) — the core QKV mechanism. This is your primary reference for this module.
- [3Blue1Brown — How might LLMs store facts (MLPs)](https://www.youtube.com/watch?v=9-Jl0dxWQs8) — the second video covers the MLP and the full transformer block. **The second half maps to module 04, not this one.** Watch it now anyway so you don't have to re-watch later — just note where the attention content ends and the MLP content begins.

**Sheets exercise (`sheets/`):**
Full QKV attention computation by hand on a 4-token sentence ("the cat sat mat"), including scaling, causal masking, softmax, and weighted sum. Grid provided. See [`sheets/README.md`](sheets/README.md).

---

## ⚠️ Required: Answer Before Your Session

**Q1. What does the attention score between two tokens actually represent geometrically?**

*Your answer:*

---

**Q2. Why do we divide by √d_k? What goes wrong without it — specifically during training?**

*Your answer:*

---

**Q3. What is the causal mask doing, mechanically? Why does autoregressive language modeling specifically require it?**

*Your answer:*

---

**Q4. Why multi-head attention instead of one big attention operation? What is each head supposed to learn?**

*Your answer:*

---

**Q5. Attention is described as "soft, differentiable retrieval from a dictionary." Explain what that means in your own words.**

*Your answer:*

---

## Post-Session: Biggest Prediction Gap

> After running the notebook — what prediction were you most wrong about, and why?

*Write it here after the session:*
