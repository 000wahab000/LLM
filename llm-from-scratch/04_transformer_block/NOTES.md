# 04 — Transformer Block

> **How this file works:**  
> Answer every question below *before* you open the notebook.  
> Your session = defending what you wrote here, not learning it cold.  
> Explain what you actually think is happening — use your own words, draw it out if needed, use an analogy, walk through an example. No length limit. The goal is that someone could read your answer and understand the concept without looking it up.

---

## Resources

**Watch first:**
- **Justin Angel's LLM Workshop** — [link needed, ask instructor] — his workshop is strongest at this specific stage. The MLP, residuals, and LayerNorm walkthrough is your primary reference here.
- **Also:** the second half of the 3b1b MLPs video from module 03 (`How might LLMs store facts`) covers this territory. If you watched it then, you already have the visual intuition — now you're implementing it.

---

## ⚠️ Required: Answer Before Your Session

**Q1. What problem do residual connections solve? Why do deep networks specifically need them?**

*Your answer:*

---

**Q2. Why LayerNorm instead of BatchNorm? What breaks if you use BatchNorm in a transformer?**

*Your answer:*

---

**Q3. The FFN inside a transformer block is typically 4× the model dimension. Why 4×? What is it doing that attention doesn't do?**

*Your answer:*

---

**Q4. If you stack 12 transformer blocks, what are the early layers likely learning vs. the later layers?**

*Your answer:*

---

## Post-Session: Biggest Prediction Gap

> After running the notebook — what prediction were you most wrong about, and why?

*Write it here after the session:*
