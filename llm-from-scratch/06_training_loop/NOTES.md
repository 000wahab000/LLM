# 06 — Training Loop

> **How this file works:**  
> Answer every question below *before* you open the notebook.  
> Your session = defending what you wrote here, not learning it cold.  
> Explain what you actually think is happening — use your own words, draw it out if needed, use an analogy, walk through an example. No length limit. The goal is that someone could read your answer and understand the concept without looking it up.

---

## Resources

**Read + watch (both):**
- [Karpathy — nanoGPT `train.py`](https://github.com/karpathy/nanoGPT/blob/master/train.py) — read the full file before touching the notebook. Every line matters: the optimizer setup, grad clipping, LR schedule, eval loop, checkpointing. This is closer to your actual stack than any other reference.
- [Karpathy — Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) — the full nanoGPT walkthrough. It is long. Watch at 1.5× and take notes on the *decisions*, not just the code — why this optimizer, why this schedule, why this batch size.

---

## ⚠️ Required: Answer Before Your Session

**Q1. Why cross-entropy loss for language modeling? What does minimizing it literally do to the model's output distribution?**

*Your answer:*

---

**Q2. What is gradient clipping doing, and what symptom in training tells you it's necessary?**

*Your answer:*

---

**Q3. Explain the learning rate schedule: why warm up, and why decay? What breaks if you skip either?**

*Your answer:*

---

**Q4. Loss goes down. Is that enough to confirm training is working? What else do you check, and why?**

*Your answer:*

---

**Q5. What is the difference between overfitting and memorization in a language model? Are they the same thing?**

*Your answer:*

---

## Training Log

> Document every run here. Loss curve screenshots go in this file or linked from it.

| Run | Date | Change Made | Result |
|-----|------|-------------|--------|
| 1   |      |             |        |

---

## Post-Session: Biggest Prediction Gap

> After running the notebook — what prediction were you most wrong about, and why?

*Write it here after the session:*
