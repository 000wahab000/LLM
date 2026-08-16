# 06 — Training Loop

> **How this file works:**  
> Answer every question below *before* you open the notebook.  
> Your session = defending what you wrote here, not learning it cold.  
> Explain what you actually think is happening — use your own words, draw it out if needed, use an analogy, walk through an example. No length limit. The goal is that someone could read your answer and understand the concept without looking it up.

---

## ⏱️ Before You Hit Run — Read This First

This is the phase most likely to surprise you with a wall-clock reality check. Read these numbers before you start, not after you've been waiting 45 minutes wondering if something is broken.

### GPU requirement

| Run type | Minimum tier | Why |
|---|---|---|
| **Toy run** (verify loss is decreasing) | Colab free T4 | Fine — short, just needs a gradient to flow |
| **Real run** (produce usable output) | Colab free T4 is enough, but session time is the limit | A full run may outlast a free session — use `--resume` |
| **Full training** (thousands of steps to convergence) | Colab Pro or Pro+ recommended | Longer sessions, background execution, faster A100 |

### Approximate wall-clock times on a Colab T4 (120M params, TinyStories, batch=32)

| Steps | Wall-clock | What you learn |
|---|---|---|
| **200 steps** | ~3–5 minutes | Whether loss is decreasing at all. This is your **toy run**. Do this first. |
| **2,000 steps** | ~25–35 minutes | Whether the loss curve is behaving (smooth decrease, no explosion). |
| **10,000 steps** | ~2–3 hours | Model begins producing vaguely story-shaped text. Minimum for phase 7. |
| **50,000+ steps** | Many hours, multi-session | Coherent completions. Requires checkpoint-resume across Colab sessions. |

### Toy run vs real run — do the toy run first, always

A **toy run** is 200 steps just to confirm the system works end-to-end:
- Loss decreases? ✓ Your model is learning.
- Loss explodes or NaN? ✗ Check gradient clipping, LR, batch size.
- Runs without error but loss doesn't move? ✗ Check your data pipeline.

Only start a **real run** after the toy run passes. Do not invest hours into a run that has a silent bug you could have caught in 5 minutes.

### Colab disconnects — `train.py` already supports resume

Colab free tier will disconnect you mid-session. This is expected, not a failure.

`train.py` saves a checkpoint every `eval_interval` steps. To resume after a disconnect:

```bash
python train.py --resume
```

This picks up from the last saved checkpoint automatically. **Set `eval_interval` to something small (e.g. 100 steps) for long runs** so you lose at most that many steps to a disconnect.

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
