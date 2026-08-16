# 05 — Data Pipeline

> **How this file works:**  
> Answer every question below *before* you open the notebook.  
> Your session = defending what you wrote here, not learning it cold.  
> Explain what you actually think is happening — use your own words, draw it out if needed, use an analogy, walk through an example. No length limit. The goal is that someone could read your answer and understand the concept without looking it up.

---

## Resources

**Reference — read the code, not a video:**
- [Karpathy — nanoGPT `data/` folder](https://github.com/karpathy/nanoGPT/tree/master/data) — look at `data/shakespeare_char/prepare.py` first (simple), then `data/openwebtext/prepare.py` (complex). The difference in complexity between the two is the lesson about what data preparation actually involves.

**Dataset decision — answer the question below BEFORE reading the answer underneath it:**

> **Predict first:** Your model has ~120M parameters. Given only that number,
> which of these three dataset choices do you predict would produce the most
> coherent generated text after training, and why?
>
> - **(A)** 10GB of Wikipedia articles (broad domain, formal prose)
> - **(B)** 100MB of short children's stories (narrow domain, simple vocabulary)
> - **(C)** 1TB of Common Crawl web text (maximum variety, noisy)
>
> Write your prediction and reasoning here before reading further:
>
> *Your prediction:*

---

**The answer — read this after you've written your prediction:**

Option **(B)** is correct. Here is why each of the other two fails at 120M:

- **(A) Wikipedia** — broad domain, formal, lots of named entities. A 120M model has
  capacity to overfit the style but not enough to learn the knowledge. Outputs will
  sound formal but hallucinate facts confidently.
- **(C) Common Crawl** — maximum noise, maximum domain variety. The model cannot find
  signal through the noise at this scale. Training loss will decrease but generated
  text will be incoherent.
- **(B) Short stories** — controlled vocabulary, consistent structure, short-range
  dependencies. A 120M model can produce completions that sound like the training
  distribution, which means you can actually tell whether it's learning.

Use [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) — short
stories, controlled vocabulary, coherent completions are achievable and inspectable.

**Do not use:** OpenWebText, FineWeb, or any broad web scrape. Your model will not
produce coherent output on those at 120M. The run will cost compute and teach you
nothing except that you need more compute. Set this expectation now.

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
