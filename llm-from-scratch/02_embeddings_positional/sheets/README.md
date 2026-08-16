# Sheets Exercise — Cosine Similarity & Positional Encoding by Hand

**Do this after watching the 3b1b GPT video and before opening the notebook.**

---

## Part 1 — Cosine Similarity

### Setup

Four word vectors (3-dimensional — kept small so you can compute without a calculator):

| Word | Vector |
|---|---|
| `king` | [2, 1, 0] |
| `queen` | [2, 1, 1] |
| `man` | [1, 0, 0] |
| `woman` | [1, 0, 1] |

### Formula

```
cos(A, B) = (A · B) / (|A| × |B|)

where  A · B = sum of elementwise products
       |A|   = sqrt(sum of squared elements)
```

### Task — Compute all pairwise similarities

Show your work for each. Don't skip steps.

| Pair | A · B | \|A\| | \|B\| | cos(A, B) |
|---|---|---|---|---|
| king / queen | | | | |
| king / man | | | | |
| king / woman | | | | |
| queen / man | | | | |
| queen / woman | | | | |
| man / woman | | | | |

Which pair is most similar? Does that match your intuition?

---

## Part 2 — Vector Arithmetic

Compute `king − man + woman` by hand (element-by-element subtraction then addition).

**Result vector:** ___

Look at your similarity table. Which of the four words does this vector land closest to?

Is this what you expected? What does it tell you about what embeddings are actually encoding?

*Your reflection:*

---

## Part 3 — Positional Encoding

Two tokens at positions 0 and 1 have the **same embedding** `[1, 0, 1]` (same word, repeated twice).

Apply sinusoidal positional encoding and show that they become distinguishable.

**Formula** (d = 3, so dimensions are 0, 1, 2):
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

For d=3: compute PE for each position (pos=0 and pos=1) across dims 0, 1, 2.

| | dim 0 | dim 1 | dim 2 |
|---|---|---|---|
| PE(pos=0) | | | |
| PE(pos=1) | | | |

**Final vectors** (embedding + PE):

| | dim 0 | dim 1 | dim 2 |
|---|---|---|---|
| pos=0 | | | |
| pos=1 | | | |

Are they now different? Compute their cosine similarity.

*Why does this matter for the transformer? What would happen if PE(0) = PE(1)?*

*Your answer:*
