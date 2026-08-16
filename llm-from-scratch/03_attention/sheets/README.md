# Sheets Exercise — Self-Attention by Hand

**Do this after watching both 3b1b attention videos and before opening the notebook.**  
Work through every step. Don't skip to the reflection.

---

## Setup

**Sentence:** `the cat sat mat` (4 tokens)

**Embeddings** (4-dimensional — kept tiny so arithmetic is tractable):

| Token | Embedding |
|---|---|
| `the` | [1, 0, 1, 0] |
| `cat` | [0, 1, 0, 1] |
| `sat` | [1, 1, 0, 0] |
| `mat` | [0, 0, 1, 1] |

**Weight matrices:** W_Q = W_K = W_V = identity matrix (I₄).  
This means Q = K = V = X (the input embeddings). It keeps the arithmetic tractable while preserving the full structure of the algorithm.

---

## Step 1 — Write out Q, K, V

Since all weights are identity, just copy the embedding matrix three times and label them Q, K, V.

**Q = K = V:**

| Token | dim0 | dim1 | dim2 | dim3 |
|---|---|---|---|---|
| the | 1 | 0 | 1 | 0 |
| cat | 0 | 1 | 0 | 1 |
| sat | 1 | 1 | 0 | 0 |
| mat | 0 | 0 | 1 | 1 |

---

## Step 2 — Raw attention scores: S = Q · Kᵀ

For each query token (row), compute its dot product with every key token (column).

Show your dot product calculation for the "cat" row explicitly:

`cat · the = ?` → `cat · cat = ?` → `cat · sat = ?` → `cat · mat = ?`

**Full score matrix S (4×4):**

| | the | cat | sat | mat |
|---|---|---|---|---|
| **the** | | | | |
| **cat** | | | | |
| **sat** | | | | |
| **mat** | | | | |

---

## Step 3 — Scale by √d_k

d_k = 4, so √d_k = 2. Divide every cell by 2.

**Scaled score matrix:**

| | the | cat | sat | mat |
|---|---|---|---|---|
| **the** | | | | |
| **cat** | | | | |
| **sat** | | | | |
| **mat** | | | | |

---

## Step 4 — Apply causal mask

This is a language model. A token at position *i* may **not** attend to any token at position *j > i*.

Set every cell where column index > row index to **−∞**.

**After masking (mark masked cells as −∞):**

| | the (0) | cat (1) | sat (2) | mat (3) |
|---|---|---|---|---|
| **the (0)** | | −∞ | −∞ | −∞ |
| **cat (1)** | | | −∞ | −∞ |
| **sat (2)** | | | | −∞ |
| **mat (3)** | | | | |

Which cells are you masking? Does that match the rule above?

---

## Step 5 — Softmax (row by row)

For each row: exp every value (exp(−∞) = 0), then divide by the row sum.

Show your full working for the **"sat" row**:

| | the | cat | sat | mat |
|---|---|---|---|---|
| raw scaled | | | | −∞ |
| after exp | | | | 0 |
| after /sum | | | | 0 |

**Full attention weight matrix A (all rows, should sum to 1.0 per row):**

| | the | cat | sat | mat |
|---|---|---|---|---|
| **the** | | 0 | 0 | 0 |
| **cat** | | | 0 | 0 |
| **sat** | | | | 0 |
| **mat** | | | | |

---

## Step 6 — Weighted sum: output = A · V

The output for each token is a weighted average of the value vectors.

Compute the output vector for **"cat"** explicitly:

`output["cat"] = A["cat","the"] × V["the"] + A["cat","cat"] × V["cat"] + 0 + 0`

**output["cat"] =** ___

---

## Step 7 — Reflect

Answer all three before the session:

1. Look at "mat"'s attention weights. Which tokens does it attend to? Does that make sense given the causal mask?

2. If you removed the causal mask entirely, how would "the"'s attention weights change? What would it now be able to "see"?

3. In this exercise d_k = 4. In a real model d_k = 64 (or 128). Without the √d_k scaling, what happens to the dot products as d_k grows? Why does that cause a softmax problem specifically?

*Your answers:*
