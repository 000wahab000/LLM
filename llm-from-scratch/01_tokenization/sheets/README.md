# Sheets Exercise — BPE by Hand

**Do this after watching Karpathy's tokenizer video and before opening the notebook.**

---

## Corpus

```
low low low low
lower lower
newest newest newest
widest widest
```

Represent each unique word as a sequence of characters with `</w>` appended to the final character (BPE convention for word boundaries). Count frequencies.

**Initial word representations:**

| Word sequence | Frequency |
|---|---|
| `l o w </w>` | 4 |
| `l o w e r </w>` | 2 |
| `n e w e s t </w>` | 3 |
| `w i d e s t </w>` | 2 |

---

## Step 1 — Initial vocabulary

List every unique character token that appears across all words.

*Your vocabulary:*

---

## Step 2 — Count all adjacent pairs (weighted by word frequency)

For each adjacent pair of tokens, count total occurrences across the corpus (multiply pair count within a word by that word's frequency).

| Pair | Count |
|---|---|
| `l o` | ? |
| `o w` | ? |
| ... | |

Fill in the full table. Which pair has the highest count?

*Winning pair:*

---

## Step 3 — Merge and repeat (5 rounds)

For each round: merge the winning pair into a single token everywhere it appears, update all word representations, recount pairs, find the new winner.

| Round | Winning pair | Merged into | New vocab size |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |

Show your updated word representations after round 5:

---

## Step 4 — Reflect

1. What tokens are now single units that weren't before? Does that match what you'd intuitively "chunk" together?
2. The word `lowest` never appeared in the corpus. How would your tokenizer (after 5 merges) encode it? Show the token sequence.
3. What would happen if you ran 500 merges on a corpus this small? What's the limiting case?

*Your answers:*
