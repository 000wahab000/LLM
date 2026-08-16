# SCAFFOLD.md — Boundary document for instructors

This document defines exactly which files are instructor scaffolding (verification
infrastructure) and which files students must write entirely from scratch. This
boundary is the no-copy-paste line.

---

## Files the instructor provides (do NOT give students these files)

| File | Purpose |
|---|---|
| `model_config.py` | Single shared config — everyone imports `GPTConfig` from here |
| `gpt.py` | Assembles student-built modules into the full model; defines exact interface contracts |
| `train.py` | Training loop skeleton — imports student's `gpt.py` + `data.py` |
| `generate.py` | Sampling at inference — loads a checkpoint, generates text |
| `SCAFFOLD.md` | This file |

**What instructors use these for:**

- `python model_config.py` — verifies config is sane, prints param count estimate
- `python gpt.py` — verifies that student's phase 2/3/4 code connects correctly (shape check)
- `python train.py` — runs training; student's `data.py` is tested here
- `python generate.py --checkpoint checkpoints/ckpt.pt` — generates text to verify training worked

**When to give these to students:**

Give `model_config.py` at the start of phase 2 so they can import it immediately.
Give `gpt.py` at the start of phase 4 (when they begin `block.py`) so they understand
what their block must produce. Give `train.py` and `generate.py` at the start of
phases 5 and 7 respectively.

---

## Files students must write from scratch (one per phase)

Students write every line themselves. No copy-paste from any reference implementation.
They may use their NOTES.md answers, their sheets exercises, and the interface contracts
in `gpt.py` as their only guide.

| File | Phase | What students build |
|---|---|---|
| `embeddings.py` | 2 | `TokenEmbedding`, `PositionalEmbedding` |
| `attention.py` | 3 | `MultiHeadAttention` (causal, multi-head) |
| `block.py` | 4 | `TransformerBlock` (pre-norm, residuals, FFN) |
| `data.py` | 5 | `get_batch(split, config, device)` |
| *(notebooks)* | 1–7 | Phase notebook code — tokenizer, training experiments, sampling experiments |

---

## Exact interface contracts (copy these into each phase's NOTES.md)

### embeddings.py

```python
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None: ...
    #
    # REQUIRED: the internal nn.Embedding MUST be stored as self.embedding
    # (gpt.py ties the output projection's weights to this attribute).
    # self.embedding = nn.Embedding(vocab_size, d_model)
    #
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx : (B, T)           int64
        # out : (B, T, d_model)  float32


class PositionalEmbedding(nn.Module):
    def __init__(self, block_size: int, d_model: int) -> None: ...
    def forward(self, T: int) -> torch.Tensor:
        # T   : int              -- sequence length (not the token tensor)
        # out : (1, T, d_model)  float32  -- broadcastable over batch
```

### attention.py

```python
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model:    int,
        n_heads:    int,
        block_size: int,
        dropout:    float,
    ) -> None: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x   : (B, T, d_model)  float32  -- input
        # out : (B, T, d_model)  float32  -- attended output
        #
        # Causal mask: build and apply INSIDE forward().
        # gpt.py does not pass a mask argument.
        #
        # Internal shapes (for your reference):
        #   Q, K, V per head : (B, n_heads, T, head_dim)
        #   attention scores : (B, n_heads, T, T)   before mask + softmax
        #   attention weights: (B, n_heads, T, T)   after softmax
        #   attended values  : (B, n_heads, T, head_dim)
        #   output           : (B, T, d_model)      after concat + W_O
```

### block.py

```python
class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None: ...
    #
    # config gives you: d_model, n_heads, block_size, dropout
    # Expected internal structure:
    #   self.ln1  = nn.LayerNorm(config.d_model)
    #   self.attn = MultiHeadAttention(config.d_model, config.n_heads,
    #                                   config.block_size, config.dropout)
    #   self.ln2  = nn.LayerNorm(config.d_model)
    #   self.ffn  = <MLP: d_model -> 4*d_model -> d_model>

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x   : (B, T, d_model)  float32
        # out : (B, T, d_model)  float32   -- same shape, always
        #
        # Pre-norm residual structure:
        #   x = x + attn(ln1(x))
        #   x = x + ffn(ln2(x))
```

### data.py

```python
def get_batch(
    split:  str,        # "train" or "val"
    config: GPTConfig,
    device: str,        # "cuda" or "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    # Returns:
    #   x : (config.batch_size, config.block_size)  int64  -- input token IDs
    #   y : (config.batch_size, config.block_size)  int64  -- target token IDs
    #
    # y[b, t] == x[b, t+1] for all t  (y is x shifted left by one position)
    # Both tensors must already be on `device`.
```

---

## Verification sequence (run in this order)

```bash
# 1. Check config
python model_config.py

# 2. Check that phase 2+4 code connects (runs gpt.py's __main__ block)
python gpt.py
# Expected output:
#   logits shape : (2, 64, 50257)
#   loss         : ~10.8  (near ln(vocab_size) at random init)
#   All shape checks passed.

# 3. Run one training step (verifies data.py + train.py + model)
python train.py

# 4. Generate from checkpoint
python generate.py --checkpoint checkpoints/ckpt.pt --prompt "Once upon a time"
```

If step 2 fails: student's embeddings.py or block.py has a shape mismatch.
If step 3 fails: student's data.py returns wrong shapes or dtypes.
If step 4 fails: checkpoint format mismatch or generate.py tokenizer issue.

---

## What "no-copy-paste" means in practice

Students may:
- Read the interface contracts in `gpt.py` and implement to match them
- Use the sheets exercises, their NOTES.md answers, and 3b1b/Karpathy videos
- Look at nanoGPT for inspiration (not copying — reading to understand decisions)

Students may not:
- Copy any implementation from nanoGPT, HuggingFace, or any other codebase
- Copy from each other
- Ask an LLM to write their phase files for them

The verification test (step 2 above) is pass/fail: either the shapes agree or they don't.
A student who copies a working implementation passes trivially. A student who
understood the mechanism and implemented it themselves also passes. The NOTES.md
answers and the sabotage tasks in each notebook are how you distinguish the two.
