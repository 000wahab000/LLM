"""
gpt.py  Reference assembler for the student-built GPT.

WHAT THIS FILE IS:
    Instructor-provided scaffolding. Students do NOT write or modify this file.
    It imports classes that students build across phases 2-4 and wires them
    into the full GPT-2-style forward pass.

WHAT STUDENTS MUST PROVIDE:
    embeddings.py  (phase 2)  ->  TokenEmbedding, PositionalEmbedding
    attention.py   (phase 3)  ->  MultiHeadAttention
    block.py       (phase 4)  ->  TransformerBlock

    The exact required class signatures and tensor shape contracts are defined
    below as comments. If any student class uses a different name, argument
    order, or forward() shape, gpt.py will raise an error immediately on
    import or on the first forward pass. That error is the verification signal.

Run to verify that student code connects correctly:
    python gpt.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from model_config import GPTConfig


# =============================================================================
# INTERFACE CONTRACTS
# These are the EXACT class signatures and forward() shapes that student code
# must satisfy. Post these to the NOTES.md of each phase so students know what
# their code must produce BEFORE they see this file.
# =============================================================================

# -----------------------------------------------------------------------------
# [PHASE 2] embeddings.py
# -----------------------------------------------------------------------------
#
# class TokenEmbedding(nn.Module):
#     """Maps integer token IDs to dense embedding vectors."""
#
#     def __init__(self, vocab_size: int, d_model: int) -> None:
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         #                ^^^^^^^^^^^
#         # REQUIRED: the internal nn.Embedding MUST be named self.embedding.
#         # gpt.py uses this attribute for weight tying (lm_head shares its weights).
#
#     def forward(self, idx: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             idx : (B, T)  int64   -- token IDs, values in [0, vocab_size)
#         Returns:
#             out : (B, T, d_model) float32  -- embedding vectors
#         """
#
#
# class PositionalEmbedding(nn.Module):
#     """Produces a positional encoding for a sequence of length T."""
#
#     def __init__(self, block_size: int, d_model: int) -> None:
#         super().__init__()
#         # Learned or sinusoidal -- your choice. Must cover positions 0..block_size-1.
#
#     def forward(self, T: int) -> torch.Tensor:
#         """
#         Args:
#             T   : int  -- current sequence length (T <= block_size)
#         Returns:
#             out : (1, T, d_model) float32  -- broadcastable over batch dimension
#
#         NOTE: The argument is an INTEGER (sequence length), NOT the token-id
#         tensor. gpt.py calls:  pos_emb = self.position_embedding(T)
#         where T = idx.size(1).
#         """

# -----------------------------------------------------------------------------
# [PHASE 3] attention.py
# -----------------------------------------------------------------------------
#
# class MultiHeadAttention(nn.Module):
#     """Causal multi-head self-attention."""
#
#     def __init__(
#         self,
#         d_model:    int,
#         n_heads:    int,
#         block_size: int,
#         dropout:    float,
#     ) -> None:
#         super().__init__()
#         # The causal mask is YOUR responsibility. Build it here (register_buffer
#         # is the idiomatic way) so it lives on the right device automatically.
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x   : (B, T, d_model) float32  -- input sequence
#         Returns:
#             out : (B, T, d_model) float32  -- attended output (same shape)
#
#         Internally:
#             Q = x @ W_Q  ->  (B, T, d_model)  split to  (B, n_heads, T, head_dim)
#             K = x @ W_K  ->  (B, T, d_model)  split to  (B, n_heads, T, head_dim)
#             V = x @ W_V  ->  (B, T, d_model)  split to  (B, n_heads, T, head_dim)
#
#             scores = (Q @ K.T) / sqrt(head_dim)    -> (B, n_heads, T, T)
#             scores = masked_fill(scores, mask, -inf)
#             weights = softmax(scores, dim=-1)       -> (B, n_heads, T, T)
#             attended = weights @ V                  -> (B, n_heads, T, head_dim)
#             out = concat(attended) @ W_O            -> (B, T, d_model)
#
#         gpt.py does NOT pass a mask argument. You apply it internally.
#         """

# -----------------------------------------------------------------------------
# [PHASE 4] block.py
# -----------------------------------------------------------------------------
#
# class TransformerBlock(nn.Module):
#     """One full transformer block: attention sublayer + FFN sublayer."""
#
#     def __init__(self, config: GPTConfig) -> None:
#         super().__init__()
#         # config gives you everything: d_model, n_heads, block_size, dropout.
#         # Build:
#         #   self.ln1  = nn.LayerNorm(config.d_model)
#         #   self.attn = MultiHeadAttention(config.d_model, config.n_heads,
#         #                                  config.block_size, config.dropout)
#         #   self.ln2  = nn.LayerNorm(config.d_model)
#         #   self.ffn  = <your MLP: d_model -> 4*d_model -> d_model>
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Pre-norm architecture (LayerNorm before each sublayer):
#
#             x = x + attn(ln1(x))    residual around attention
#             x = x + ffn(ln2(x))     residual around FFN
#
#         Args:
#             x   : (B, T, d_model) float32
#         Returns:
#             out : (B, T, d_model) float32  -- same shape, always
#         """

# =============================================================================
# IMPORTS  (will raise ImportError if student hasn't built the file yet)
# =============================================================================

from embeddings import TokenEmbedding, PositionalEmbedding   # phase 2
from block import TransformerBlock                            # phase 4
# MultiHeadAttention is imported inside block.py; gpt.py doesn't need it directly.


# =============================================================================
# GPT
# =============================================================================

class GPT(nn.Module):
    """
    Full GPT-2-style decoder-only transformer.

    Forward pass data flow (all shapes shown for batch B, sequence T):

        idx  (B, T)  int64
          |
          +-- TokenEmbedding(idx)        -->  tok_emb  (B, T, d_model)
          +-- PositionalEmbedding(T)     -->  pos_emb  (1, T, d_model)
          |
          +-- tok_emb + pos_emb          -->  x        (B, T, d_model)
          |
          +-- Dropout(x)                 -->  x        (B, T, d_model)
          |
          +-- TransformerBlock_0(x)      -->  x        (B, T, d_model)
          +-- TransformerBlock_1(x)      -->  x        (B, T, d_model)
          ...  (n_layers blocks total)
          |
          +-- LayerNorm(x)               -->  x        (B, T, d_model)
          |
          +-- Linear(x)                  -->  logits   (B, T, vocab_size)

        If targets provided:
          loss = cross_entropy(
              logits.view(B*T, vocab_size),
              targets.view(B*T),
          )  --> scalar
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # ── Embeddings (student-built, phase 2) ─────────────────────────────
        self.token_embedding    = TokenEmbedding(config.vocab_size, config.d_model)
        self.position_embedding = PositionalEmbedding(config.block_size, config.d_model)
        self.emb_dropout        = nn.Dropout(config.dropout)

        # ── Transformer blocks (student-built, phase 4) ──────────────────────
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # ── Output head (instructor-provided) ────────────────────────────────
        self.ln_f    = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share the token embedding matrix with the output projection.
        # This is standard in GPT-2 and saves vocab_size * d_model = ~38M parameters.
        # REQUIRES: TokenEmbedding must expose an nn.Embedding named exactly self.embedding.
        #
        # Check fires BEFORE any tensor operations so the error message is the first
        # thing the student sees, not a confusing downstream AttributeError.
        if not hasattr(self.token_embedding, "embedding"):
            found = [
                attr for attr in vars(self.token_embedding)
                if not attr.startswith("_")
            ]
            raise AttributeError(
                "Weight tying failed: TokenEmbedding must expose 'self.embedding' "
                "(an nn.Embedding instance).\n"
                f"Found attributes on your TokenEmbedding: {found}\n"
                "Fix in embeddings.py — the nn.Embedding layer must be named exactly:\n"
                "    self.embedding = nn.Embedding(vocab_size, d_model)\n"
                "Rename whichever attribute above holds your nn.Embedding to 'embedding'."
            )
        if not isinstance(self.token_embedding.embedding, nn.Embedding):
            raise TypeError(
                "Weight tying failed: TokenEmbedding.embedding must be an nn.Embedding, "
                f"but found {type(self.token_embedding.embedding).__name__}.\n"
                "Fix in embeddings.py:\n"
                "    self.embedding = nn.Embedding(vocab_size, d_model)"
            )
        self.lm_head.weight = self.token_embedding.embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """
        GPT-2-style weight initialisation.
        Applied only to instructor-owned parameters (ln_f, lm_head).
        Student-built modules initialise their own weights.
        """
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        nn.init.ones_(self.ln_f.weight)
        nn.init.zeros_(self.ln_f.bias)

    def forward(
        self,
        idx:     torch.Tensor,                # (B, T) int64
        targets: Optional[torch.Tensor] = None,  # (B, T) int64 or None
    ):
        """
        Args:
            idx     : (B, T) int64   -- input token IDs; T must be <= block_size
            targets : (B, T) int64   -- ground-truth next tokens; None at inference

        Returns:
            logits  : (B, T, vocab_size) float32
            loss    : scalar float32 if targets provided, else None
        """
        B, T = idx.shape

        # Shape guard -- catches the case where a student passes a sequence
        # longer than the positional embedding supports.
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}. "
            f"Truncate your input before calling forward()."
        )

        # ── Embedding ────────────────────────────────────────────────────────
        tok_emb = self.token_embedding(idx)    # (B, T, d_model)
        pos_emb = self.position_embedding(T)   # (1, T, d_model)  -- broadcast over B
        x = self.emb_dropout(tok_emb + pos_emb)  # (B, T, d_model)

        # ── Transformer blocks ───────────────────────────────────────────────
        for block in self.blocks:
            x = block(x)  # (B, T, d_model) -> (B, T, d_model)  [shape must be preserved]

        # ── Output head ──────────────────────────────────────────────────────
        x      = self.ln_f(x)      # (B, T, d_model)
        logits = self.lm_head(x)   # (B, T, vocab_size)

        # ── Loss (training only) ─────────────────────────────────────────────
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),  # (B*T, vocab_size)
                targets.view(-1),                          # (B*T,)
            )

        return logits, loss

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return parameter count. Use for sanity-checking ~120M."""
        params = (
            (p for p in self.parameters() if p.requires_grad)
            if trainable_only else self.parameters()
        )
        return sum(p.numel() for p in params)


# =============================================================================
# VERIFICATION
# Run:  python gpt.py
# Two tests:
#   1. Shape test  -- output tensors have the right dimensions
#   2. Causality test -- earlier positions are unaffected by later tokens
#                        (proves the causal mask is actually working)
# Both must pass. Shapes passing alone does NOT verify correctness.
# =============================================================================

if __name__ == "__main__":
    cfg   = GPTConfig()
    model = GPT(cfg)
    model.eval()

    # ── Test 1: Shape verification ───────────────────────────────────────────
    B, T = 2, 64
    idx     = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    print("[1/2] Shape verification...")
    with torch.no_grad():
        logits, loss = model(idx, targets)

    assert logits.shape == (B, T, cfg.vocab_size), (
        f"SHAPE TEST FAILED: expected logits ({B}, {T}, {cfg.vocab_size}), "
        f"got {tuple(logits.shape)}"
    )
    assert loss is not None and loss.ndim == 0, (
        "SHAPE TEST FAILED: loss should be a scalar tensor."
    )
    print(f"     logits : {tuple(logits.shape)}")
    print(f"     loss   : {loss.item():.4f}")
    print(f"     params : {model.num_parameters():,}")
    print("     PASSED")

    # ── Test 2: Causality verification ───────────────────────────────────────
    # A correct causal mask means token at position t can only attend to
    # positions 0..t. Changing a token at position t must NOT affect the
    # model's output at any position 0..t-1.
    #
    # Test: build two sequences identical everywhere EXCEPT the very last
    # token. If the causal mask is correct, the logits at all earlier
    # positions must be identical for both sequences.
    #
    # A model with a broken or missing causal mask WILL fail this test
    # even though it passes the shape test above.
    print("\n[2/2] Causality verification...")

    T_causal = 16   # short enough to be fast, long enough to be meaningful
    seq_a = torch.randint(0, cfg.vocab_size, (1, T_causal))
    seq_b = seq_a.clone()
    # Change only the last token so it is guaranteed to be different
    seq_b[0, -1] = (seq_a[0, -1] + 1) % cfg.vocab_size

    with torch.no_grad():
        logits_a, _ = model(seq_a)   # (1, T_causal, vocab_size)
        logits_b, _ = model(seq_b)   # (1, T_causal, vocab_size)

    # All positions except the last must be bit-for-bit identical.
    # We use allclose with a tight tolerance to allow for floating-point
    # arithmetic differences (there should be none, but hardware can vary).
    early_a = logits_a[0, :-1, :]   # (T_causal-1, vocab_size)
    early_b = logits_b[0, :-1, :]   # (T_causal-1, vocab_size)

    if not torch.allclose(early_a, early_b, atol=1e-5):
        max_diff = (early_a - early_b).abs().max().item()
        raise AssertionError(
            "CAUSALITY TEST FAILED\n"
            f"Max difference at earlier positions: {max_diff:.6f}\n"
            "Changing only the last input token should not affect logits at "
            "any earlier position. It did, which means your causal mask is "
            "broken or missing.\n"
            "Where to look: MultiHeadAttention.forward() in attention.py.\n"
            "The upper-triangular mask must be applied BEFORE softmax so that "
            "each token can only attend to itself and tokens to its left."
        )

    print("     seq_a and seq_b differ only at position -1")
    print(f"    logits at positions 0..{T_causal-2} are identical for both: YES")
    print("     PASSED")

    print("\nAll verification checks passed. Student code is correctly connected.")
