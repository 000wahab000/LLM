"""
generate.py  Load a checkpoint and generate text with temperature/top-k sampling.

WHAT THIS FILE IS:
    Instructor-provided scaffolding. Students use this in phase 7 to verify
    that their trained model produces coherent output. They read the sampling
    code but do not write it.

    Students extend this in phase 7 by adding their own sampling experiments
    (e.g., top-p / nucleus sampling) as a copy-then-modify exercise.

Usage:
    python generate.py --prompt "Once upon a time"
    python generate.py --prompt "Once upon a time" --max_new_tokens 200
    python generate.py --prompt "Once upon a time" --temperature 0.8 --top_k 50
    python generate.py --checkpoint checkpoints/ckpt.pt

Requirements:
    tiktoken must be installed: pip install tiktoken
    (or swap in your own tokenizer by editing the load_tokenizer() function)
"""

import os
import argparse
import torch
import torch.nn.functional as F

from model_config import GPTConfig
from gpt import GPT


# =============================================================================
# TOKENIZER
# Using GPT-2's BPE tokenizer via tiktoken.
# If you trained with a custom tokenizer, replace encode/decode below.
# =============================================================================

def load_tokenizer():
    """Returns (encode_fn, decode_fn) for the GPT-2 tokenizer."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return enc.encode, enc.decode
    except ImportError:
        raise ImportError(
            "tiktoken not installed. Run: pip install tiktoken\n"
            "Or replace load_tokenizer() with your own tokenizer."
        )


# =============================================================================
# SAMPLING
# =============================================================================

def sample_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Filter logits to the top-k values. All other positions are set to -inf
    before the softmax, so they receive zero probability.

    Args:
        logits : (vocab_size,) float32  -- raw logits for the next token
        k      : int  -- number of top candidates to keep

    Returns:
        filtered_logits : (vocab_size,) float32
    """
    if k == 0:
        return logits  # no filtering
    values, _ = torch.topk(logits, min(k, logits.size(-1)))
    threshold  = values[-1]  # k-th largest value
    return logits.masked_fill(logits < threshold, float("-inf"))


@torch.no_grad()
def generate(
    model:          GPT,
    idx:            torch.Tensor,    # (1, T)  int64  -- prompt token IDs
    max_new_tokens: int   = 100,
    temperature:    float = 1.0,
    top_k:          int   = 0,       # 0 = disabled (use full distribution)
) -> torch.Tensor:
    """
    Autoregressively sample max_new_tokens tokens, appending each to idx.

    How sampling works at each step:
        1. Forward pass: model(idx) -> logits (1, T, vocab_size)
        2. Take logits for the LAST position: logits[:, -1, :] -> (1, vocab_size)
        3. Divide by temperature:
               temperature > 1  ->  flatter distribution  ->  more random
               temperature < 1  ->  sharper distribution  ->  more greedy
               temperature = 1  ->  no change (default)
        4. Apply top-k filtering (optional): zero out all but the top-k logits
        5. Softmax -> probability distribution over vocab
        6. Sample one token: torch.multinomial(probs, num_samples=1)
        7. Append the sampled token to idx; repeat

    Args:
        model          : GPT in eval mode
        idx            : (1, T) int64 -- prompt token IDs
        max_new_tokens : int   -- number of tokens to generate
        temperature    : float -- sampling temperature (> 0)
        top_k          : int   -- top-k filtering; 0 = no filtering

    Returns:
        idx : (1, T + max_new_tokens) int64 -- prompt + generated tokens
    """
    assert temperature > 0, "Temperature must be > 0 (use a small positive value, not 0)"
    model.eval()
    config = model.config

    for _ in range(max_new_tokens):
        # Truncate context to block_size (sliding window)
        idx_cond = idx[:, -config.block_size:]   # (1, min(T, block_size))

        # Forward pass -- we only need the logits, not the loss
        logits, _ = model(idx_cond)              # (1, T, vocab_size)

        # Focus on the last position (next-token prediction)
        next_logits = logits[:, -1, :] / temperature   # (1, vocab_size)

        # Optional top-k filtering
        if top_k > 0:
            next_logits = sample_top_k(next_logits.squeeze(0), top_k).unsqueeze(0)

        # Convert to probabilities
        probs = F.softmax(next_logits, dim=-1)   # (1, vocab_size)

        # Sample one token
        idx_next = torch.multinomial(probs, num_samples=1)  # (1, 1)

        # Append to the running sequence
        idx = torch.cat([idx, idx_next], dim=1)  # (1, T+1)

    return idx


# =============================================================================
# LOAD CHECKPOINT
# =============================================================================

def load_model(checkpoint_path: str, device: str) -> GPT:
    """
    Load a GPT model from a checkpoint saved by train.py.

    Checkpoint format (saved by train.py):
        {
            "model":     state_dict,
            "optimiser": state_dict,
            "config":    GPTConfig instance,
            "step":      int,
            "best_val":  float,
        }
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt   = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]

    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"  Loaded step    : {ckpt['step']}")
    print(f"  Best val loss  : {ckpt['best_val']:.4f}")
    print(f"  Parameters     : {model.num_parameters():,}")
    return model


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained GPT checkpoint.")
    parser.add_argument("--checkpoint",      default="checkpoints/ckpt.pt",
                        help="Path to the checkpoint file.")
    parser.add_argument("--prompt",          default="Once upon a time",
                        help="Prompt string to continue.")
    parser.add_argument("--max_new_tokens",  type=int,   default=200,
                        help="Number of new tokens to generate.")
    parser.add_argument("--temperature",     type=float, default=0.8,
                        help="Sampling temperature (> 0). Lower = more deterministic.")
    parser.add_argument("--top_k",           type=int,   default=50,
                        help="Top-k filtering. 0 = disabled (full distribution).")
    parser.add_argument("--num_samples",     type=int,   default=1,
                        help="Number of independent samples to generate.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_model(args.checkpoint, device)
    encode, decode = load_tokenizer()

    print(f"\n{'='*60}")
    print(f"Prompt      : {args.prompt!r}")
    print(f"Temperature : {args.temperature}   top_k : {args.top_k}")
    print(f"Max tokens  : {args.max_new_tokens}")
    print(f"{'='*60}\n")

    prompt_ids = encode(args.prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # (1, T_prompt)

    for i in range(args.num_samples):
        out_ids = generate(
            model,
            idx,
            max_new_tokens = args.max_new_tokens,
            temperature    = args.temperature,
            top_k          = args.top_k,
        )
        generated_ids = out_ids[0, len(prompt_ids):].tolist()
        text = decode(generated_ids)

        if args.num_samples > 1:
            print(f"--- Sample {i+1} ---")
        print(args.prompt + text)
        print()


if __name__ == "__main__":
    main()
