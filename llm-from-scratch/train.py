"""
train.py  Training loop skeleton for a ~120M parameter GPT on a single GPU
          (Google Colab T4/A100).

WHAT THIS FILE IS:
    Instructor-provided scaffolding. Students do NOT write this file.
    They read it to understand the training loop structure and verify their
    model plugs in correctly.

WHAT STUDENTS MUST PROVIDE:
    embeddings.py  (phase 2)  -- imported transitively via gpt.py
    attention.py   (phase 3)  -- imported transitively via block.py
    block.py       (phase 4)  -- imported transitively via gpt.py
    data.py        (phase 5)  -- imported directly here

INTERFACE CONTRACT FOR data.py:
    Students must implement ONE function:

        def get_batch(
            split:  str,           # "train" or "val"
            config: GPTConfig,
            device: str,           # "cuda" or "cpu"
        ) -> tuple[torch.Tensor, torch.Tensor]:
            \"\"\"
            Return a random batch of (input, target) sequences.

            Returns:
                x : (batch_size, block_size)  int64  -- input token IDs
                y : (batch_size, block_size)  int64  -- target token IDs (x shifted by 1)
                    y[b, t] = x[b, t+1]  for all valid t,
                    i.e. y is the next-token label for every position in x.
            \"\"\"

    data.py may load its data any way it likes internally (memory-mapped
    numpy, HuggingFace datasets, etc.) as long as get_batch() returns the
    correct shapes and dtypes.

Run:
    python train.py
    python train.py --resume   # resume from checkpoint
"""

import os
import math
import time
import argparse
import torch
import torch.nn as nn
from contextlib import nullcontext

from model_config import GPTConfig
from gpt import GPT
from data import get_batch   # student-built in phase 5


# =============================================================================
# LEARNING RATE SCHEDULE
# Cosine decay with linear warmup -- standard for transformer training.
# =============================================================================

def get_lr(step: int, config: GPTConfig) -> float:
    """
    Returns the learning rate for the given step.

    Schedule:
        steps 0 .. warmup_iters-1   : linear ramp from 0 to learning_rate
        steps warmup_iters .. end   : cosine decay from learning_rate to min_lr
    """
    # Linear warmup
    if step < config.warmup_iters:
        return config.learning_rate * (step + 1) / config.warmup_iters

    # After decay period: floor at min_lr
    if step >= config.max_iters:
        return config.min_lr

    # Cosine decay
    decay_ratio = (step - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1.0 -> 0.0
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate(model: GPT, config: GPTConfig, device: str) -> dict[str, float]:
    """
    Estimate train and val loss over eval_iters random batches each.
    Returns a dict: {"train": float, "val": float}
    """
    model.eval()
    losses = {}
    for split in ("train", "val"):
        split_losses = torch.zeros(config.eval_iters, device=device)
        for k in range(config.eval_iters):
            x, y = get_batch(split, config, device)
            _, loss = model(x, y)
            split_losses[k] = loss.item()
        losses[split] = split_losses.mean().item()
    model.train()
    return losses


# =============================================================================
# OPTIMISER SETUP
# Weight decay is applied only to 2-D parameters (weight matrices).
# Biases, layer-norm parameters, and embeddings are NOT decayed.
# =============================================================================

def configure_optimiser(model: GPT, config: GPTConfig) -> torch.optim.AdamW:
    """
    Build AdamW with selective weight decay.

    2-D params  (weight matrices)   -> weight_decay
    1-D params  (biases, layernorm) -> no weight decay
    """
    decay_params    = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]

    print(f"  decay params   : {sum(p.numel() for p in decay_params):,}")
    print(f"  no-decay params: {sum(p.numel() for p in no_decay_params):,}")

    return torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train(resume: bool = False) -> None:
    config = GPTConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"Device : {device}")
    print(f"dtype  : {dtype}")

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(config.checkpoint_dir, config.checkpoint_name)

    # ── Build model ──────────────────────────────────────────────────────────
    model = GPT(config).to(device)
    print(f"Parameters: {model.num_parameters():,}")

    optimiser = configure_optimiser(model, config)

    # ── Resume from checkpoint ───────────────────────────────────────────────
    start_iter  = 0
    best_val    = float("inf")

    if resume and os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
        ckpt       = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimiser.load_state_dict(ckpt["optimiser"])
        start_iter = ckpt["step"] + 1
        best_val   = ckpt.get("best_val", float("inf"))
        print(f"Resumed at step {start_iter}  (best val so far: {best_val:.4f})")

    # ── Mixed-precision context ───────────────────────────────────────────────
    # torch.autocast: runs the forward pass in lower precision (bfloat16/float16)
    # while keeping master weights in float32. Speeds up training on Colab GPUs.
    autocast_ctx = (
        torch.autocast(device_type=device, dtype=dtype)
        if device == "cuda" else nullcontext()
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    t0 = time.time()

    for step in range(start_iter, config.max_iters):

        # -- Set learning rate for this step --
        lr = get_lr(step, config)
        for param_group in optimiser.param_groups:
            param_group["lr"] = lr

        # -- Evaluation --
        if step % config.eval_interval == 0:
            losses = evaluate(model, config, device)
            print(
                f"step {step:6d} | "
                f"train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | "
                f"lr {lr:.2e}"
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                ckpt = {
                    "model":     model.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "config":    config,
                    "step":      step,
                    "best_val":  best_val,
                }
                torch.save(ckpt, checkpoint_path)
                print(f"  -> checkpoint saved  (val {best_val:.4f})")

        # -- Gradient accumulation --
        # Accumulate gradients over grad_accum_steps micro-batches before
        # updating weights. Effective batch = batch_size x grad_accum_steps.
        optimiser.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for micro_step in range(config.grad_accum_steps):
            x, y = get_batch("train", config, device)

            with autocast_ctx:
                _, loss = model(x, y)
                # Divide by grad_accum_steps so the accumulated gradient is the
                # average over the full effective batch, not the sum.
                loss = loss / config.grad_accum_steps

            loss.backward()
            loss_accum += loss.item()

        # -- Gradient clipping --
        # Clips the global gradient norm to grad_clip. Without this, a single
        # bad batch can cause a gradient explosion that destabilises training.
        # The norm BEFORE clipping is printed every 100 steps so you can watch it.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        if step % 100 == 0:
            t1  = time.time()
            dt  = t1 - t0
            t0  = t1
            print(
                f"step {step:6d} | loss {loss_accum:.4f} | "
                f"grad_norm {grad_norm:.3f} | {dt*1000:.1f}ms"
            )

        optimiser.step()

    print(f"Training complete. Best val loss: {best_val:.4f}")
    print(f"Checkpoint saved at: {checkpoint_path}")


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the GPT model.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the latest checkpoint.")
    args = parser.parse_args()
    train(resume=args.resume)
