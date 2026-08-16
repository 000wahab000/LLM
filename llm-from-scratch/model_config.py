"""
model_config.py  Single source of truth for all model dimensions and
                 training hyperparameters.

Every phase notebook and every scaffold file imports GPTConfig from here.
Nobody redefines d_model, n_heads, etc. anywhere else.

Defaults produce a ~117M parameter GPT-2-style model.

Sanity check:
    python model_config.py
"""

from dataclasses import dataclass, field


@dataclass
class GPTConfig:

    # ── Model Architecture ───────────────────────────────────────────────────
    # Change these only if you deliberately want a different model size.
    # All other files read these values -- edit here and the change propagates.

    vocab_size: int   = 50_257   # GPT-2 BPE vocabulary size.
                                 # If you use a custom tokenizer, update this
                                 # to your tokenizer's vocab_size.

    d_model: int      = 768      # Embedding dimension (also called n_embd, C,
                                 # or hidden_size in other codebases).
                                 # Must be divisible by n_heads.

    n_heads: int      = 12       # Number of attention heads per block.
                                 # head_dim = d_model // n_heads = 64.

    n_layers: int     = 12       # Number of stacked TransformerBlocks.

    block_size: int   = 1_024    # Maximum sequence length (context window).
                                 # Positional embeddings are sized to this.

    dropout: float    = 0.1      # Dropout probability during training.
                                 # Set to 0.0 when evaluating or generating.

    # ── Training Hyperparameters ─────────────────────────────────────────────

    batch_size: int       = 32   # Sequences per gradient step.
    grad_accum_steps: int = 4    # Gradient accumulation steps.
                                 # Effective batch = batch_size x grad_accum_steps = 128.

    max_iters: int    = 10_000   # Total training steps.
    eval_interval: int =   500   # Evaluate every N steps.
    eval_iters: int   =   100    # Number of batches averaged per eval.

    # ── Optimiser (AdamW + cosine LR with linear warmup) ────────────────────

    learning_rate: float = 6e-4   # Peak learning rate.
    min_lr:        float = 6e-5   # Minimum LR (= 0.1 x learning_rate, cosine floor).
    warmup_iters:  int   = 200    # Linear warmup steps before cosine decay begins.
    weight_decay:  float = 0.1    # Applied to 2-D parameters only (weights, not biases).
    beta1:         float = 0.9
    beta2:         float = 0.95
    grad_clip:     float = 1.0    # Global gradient norm clip value.

    # ── Paths ────────────────────────────────────────────────────────────────

    data_dir:        str = "data/"
    checkpoint_dir:  str = "checkpoints/"
    checkpoint_name: str = "ckpt.pt"

    # ── Derived values (computed in __post_init__) ───────────────────────────

    head_dim: int = field(init=False)

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})."
            )
        object.__setattr__(self, "head_dim", self.d_model // self.n_heads)
        # head_dim = 768 // 12 = 64 with defaults.

    @property
    def approx_params(self) -> int:
        """
        Rough parameter count for sanity-checking model size.

        Calculation (ignores bias terms and layer-norm params):
            embeddings  : vocab_size x d_model  +  block_size x d_model
            per block   : 4 x d_model^2  (QKV + output proj)
                        + 8 x d_model^2  (FFN: d_model -> 4d_model -> d_model)
            output head : d_model x vocab_size
        """
        embed = (self.vocab_size + self.block_size) * self.d_model
        block = 12 * self.d_model ** 2
        head  = self.d_model * self.vocab_size
        return embed + self.n_layers * block + head


# ── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = GPTConfig()
    print("GPTConfig")
    print(f"  vocab_size = {cfg.vocab_size:,}")
    print(f"  d_model    = {cfg.d_model}")
    print(f"  n_heads    = {cfg.n_heads}  (head_dim = {cfg.head_dim})")
    print(f"  n_layers   = {cfg.n_layers}")
    print(f"  block_size = {cfg.block_size}")
    print(f"  dropout    = {cfg.dropout}")
    print(f"  ~params    = {cfg.approx_params:,}  ({cfg.approx_params / 1e6:.0f}M)")
    print("\nConfig is valid.")
