"""
IvLLM 671M — clean, importable model definition.

Architecture: DIM=1536, 24 blocks, GQA (24Q / 6KV groups),
SwiGLU FFN, RoPE, RMSNorm, tied embeddings.

This matches the released checkpoint at release/ivllm-671m-fineweb12b/.
All training scripts (sft, jepa, reward_model, grpo, agent) import
IvLLM, DIM, and SEQ_LENGTH from here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# Architecture constants
VOCAB_SIZE = 50304
DIM = 1536
NUM_BLOCKS = 24
NUM_Q_HEADS = 24
NUM_KV_GROUPS = 6
QUERIES_PER_GROUP = NUM_Q_HEADS // NUM_KV_GROUPS
HEAD_DIM = DIM // NUM_Q_HEADS   # 64
SEQ_LENGTH = 2048

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(var + self.eps)).to(x.dtype) * self.weight


class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", torch.cos(freqs), persistent=False)
        self.register_buffer("sin", torch.sin(freqs), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        seq_len = q.shape[2]
        cos = self.cos[:seq_len].view(1, 1, seq_len, -1)
        sin = self.sin[:seq_len].view(1, 1, seq_len, -1)

        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class GroupedQueryAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(DIM, NUM_Q_HEADS * HEAD_DIM, bias=False)
        self.wk = nn.Linear(DIM, NUM_KV_GROUPS * HEAD_DIM, bias=False)
        self.wv = nn.Linear(DIM, NUM_KV_GROUPS * HEAD_DIM, bias=False)
        self.wo = nn.Linear(NUM_Q_HEADS * HEAD_DIM, DIM, bias=False)
        self.wo.NANOGPT_SCALE_INIT = 1

    def forward(self, x: torch.Tensor, rope: RoPE) -> torch.Tensor:
        q = rearrange(self.wq(x), "b t (h d) -> b h t d", h=NUM_Q_HEADS)
        k = rearrange(self.wk(x), "b t (g d) -> b g t d", g=NUM_KV_GROUPS)
        v = rearrange(self.wv(x), "b t (g d) -> b g t d", g=NUM_KV_GROUPS)
        q, k = rope(q, k)
        k = repeat(k, "b g t d -> b (g r) t d", r=QUERIES_PER_GROUP)
        v = repeat(v, "b g t d -> b (g r) t d", r=QUERIES_PER_GROUP)
        context = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(rearrange(context, "b h t d -> b t (h d)"))


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = int(2 * (4 * DIM / 3))
        hidden_dim = ((hidden_dim + 127) // 128) * 128   # = 4096
        self.w_gate_val = nn.Linear(DIM, 2 * hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, DIM, bias=False)
        self.w_down.NANOGPT_SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, val = rearrange(self.w_gate_val(x), "b t (s h) -> s b t h", s=2)
        return self.w_down(F.silu(gate) * val)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_norm = RMSNorm(DIM)
        self.attn = GroupedQueryAttention()
        self.ffn_norm = RMSNorm(DIM)
        self.ffn = SwiGLU()

    def forward(self, x: torch.Tensor, rope: RoPE) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope)
        return x + self.ffn(self.ffn_norm(x))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class IvLLM(nn.Module):
    """
    IvLLM 671M.

    forward(tokens, targets=None) → (logits, loss | None)
    get_hidden(tokens)            → (B, T, DIM) normalized hidden states
    """

    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(VOCAB_SIZE, DIM)
        self.rope = RoPE(dim=HEAD_DIM, max_seq_len=SEQ_LENGTH)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(NUM_BLOCKS)])
        self.final_norm = RMSNorm(DIM)
        self.output_layer = nn.Linear(DIM, VOCAB_SIZE, bias=False)
        self.embeddings.weight = self.output_layer.weight   # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if hasattr(module, "NANOGPT_SCALE_INIT"):
            std *= (2 * NUM_BLOCKS) ** -0.5
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0.0, std)

    def get_hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return final normalized hidden states before vocab projection. Shape: (B, T, D)."""
        x = self.embeddings(tokens)
        for block in self.blocks:
            x = block(x, self.rope)
        return self.final_norm(x)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor = None):
        x = self.get_hidden(tokens)
        logits = self.output_layer(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                rearrange(logits, "b t v -> (b t) v"),
                rearrange(targets, "b t -> (b t)"),
            )
        return logits, loss
