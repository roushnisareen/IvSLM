"""
Stage 2 — Language JEPA (Joint Embedding Predictive Architecture).

Trains two encoders on (question, answer) pairs:
  - Context encoder  : encodes the question → pooled representation
  - Target encoder   : EMA copy of context encoder, encodes the answer
  - Predictor        : small MLP that predicts target repr from context repr

During RLHF (GRPO), the JEPA reward signal measures how close a generated
response's embedding is to the predicted "correct answer" embedding for a
given question. This gives a representation-level quality signal that is
complementary to the reward model's preference signal.

Usage:
    python jepa.py --base-ckpt checkpoints/ivllm_sft.pt --epochs 5
"""
import os
import copy
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tiktoken

from ivllm import IvLLM, RMSNorm, DIM, SEQ_LENGTH, device

enc = tiktoken.get_encoding("gpt2")

JEPA_CHECKPOINT = "checkpoints/ivllm_jepa.pt"
PRED_HIDDEN = 512       # predictor MLP hidden dim
MAX_ENC_LEN = 512       # max tokens fed to encoders (separate from generation ctx)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class SequenceEncoder(nn.Module):
    """
    Mean-pooled sequence encoder built from a deep-copied IvLLM.
    Produces a fixed-size vector from a variable-length token sequence.
    """

    def __init__(self, base_model: IvLLM):
        super().__init__()
        # Deep copy so encoder weights are independent from the policy
        m = copy.deepcopy(base_model)
        self.embeddings = m.embeddings
        self.rope = m.rope
        self.blocks = m.blocks
        self.norm = m.final_norm

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T) → repr: (B, D)"""
        x = self.embeddings(tokens)
        v_first = None
        for block in self.blocks:
            x, v_first = block(x, self.rope, v_first)
        x = self.norm(x)
        # Mean pool, ignoring padding zeros
        mask = (tokens != 0).float().unsqueeze(-1)          # (B, T, 1)
        return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, D)


class JEPAPredictor(nn.Module):
    """Small MLP: context repr → predicted target repr."""

    def __init__(self, repr_dim: int = DIM, hidden_dim: int = PRED_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, repr_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LanguageJEPA(nn.Module):
    """
    Full JEPA model.

    Training:
      loss = smooth_l1(normalize(predictor(ctx_enc(question))),
                       normalize(tgt_enc(answer)))
      After each step: EMA update of target encoder from context encoder.

    Inference (GRPO reward):
      jepa.reward(question_tokens, response_tokens)
      → cosine similarity in [-1, 1]; higher = response is closer to the
        representation JEPA predicts a correct answer should have.
    """

    def __init__(self, base_model: IvLLM, ema_momentum: float = 0.996):
        super().__init__()
        self.ema_momentum = ema_momentum
        self.context_encoder = SequenceEncoder(base_model)
        # Target encoder: EMA copy, no gradients
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.predictor = JEPAPredictor()

    def forward(self, question_tokens: torch.Tensor, answer_tokens: torch.Tensor):
        """
        Returns (loss, pred_repr, tgt_repr) where repr shapes are (B, D).
        """
        ctx_repr = self.context_encoder(question_tokens)        # (B, D) — gradient flows
        with torch.no_grad():
            tgt_repr = self.target_encoder(answer_tokens)       # (B, D) — EMA, no grad
        pred_repr = self.predictor(ctx_repr)                    # (B, D)

        pred_n = F.normalize(pred_repr, dim=-1)
        tgt_n = F.normalize(tgt_repr, dim=-1)
        loss = F.smooth_l1_loss(pred_n, tgt_n)
        return loss, pred_repr, tgt_repr

    @torch.no_grad()
    def update_ema(self):
        """EMA: target ← m*target + (1−m)*context."""
        m = self.ema_momentum
        for p_ctx, p_tgt in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            p_tgt.data.mul_(m).add_(p_ctx.data, alpha=1.0 - m)

    @torch.no_grad()
    def predict_correct_repr(self, question_tokens: torch.Tensor) -> torch.Tensor:
        """Predict normalized representation of a correct answer for a given question."""
        ctx = self.context_encoder(question_tokens)
        return F.normalize(self.predictor(ctx), dim=-1)

    @torch.no_grad()
    def encode_response(self, response_tokens: torch.Tensor) -> torch.Tensor:
        """Encode a generated response via the target encoder."""
        return F.normalize(self.target_encoder(response_tokens), dim=-1)

    @torch.no_grad()
    def reward(
        self,
        question_tokens: torch.Tensor,
        response_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Cosine similarity in [-1, 1]; used as RLHF reward component. Shape: (B,)."""
        pred = self.predict_correct_repr(question_tokens)
        resp = self.encode_response(response_tokens)
        return (pred * resp).sum(dim=-1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class QAPairsDataset(Dataset):
    def __init__(self, pairs: list, max_len: int = MAX_ENC_LEN):
        self.pairs = pairs
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        q, a = self.pairs[idx]
        q_ids = enc.encode(q)[: self.max_len]
        a_ids = enc.encode(a)[: self.max_len]
        return (
            torch.tensor(q_ids, dtype=torch.long),
            torch.tensor(a_ids, dtype=torch.long),
        )


def _jepa_collate(batch):
    q_list, a_list = zip(*batch)
    max_q = max(t.size(0) for t in q_list)
    max_a = max(t.size(0) for t in a_list)
    B = len(batch)
    q_pad = torch.zeros(B, max_q, dtype=torch.long)
    a_pad = torch.zeros(B, max_a, dtype=torch.long)
    for i, (q, a) in enumerate(zip(q_list, a_list)):
        q_pad[i, : q.size(0)] = q
        a_pad[i, : a.size(0)] = a
    return q_pad, a_pad


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_jepa(
    base_model: IvLLM,
    qa_pairs: list = None,
    lr: float = 1e-4,
    epochs: int = 5,
    batch_size: int = 16,
    ema_momentum: float = 0.996,
    wandb_enabled: bool = False,
) -> "LanguageJEPA":

    jepa = LanguageJEPA(base_model, ema_momentum=ema_momentum).to(device)

    if qa_pairs is None:
        from sft_data import build_qa_pairs
        print("Loading QA pairs for JEPA training …")
        qa_pairs = build_qa_pairs()

    dataset = QAPairsDataset(qa_pairs)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=_jepa_collate, num_workers=2, pin_memory=True,
    )

    # Only optimize context encoder + predictor (target encoder is EMA)
    params = list(jepa.context_encoder.parameters()) + list(jepa.predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    wandb_run = None
    if wandb_enabled:
        import wandb
        wandb_run = wandb.init(project="ivllm-jepa", name="jepa-alpaca")

    jepa.train()
    global_step = 0

    for epoch in range(epochs):
        for q_tokens, a_tokens in loader:
            q_tokens, a_tokens = q_tokens.to(device), a_tokens.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, pred_repr, _ = jepa(q_tokens, a_tokens)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            jepa.update_ema()
            global_step += 1

            if global_step % 50 == 0:
                # Representation std tracks representation collapse (higher is healthier)
                repr_std = pred_repr.detach().float().std(dim=0).mean().item()
                print(
                    f"Epoch {epoch+1}/{epochs} | Step {global_step} | "
                    f"Loss {loss.item():.4f} | Repr Std {repr_std:.4f}"
                )
                if wandb_run:
                    wandb_run.log({"jepa/loss": loss.item(), "jepa/repr_std": repr_std}, step=global_step)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"jepa_state_dict": jepa.state_dict(), "global_step": global_step}, JEPA_CHECKPOINT)
    print(f"JEPA checkpoint saved → {JEPA_CHECKPOINT}")
    if wandb_run:
        wandb_run.finish()
    return jepa


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", default="checkpoints/ivllm_sft.pt")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ema-momentum", type=float, default=0.996)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    m = IvLLM().to(device)
    if os.path.exists(args.base_ckpt):
        ckpt = torch.load(args.base_ckpt, map_location=device, weights_only=False)
        m.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        print(f"Encoder weights from {args.base_ckpt}")

    train_jepa(
        base_model=m,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        ema_momentum=args.ema_momentum,
        wandb_enabled=args.wandb,
    )
