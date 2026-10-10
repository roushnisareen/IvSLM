"""
Stage 3 — Reward Model.

Wraps IvLLM with a scalar head trained on human preference pairs
(Bradley-Terry / Elo-style loss).  The reward model is later used
inside GRPO to score generated responses.

Dataset: Anthropic/hh-rlhf (chosen vs rejected conversation pairs).

Usage:
    python reward_model.py --sft-ckpt checkpoints/ivllm_sft.pt --epochs 3
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tiktoken

from ivllm import IvLLM, DIM, device

enc = tiktoken.get_encoding("gpt2")
RM_CHECKPOINT = "checkpoints/ivllm_rm.pt"
MAX_RM_LEN = 512


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RewardModel(nn.Module):
    """
    IvLLM backbone + scalar reward head.

    Uses the last-token hidden state (causally computed) as a
    fixed-size sequence summary, then projects to a scalar reward.

    Initialized from an SFT checkpoint so the backbone already
    understands instructions before reward training begins.
    """

    def __init__(self, base_model: IvLLM):
        super().__init__()
        self.backbone = base_model
        # Zero-init so rewards start near 0 and scale up during training
        self.reward_head = nn.Linear(DIM, 1, bias=False)
        nn.init.zeros_(self.reward_head.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T) → rewards: (B,)"""
        hidden = self.backbone.get_hidden(tokens)       # (B, T, D)
        last_h = hidden[:, -1, :]                       # (B, D) — last token summary
        return self.reward_head(last_h).squeeze(-1)     # (B,)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PreferenceDataset(Dataset):
    """(chosen_ids, rejected_ids) pairs for Bradley-Terry training."""

    def __init__(self, pairs: list, max_len: int = MAX_RM_LEN):
        self.examples = []
        for item in pairs:
            c = enc.encode(item["chosen"])[: max_len]
            r = enc.encode(item["rejected"])[: max_len]
            self.examples.append((c, r))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        chosen, rejected = self.examples[idx]
        return (
            torch.tensor(chosen, dtype=torch.long),
            torch.tensor(rejected, dtype=torch.long),
        )


def _rm_collate(batch):
    chosen_list, rejected_list = zip(*batch)
    max_c = max(t.size(0) for t in chosen_list)
    max_r = max(t.size(0) for t in rejected_list)
    B = len(batch)
    c_pad = torch.zeros(B, max_c, dtype=torch.long)
    r_pad = torch.zeros(B, max_r, dtype=torch.long)
    for i, (c, r) in enumerate(zip(chosen_list, rejected_list)):
        c_pad[i, : c.size(0)] = c
        r_pad[i, : r.size(0)] = r
    return c_pad, r_pad


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _bt_loss(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry preference loss: −log σ(r_chosen − r_rejected)."""
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


def train_reward_model(
    sft_checkpoint: str = "checkpoints/ivllm_sft.pt",
    preference_pairs: list = None,
    lr: float = 1e-5,
    epochs: int = 3,
    batch_size: int = 8,
    wandb_enabled: bool = False,
) -> RewardModel:

    base = IvLLM().to(device)
    if os.path.exists(sft_checkpoint):
        ckpt = torch.load(sft_checkpoint, map_location=device, weights_only=False)
        base.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        print(f"Backbone loaded from {sft_checkpoint}")
    else:
        print("No SFT checkpoint found — reward model starts from random init")

    rm = RewardModel(base).to(device)

    if preference_pairs is None:
        from sft_data import build_preference_pairs
        print("Loading HH-RLHF preference pairs …")
        preference_pairs = build_preference_pairs()

    dataset = PreferenceDataset(preference_pairs)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=_rm_collate, num_workers=2, pin_memory=True,
    )

    optimizer = torch.optim.AdamW(rm.parameters(), lr=lr, weight_decay=0.01)

    wandb_run = None
    if wandb_enabled:
        import wandb
        wandb_run = wandb.init(project="ivllm-rm", name="rm-hh-rlhf")

    rm.train()
    global_step = 0
    running_correct = 0
    running_total = 0

    print(f"Reward model training: {epochs} epochs over {len(dataset)} preference pairs")

    for epoch in range(epochs):
        for chosen_ids, rejected_ids in loader:
            chosen_ids = chosen_ids.to(device)
            rejected_ids = rejected_ids.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                r_chosen = rm(chosen_ids)
                r_rejected = rm(rejected_ids)
                loss = _bt_loss(r_chosen, r_rejected)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            running_correct += (r_chosen > r_rejected).sum().item()
            running_total += chosen_ids.size(0)

            if global_step % 20 == 0:
                acc = running_correct / max(running_total, 1)
                print(
                    f"Epoch {epoch+1}/{epochs} | Step {global_step} | "
                    f"Loss {loss.item():.4f} | Accuracy {acc:.3f} | "
                    f"Chosen {r_chosen.mean().item():.3f} | Rejected {r_rejected.mean().item():.3f}"
                )
                if wandb_run:
                    wandb_run.log({
                        "rm/loss": loss.item(),
                        "rm/accuracy": acc,
                        "rm/chosen_reward": r_chosen.mean().item(),
                        "rm/rejected_reward": r_rejected.mean().item(),
                    }, step=global_step)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"model_state_dict": rm.state_dict()}, RM_CHECKPOINT)
    print(f"Reward model saved → {RM_CHECKPOINT}")
    if wandb_run:
        wandb_run.finish()
    return rm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-ckpt", default="checkpoints/ivllm_sft.pt")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    train_reward_model(
        sft_checkpoint=args.sft_ckpt,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        wandb_enabled=args.wandb,
    )
