"""
Stage 1 — Supervised Fine-Tuning (SFT).

Instruction-tunes the base IvLLM on Alpaca-style data.
Loss is masked to response tokens only so the model learns
to follow instructions without over-fitting to prompt phrasing.

Usage:
    python sft.py --base-ckpt checkpoints/ivllm_latest.pt --epochs 3
"""
import os
import math
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange

from base_model import IvLLM, SEQ_LENGTH, device
from sft_data import InstructionDataset, sft_collate

SFT_CHECKPOINT = "checkpoints/ivllm_sft.pt"


def _cosine_lr(step: int, total_steps: int, peak_lr: float, warmup_frac: float = 0.1) -> float:
    warmup = max(1, int(total_steps * warmup_frac))
    if step < warmup:
        return peak_lr * step / warmup
    ratio = (step - warmup) / max(1, total_steps - warmup)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * ratio))


def train_sft(
    base_checkpoint: str = "checkpoints/ivllm_latest.pt",
    lr: float = 2e-5,
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 8,
    max_samples: int = 50000,
    wandb_enabled: bool = False,
    dataset=None,
    output_path: str = SFT_CHECKPOINT,
) -> IvLLM:
    model = IvLLM().to(device)

    if base_checkpoint and os.path.exists(base_checkpoint):
        ckpt = torch.load(base_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        print(f"Loaded base weights from {base_checkpoint}")
    else:
        print("No base checkpoint found — SFT from random init")

    model = torch.compile(model, dynamic=False)

    if dataset is None:
        dataset = InstructionDataset(max_seq_len=SEQ_LENGTH, max_samples=max_samples)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=sft_collate, num_workers=2, pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95),
        weight_decay=0.01, fused=torch.cuda.is_available(),
    )

    total_steps = epochs * (len(loader) // grad_accum)
    global_step = 0
    optimizer.zero_grad()

    wandb_run = None
    if wandb_enabled:
        import wandb
        wandb_run = wandb.init(project="ivllm-sft", name="sft-alpaca")

    model.train()
    print(f"SFT: {epochs} epochs, {total_steps} optimizer steps, lr={lr}")

    for epoch in range(epochs):
        for batch_idx, (tokens, loss_mask) in enumerate(loader):
            tokens = tokens.to(device)
            loss_mask = loss_mask.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = model(tokens[:, :-1])
                targets = tokens[:, 1:]
                mask = loss_mask[:, 1:].float()

                flat_logits = rearrange(logits, "b t v -> (b t) v")
                flat_targets = rearrange(targets, "b t -> (b t)")
                flat_mask = rearrange(mask, "b t -> (b t)")

                per_token_loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")
                loss = (per_token_loss * flat_mask).sum() / flat_mask.sum().clamp(min=1)

            (loss / grad_accum).backward()

            if (batch_idx + 1) % grad_accum == 0:
                new_lr = _cosine_lr(global_step, total_steps, lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = new_lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 20 == 0:
                    print(
                        f"Epoch {epoch+1}/{epochs} | Step {global_step}/{total_steps} | "
                        f"Loss {loss.item():.4f} | LR {new_lr:.2e}"
                    )
                    if wandb_run:
                        wandb_run.log({"sft/loss": loss.item(), "sft/lr": new_lr}, step=global_step)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    # Save the underlying raw module (unwrap compile wrapper)
    raw = getattr(model, "_orig_mod", model)
    torch.save({"model_state_dict": raw.state_dict(), "global_step": global_step}, output_path)
    print(f"SFT checkpoint saved → {output_path}")
    if wandb_run:
        wandb_run.finish()
    return raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", default="checkpoints/ivllm_latest.pt")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    train_sft(
        base_checkpoint=args.base_ckpt,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_samples=args.max_samples,
        wandb_enabled=args.wandb,
    )
