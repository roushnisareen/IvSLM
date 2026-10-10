"""
Full training pipeline for the IvLLM Self-Correcting Agent.

Run all stages in order:
    python train_pipeline.py --base-ckpt checkpoints/ivllm_latest.pt --all

Or run individual stages:
    python train_pipeline.py --stage sft
    python train_pipeline.py --stage jepa
    python train_pipeline.py --stage rm
    python train_pipeline.py --stage grpo

Stages
------
1. SFT  — instruction-tune the base model on Alpaca
2. JEPA — train joint embedding predictor on (question, answer) pairs
3. RM   — train reward model on HH-RLHF preference pairs
4. GRPO — RL alignment with combined reward (RM + JEPA + KL)

After training, run the agent:
    python agent.py --question "Your question here"
"""
import os
import argparse
import torch

CKPTS = {
    "base": "checkpoints/ivllm_latest.pt",
    "sft":  "checkpoints/ivllm_sft.pt",
    "jepa": "checkpoints/ivllm_jepa.pt",
    "rm":   "checkpoints/ivllm_rm.pt",
    "grpo": "checkpoints/ivllm_grpo.pt",
}


def _banner(title: str):
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def run_sft(args):
    _banner("Stage 1 — Supervised Fine-Tuning (SFT)")
    from sft import train_sft
    train_sft(
        base_checkpoint=args.base_ckpt,
        lr=args.sft_lr,
        epochs=args.sft_epochs,
        batch_size=args.sft_batch,
        grad_accum=args.sft_grad_accum,
        max_samples=args.sft_samples,
        wandb_enabled=args.wandb,
    )


def run_jepa(args):
    _banner("Stage 2 — JEPA Training")
    from ivllm import IvLLM, device
    from jepa import train_jepa
    model = IvLLM().to(device)
    if os.path.exists(CKPTS["sft"]):
        ckpt = torch.load(CKPTS["sft"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        print(f"JEPA encoders initialised from {CKPTS['sft']}")
    train_jepa(
        base_model=model,
        lr=args.jepa_lr,
        epochs=args.jepa_epochs,
        batch_size=args.jepa_batch,
        ema_momentum=args.jepa_ema,
        wandb_enabled=args.wandb,
    )


def run_rm(args):
    _banner("Stage 3 — Reward Model Training")
    from reward_model import train_reward_model
    train_reward_model(
        sft_checkpoint=CKPTS["sft"],
        lr=args.rm_lr,
        epochs=args.rm_epochs,
        batch_size=args.rm_batch,
        wandb_enabled=args.wandb,
    )


def run_grpo(args):
    _banner("Stage 4 — GRPO Alignment")
    from grpo import train_grpo
    train_grpo(
        sft_checkpoint=CKPTS["sft"],
        rm_checkpoint=CKPTS["rm"],
        jepa_checkpoint=CKPTS["jepa"],
        lr=args.grpo_lr,
        steps=args.grpo_steps,
        prompts_per_step=args.grpo_prompts,
        G=args.grpo_G,
        kl_coef=args.grpo_kl,
        jepa_weight=args.grpo_jepa_w,
        wandb_enabled=args.wandb,
    )


def main():
    parser = argparse.ArgumentParser(
        description="IvLLM Self-Correcting Agent — Training Pipeline"
    )
    parser.add_argument(
        "--stage",
        choices=["sft", "jepa", "rm", "grpo", "all"],
        default="all",
        help="Which stage to run (default: all)",
    )
    parser.add_argument("--base-ckpt", default=CKPTS["base"],
                        help="Pre-trained base model checkpoint")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")

    # SFT
    sft = parser.add_argument_group("SFT")
    sft.add_argument("--sft-lr", type=float, default=2e-5)
    sft.add_argument("--sft-epochs", type=int, default=3)
    sft.add_argument("--sft-batch", type=int, default=4)
    sft.add_argument("--sft-grad-accum", type=int, default=8)
    sft.add_argument("--sft-samples", type=int, default=50000)

    # JEPA
    jp = parser.add_argument_group("JEPA")
    jp.add_argument("--jepa-lr", type=float, default=1e-4)
    jp.add_argument("--jepa-epochs", type=int, default=5)
    jp.add_argument("--jepa-batch", type=int, default=16)
    jp.add_argument("--jepa-ema", type=float, default=0.996)

    # RM
    rm = parser.add_argument_group("Reward Model")
    rm.add_argument("--rm-lr", type=float, default=1e-5)
    rm.add_argument("--rm-epochs", type=int, default=3)
    rm.add_argument("--rm-batch", type=int, default=8)

    # GRPO
    grp = parser.add_argument_group("GRPO")
    grp.add_argument("--grpo-lr", type=float, default=5e-7)
    grp.add_argument("--grpo-steps", type=int, default=1000)
    grp.add_argument("--grpo-prompts", type=int, default=4)
    grp.add_argument("--grpo-G", type=int, default=8)
    grp.add_argument("--grpo-kl", type=float, default=0.04)
    grp.add_argument("--grpo-jepa-w", type=float, default=0.2)

    args = parser.parse_args()
    run_all = args.stage == "all"

    if run_all or args.stage == "sft":
        run_sft(args)
    if run_all or args.stage == "jepa":
        run_jepa(args)
    if run_all or args.stage == "rm":
        run_rm(args)
    if run_all or args.stage == "grpo":
        run_grpo(args)

    _banner("Pipeline complete")
    print("Run the agent:")
    print("  python agent.py --question 'Your question here'")


if __name__ == "__main__":
    main()
