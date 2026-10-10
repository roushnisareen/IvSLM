"""
Stage 4 — GRPO (Group Relative Policy Optimization).

For each training prompt:
  1. Generate G responses with the policy.
  2. Score each response:
       total_reward = rm_reward + jepa_weight * jepa_reward − kl_coef * KL(policy ‖ ref)
  3. Compute group-relative advantage:
       A_i = (r_i − mean(r)) / (std(r) + ε)
  4. Policy gradient loss:
       L = −mean_i [ A_i · Σ_t log π(y_t | x, y<t) ]

No value function is needed (unlike PPO), which greatly simplifies the
implementation while achieving comparable alignment quality for instruction
following and self-correction tasks.

Usage:
    python grpo.py --sft-ckpt checkpoints/ivllm_sft.pt \\
                   --rm-ckpt  checkpoints/ivllm_rm.pt  \\
                   --jepa-ckpt checkpoints/ivllm_jepa.pt \\
                   --steps 1000
"""
import os
import copy
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tiktoken

from base_model import IvLLM, SEQ_LENGTH, device
from generate import generate, compute_log_probs, encode
from reward_model import RewardModel
from jepa import LanguageJEPA

enc = tiktoken.get_encoding("gpt2")
GRPO_CHECKPOINT = "checkpoints/ivllm_grpo.pt"

# Default hyper-parameters
G_DEFAULT = 8
KL_COEF_DEFAULT = 0.04
JEPA_W_DEFAULT = 0.2


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class GRPOTrainer:
    """
    GRPO trainer.

    Combines three reward signals:
      • Reward model  — learned human preference signal
      • JEPA          — representation-level correctness proxy
      • KL penalty    — prevents the policy drifting too far from the SFT reference
    """

    def __init__(
        self,
        policy: IvLLM,
        reward_model: RewardModel,
        jepa: LanguageJEPA,
        ref_policy: IvLLM = None,
        G: int = G_DEFAULT,
        kl_coef: float = KL_COEF_DEFAULT,
        jepa_weight: float = JEPA_W_DEFAULT,
    ):
        self.policy = policy
        self.rm = reward_model
        self.jepa = jepa
        # Freeze a copy of the SFT policy as the KL reference
        self.ref = ref_policy or copy.deepcopy(policy)
        for p in self.ref.parameters():
            p.requires_grad = False
        self.ref.eval()

        self.G = G
        self.kl_coef = kl_coef
        self.jepa_weight = jepa_weight

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _total_reward(self, prompt_ids: list, response_ids: list) -> float:
        """Scalar reward for a single (prompt, response) pair."""
        max_len = SEQ_LENGTH
        p_t = torch.tensor(prompt_ids[-max_len:], dtype=torch.long, device=device).unsqueeze(0)
        r_t = torch.tensor(response_ids[:max_len], dtype=torch.long, device=device).unsqueeze(0)
        full = torch.cat([p_t, r_t], dim=1)[:, :max_len]

        self.rm.eval()
        rm_r = self.rm(full).item()

        self.jepa.eval()
        jepa_r = self.jepa.reward(p_t, r_t).item()

        return rm_r + self.jepa_weight * jepa_r

    # ------------------------------------------------------------------
    # KL penalty
    # ------------------------------------------------------------------

    def _kl_penalty(self, full_ids: torch.Tensor, response_start: int) -> torch.Tensor:
        """
        Approximate KL(policy ‖ ref) per sequence.
        Uses the unbiased estimator: exp(Δ) − Δ − 1  where Δ = log π − log π_ref.
        """
        with torch.no_grad():
            ref_lp = compute_log_probs(self.ref, full_ids, response_start)
        policy_lp = compute_log_probs(self.policy, full_ids, response_start)
        delta = policy_lp - ref_lp.detach()
        kl = (torch.exp(delta) - delta - 1.0)
        return kl.sum(dim=-1)  # (B,)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, prompts: list, optimizer: torch.optim.Optimizer) -> dict:
        """One GRPO optimizer step over a list of prompt strings."""
        self.policy.train()
        total_loss = torch.zeros((), device=device, requires_grad=False)
        all_rewards = []
        n_responses = 0

        for prompt in prompts:
            prompt_ids = encode(prompt)
            if len(prompt_ids) > SEQ_LENGTH - 128:
                prompt_ids = prompt_ids[-(SEQ_LENGTH - 128):]

            # ---- Generate G responses ----
            responses = []
            for _ in range(self.G):
                with torch.no_grad():
                    resp = generate(
                        self.policy, prompt_ids,
                        max_new_tokens=256, temperature=0.9, top_p=0.9,
                        device=device,
                    )
                responses.append(resp)

            # ---- Score ----
            rewards = [self._total_reward(prompt_ids, r) for r in responses]
            r_t = torch.tensor(rewards, dtype=torch.float32, device=device)

            # ---- Group-relative advantages ----
            mean_r = r_t.mean()
            std_r = r_t.std().clamp(min=1e-8)
            advantages = (r_t - mean_r) / std_r  # (G,)

            # ---- Policy gradient ----
            for resp_ids, adv in zip(responses, advantages.tolist()):
                if not resp_ids:
                    continue

                full = torch.tensor(
                    prompt_ids + resp_ids, dtype=torch.long, device=device
                ).unsqueeze(0)[:, :SEQ_LENGTH]

                resp_start = min(len(prompt_ids), SEQ_LENGTH - 1)

                # Per-token log probs for the response slice (with grad)
                logits, _ = self.policy(full[:, :-1])
                lp = F.log_softmax(logits.float(), dim=-1)
                next_ids = full[:, 1:]
                tok_lp = lp.gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)
                resp_lp = tok_lp[:, resp_start - 1:]      # (1, resp_len)

                # KL penalty (policy head has grad here)
                kl = self._kl_penalty(full, resp_start)

                step_loss = -(adv * resp_lp.sum()) + self.kl_coef * kl.sum()
                total_loss = total_loss + step_loss
                n_responses += 1

            all_rewards.extend(rewards)

        if n_responses > 0:
            (total_loss / n_responses).backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        return {
            "loss": (total_loss / max(n_responses, 1)).item(),
            "mean_reward": sum(all_rewards) / max(len(all_rewards), 1),
            "max_reward": max(all_rewards) if all_rewards else 0.0,
            "min_reward": min(all_rewards) if all_rewards else 0.0,
        }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PromptDataset(Dataset):
    def __init__(self, prompts: list):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train_grpo(
    sft_checkpoint: str = "checkpoints/ivllm_sft.pt",
    rm_checkpoint: str = "checkpoints/ivllm_rm.pt",
    jepa_checkpoint: str = "checkpoints/ivllm_jepa.pt",
    lr: float = 5e-7,
    steps: int = 1000,
    prompts_per_step: int = 4,
    G: int = G_DEFAULT,
    kl_coef: float = KL_COEF_DEFAULT,
    jepa_weight: float = JEPA_W_DEFAULT,
    wandb_enabled: bool = False,
) -> IvLLM:

    # ---- Policy ----
    policy = IvLLM().to(device)
    if os.path.exists(sft_checkpoint):
        ckpt = torch.load(sft_checkpoint, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        print(f"Policy loaded from {sft_checkpoint}")

    # ---- Reward model ----
    rm_base = IvLLM().to(device)
    rm = RewardModel(rm_base).to(device)
    if os.path.exists(rm_checkpoint):
        ckpt = torch.load(rm_checkpoint, map_location=device, weights_only=False)
        rm.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        print(f"Reward model loaded from {rm_checkpoint}")
    rm.eval()

    # ---- JEPA ----
    jepa_base = IvLLM().to(device)
    jepa = LanguageJEPA(jepa_base).to(device)
    if os.path.exists(jepa_checkpoint):
        ckpt = torch.load(jepa_checkpoint, map_location=device, weights_only=False)
        jepa.load_state_dict(ckpt.get("jepa_state_dict", ckpt), strict=False)
        print(f"JEPA loaded from {jepa_checkpoint}")
    jepa.eval()

    trainer = GRPOTrainer(
        policy=policy, reward_model=rm, jepa=jepa,
        G=G, kl_coef=kl_coef, jepa_weight=jepa_weight,
    )

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=lr, weight_decay=0.01,
        fused=torch.cuda.is_available(),
    )
    optimizer.zero_grad()

    from sft_data import build_prompts
    prompts = build_prompts()
    loader = DataLoader(PromptDataset(prompts), batch_size=prompts_per_step, shuffle=True)
    loader_iter = iter(loader)

    wandb_run = None
    if wandb_enabled:
        import wandb
        wandb_run = wandb.init(project="ivllm-grpo", name="grpo-alignment")

    print(f"GRPO: {steps} steps | G={G} responses/prompt | lr={lr:.1e}")

    for step in range(1, steps + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        metrics = trainer.train_step(list(batch), optimizer)

        if step % 10 == 0:
            print(
                f"Step {step:5d}/{steps} | Loss {metrics['loss']:+.4f} | "
                f"Reward  mean={metrics['mean_reward']:.3f} "
                f"max={metrics['max_reward']:.3f} "
                f"min={metrics['min_reward']:.3f}"
            )
            if wandb_run:
                wandb_run.log({f"grpo/{k}": v for k, v in metrics.items()}, step=step)

        if step % 200 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({"model_state_dict": policy.state_dict(), "step": step}, GRPO_CHECKPOINT)
            print(f"Checkpoint saved at step {step}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"model_state_dict": policy.state_dict(), "step": steps}, GRPO_CHECKPOINT)
    print(f"GRPO complete → {GRPO_CHECKPOINT}")
    if wandb_run:
        wandb_run.finish()
    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-ckpt", default="checkpoints/ivllm_sft.pt")
    parser.add_argument("--rm-ckpt", default="checkpoints/ivllm_rm.pt")
    parser.add_argument("--jepa-ckpt", default="checkpoints/ivllm_jepa.pt")
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--prompts-per-step", type=int, default=4)
    parser.add_argument("--G", type=int, default=8)
    parser.add_argument("--kl-coef", type=float, default=0.04)
    parser.add_argument("--jepa-weight", type=float, default=0.2)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    train_grpo(
        sft_checkpoint=args.sft_ckpt,
        rm_checkpoint=args.rm_ckpt,
        jepa_checkpoint=args.jepa_ckpt,
        lr=args.lr,
        steps=args.steps,
        prompts_per_step=args.prompts_per_step,
        G=args.G,
        kl_coef=args.kl_coef,
        jepa_weight=args.jepa_weight,
        wandb_enabled=args.wandb,
    )
