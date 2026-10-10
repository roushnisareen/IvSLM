"""
Modal training script for IvLLM Self-Correcting Agent.

Runs the full 4-stage pipeline on a Modal A10G GPU:
  1. Download pretrained 671M weights from HuggingFace
  2. SFT  — instruction fine-tuning on Alpaca
  3. JEPA — joint embedding predictor training
  4. RM   — reward model on HH-RLHF preference pairs
  5. GRPO — RL alignment

Usage:
    pip install modal
    modal setup          # one-time auth
    modal run modal_train.py            # run full pipeline
    modal run modal_train.py::sft       # run one stage
    modal run modal_train.py::agent     # test the trained agent
"""
import modal
import os

# ---------------------------------------------------------------------------
# Image: all dependencies pre-installed
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.2",
        "torchvision",
        "einops==0.8.2",
        "tiktoken==0.6.0",
        "datasets==2.19.0",
        "safetensors==0.4.3",
        "huggingface-hub==0.23.0",
        "wandb",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .run_commands("pip install flash-attn --no-build-isolation || true")
)

# ---------------------------------------------------------------------------
# Persistent volume — checkpoints survive across runs
# ---------------------------------------------------------------------------
volume = modal.Volume.from_name("ivllm-checkpoints", create_if_missing=True)
VOLUME_PATH = "/checkpoints"

app = modal.App("ivllm-self-correcting-agent")

# ---------------------------------------------------------------------------
# Helper: copy repo source into the container
# ---------------------------------------------------------------------------
repo_mount = modal.Mount.from_local_dir(
    ".",
    remote_path="/app",
    condition=lambda p: p.endswith(".py"),
)


# ---------------------------------------------------------------------------
# Stage 0: Download pretrained 671M weights
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    mounts=[repo_mount],
    timeout=600,
)
def download_weights(model_id: str = "roushnisareen/ivllm-671m-fineweb12b"):
    import sys, os
    sys.path.insert(0, "/app")
    os.chdir("/app")
    os.makedirs(f"{VOLUME_PATH}/checkpoints", exist_ok=True)

    output = f"{VOLUME_PATH}/checkpoints/ivllm_base.pt"
    if os.path.exists(output):
        print(f"Base weights already at {output} — skipping download.")
        return

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import torch

    print(f"Downloading {model_id} …")
    local = hf_hub_download(repo_id=model_id, filename="model.safetensors")
    sd = load_file(local)

    if "output_layer.weight" not in sd and "embeddings.weight" in sd:
        sd["output_layer.weight"] = sd["embeddings.weight"]

    torch.save({"model_state_dict": sd}, output)
    volume.commit()
    print(f"Saved → {output}")


# ---------------------------------------------------------------------------
# Stage 1: SFT
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",
    volumes={VOLUME_PATH: volume},
    mounts=[repo_mount],
    timeout=3600 * 6,
)
def sft(epochs: int = 3, lr: float = 2e-5, batch_size: int = 4, max_samples: int = 50000):
    import sys, os
    sys.path.insert(0, "/app")
    os.chdir("/app")

    # Point checkpoints at the volume
    os.environ["IVLLM_CKPT_DIR"] = f"{VOLUME_PATH}/checkpoints"

    from sft import train_sft
    train_sft(
        base_checkpoint=f"{VOLUME_PATH}/checkpoints/ivllm_base.pt",
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        max_samples=max_samples,
    )
    volume.commit()
    print("SFT complete.")


# ---------------------------------------------------------------------------
# Stage 2: JEPA
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",
    volumes={VOLUME_PATH: volume},
    mounts=[repo_mount],
    timeout=3600 * 4,
)
def jepa(epochs: int = 5, lr: float = 1e-4, batch_size: int = 32):
    import sys, os, torch
    sys.path.insert(0, "/app")
    os.chdir("/app")

    from base_model import IvLLM, device
    from jepa import train_jepa

    model = IvLLM().to(device)
    ckpt_path = f"{VOLUME_PATH}/checkpoints/ivllm_sft.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        print(f"Loaded SFT weights from {ckpt_path}")

    train_jepa(base_model=model, lr=lr, epochs=epochs, batch_size=batch_size)

    import shutil
    shutil.copy("checkpoints/ivllm_jepa.pt", f"{VOLUME_PATH}/checkpoints/ivllm_jepa.pt")
    volume.commit()
    print("JEPA complete.")


# ---------------------------------------------------------------------------
# Stage 3: Reward Model
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",
    volumes={VOLUME_PATH: volume},
    mounts=[repo_mount],
    timeout=3600 * 4,
)
def reward_model(epochs: int = 3, lr: float = 1e-5, batch_size: int = 8):
    import sys, os
    sys.path.insert(0, "/app")
    os.chdir("/app")

    from reward_model import train_reward_model
    train_reward_model(
        sft_checkpoint=f"{VOLUME_PATH}/checkpoints/ivllm_sft.pt",
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
    )

    import shutil
    shutil.copy("checkpoints/ivllm_rm.pt", f"{VOLUME_PATH}/checkpoints/ivllm_rm.pt")
    volume.commit()
    print("Reward model complete.")


# ---------------------------------------------------------------------------
# Stage 4: GRPO
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",
    volumes={VOLUME_PATH: volume},
    mounts=[repo_mount],
    timeout=3600 * 8,
)
def grpo(steps: int = 1000, lr: float = 5e-7, G: int = 8):
    import sys, os
    sys.path.insert(0, "/app")
    os.chdir("/app")

    from grpo import train_grpo
    train_grpo(
        sft_checkpoint=f"{VOLUME_PATH}/checkpoints/ivllm_sft.pt",
        rm_checkpoint=f"{VOLUME_PATH}/checkpoints/ivllm_rm.pt",
        jepa_checkpoint=f"{VOLUME_PATH}/checkpoints/ivllm_jepa.pt",
        lr=lr,
        steps=steps,
        G=G,
    )

    import shutil
    shutil.copy("checkpoints/ivllm_grpo.pt", f"{VOLUME_PATH}/checkpoints/ivllm_grpo.pt")
    volume.commit()
    print("GRPO complete.")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",
    volumes={VOLUME_PATH: volume},
    mounts=[repo_mount],
    timeout=3600 * 20,
)
def train_all():
    """Run all 4 stages end-to-end."""
    download_weights.remote()
    sft.remote()
    jepa.remote()
    reward_model.remote()
    grpo.remote()
    print("\n=== Full pipeline complete! ===")
    print("Run the agent: modal run modal_train.py::run_agent")


# ---------------------------------------------------------------------------
# Test the trained agent
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",
    volumes={VOLUME_PATH: volume},
    mounts=[repo_mount],
    timeout=300,
)
def run_agent(question: str = "What is 17 × 23? Show your working."):
    import sys, os
    sys.path.insert(0, "/app")
    os.chdir("/app")

    from agent import load_agent
    agent = load_agent(
        model_checkpoint=f"{VOLUME_PATH}/checkpoints/ivllm_grpo.pt",
        jepa_checkpoint=f"{VOLUME_PATH}/checkpoints/ivllm_jepa.pt",
    )
    result = agent.run(question)
    print("\nFINAL ANSWER:", result["answer"])
    print(f"Iterations: {result['iterations']}")
    return result["answer"]


# ---------------------------------------------------------------------------
# Local entrypoint — runs the full pipeline when you do `modal run modal_train.py`
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main():
    print("Starting IvLLM Self-Correcting Agent training pipeline on Modal …")
    print("Step 1/5: Downloading pretrained 671M weights")
    download_weights.remote()
    print("Step 2/5: SFT")
    sft.remote()
    print("Step 3/5: JEPA")
    jepa.remote()
    print("Step 4/5: Reward Model")
    reward_model.remote()
    print("Step 5/5: GRPO")
    grpo.remote()
    print("\nDone! Test it with:")
    print("  modal run modal_train.py::run_agent --question 'Your question'")
