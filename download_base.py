"""
Download the pretrained IvLLM 671M weights from Hugging Face Hub
and save them as a PyTorch checkpoint ready for SFT/JEPA/RM/GRPO.

Usage:
    python download_base.py
    python download_base.py --model-id Samarth0710/ivllm-671m-fineweb12b
    python download_base.py --local release/ivllm-671m-fineweb12b/model.safetensors
"""
import os
import argparse
import torch
from safetensors.torch import load_file

CHECKPOINT_DIR = "checkpoints"
OUTPUT_PATH = os.path.join(CHECKPOINT_DIR, "ivllm_base.pt")

DEFAULT_MODEL_ID = "Samarth0710/ivllm-671m-fineweb12b"


def download_from_hub(model_id: str) -> dict:
    from huggingface_hub import hf_hub_download
    print(f"Downloading weights from HuggingFace: {model_id}")
    local_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
    return load_file(local_path)


def load_from_local(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Safetensors file not found: {path}")
    print(f"Loading weights from local file: {path}")
    return load_file(path)


def convert_and_save(state_dict: dict, output_path: str):
    """
    The safetensors release uses tied embeddings stored separately.
    Ensure output_layer.weight is present (tied to embeddings.weight).
    """
    # Some releases omit output_layer.weight since it's tied — add it back
    if "output_layer.weight" not in state_dict and "embeddings.weight" in state_dict:
        state_dict["output_layer.weight"] = state_dict["embeddings.weight"]
        print("  Restored tied output_layer.weight from embeddings.weight")

    # Verify the checkpoint loads correctly
    from base_model import IvLLM
    print("Verifying weights against IvLLM 671M architecture …")
    model = IvLLM()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys  : {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    if not missing and not unexpected:
        print("  All keys matched perfectly.")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({"model_state_dict": state_dict}, output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"\nCheckpoint saved → {output_path}  ({size_mb:.0f} MB)")
    print("\nNext step — run SFT:")
    print(f"  python sft.py --base-ckpt {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download IvLLM 671M pretrained weights")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                        help="HuggingFace model ID (default: Samarth0710/ivllm-671m-fineweb12b)")
    parser.add_argument("--local", default=None,
                        help="Path to a local model.safetensors file (skips HF download)")
    parser.add_argument("--output", default=OUTPUT_PATH,
                        help=f"Output checkpoint path (default: {OUTPUT_PATH})")
    args = parser.parse_args()

    if os.path.exists(args.output):
        print(f"Checkpoint already exists at {args.output} — skipping download.")
        print("Delete it to re-download.")
        return

    if args.local:
        state_dict = load_from_local(args.local)
    else:
        state_dict = download_from_hub(args.model_id)

    convert_and_save(state_dict, args.output)


if __name__ == "__main__":
    main()
