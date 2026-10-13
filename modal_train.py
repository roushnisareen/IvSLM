"""
Modal training script for IvLLM Self-Correcting Agent.

Runs the full 4-stage pipeline on a Modal L40S GPU:
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
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.2",
        "numpy<2",
        "pyarrow==15.0.2",
        "pandas==2.2.2",
        "einops==0.8.2",
        "tiktoken==0.6.0",
        "datasets==2.19.0",
        "safetensors==0.4.3",
        "huggingface-hub==0.23.0",
        "wandb",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
)

# Copy the repo's .py files into the container (replaces modal.Mount, removed in Modal 1.0)
image = base_image.add_local_dir(".", remote_path="/app", ignore=lambda p: p.suffix != ".py")

# Benchmark image: adds lm-eval (pinned to versions compatible with torch 2.2 / datasets 2.19)
eval_image = (
    base_image
    .pip_install("lm-eval==0.4.3", "transformers==4.44.2", "accelerate==0.33.0", "peft==0.12.0")
    .add_local_dir(".", remote_path="/app", ignore=lambda p: p.suffix != ".py")
)

# 48 GB GPU — JEPA and GRPO hold several model copies and don't fit on a 24 GB A10G
GPU = "L40S"
BASE_MODEL_ID = "Samarth0710/ivllm-671m-fineweb12b"

# ---------------------------------------------------------------------------
# Persistent volume — checkpoints survive across runs
# ---------------------------------------------------------------------------
volume = modal.Volume.from_name("ivllm-checkpoints", create_if_missing=True)
VOLUME_PATH = "/checkpoints"

app = modal.App("ivllm-self-correcting-agent")

# ---------------------------------------------------------------------------
# Stage 0: Download pretrained 671M weights
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=600,
)
def download_weights(model_id: str = BASE_MODEL_ID):
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

    # Fail loudly if the weights don't match the IvLLM architecture
    from base_model import IvLLM
    IvLLM().load_state_dict(sd, strict=True)
    print(f"All {len(sd)} tensors match IvLLM 671M.")

    torch.save({"model_state_dict": sd}, output)
    volume.commit()
    print(f"Saved → {output}")


# ---------------------------------------------------------------------------
# Stage 1: SFT
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu=GPU,
    volumes={VOLUME_PATH: volume},
    timeout=3600 * 6,
)
def sft(epochs: int = 3, lr: float = 2e-5, batch_size: int = 4, max_samples: int = 50000):
    import sys, os
    sys.path.insert(0, "/app")
    # sft.py saves to checkpoints/ivllm_sft.pt relative to cwd — run from the
    # volume so the checkpoint persists for the later stages
    os.chdir(VOLUME_PATH)

    base = f"{VOLUME_PATH}/checkpoints/ivllm_base.pt"
    if not os.path.exists(base):
        raise FileNotFoundError(f"{base} missing — run download_weights first")

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
# Compare base vs SFT: held-out Alpaca loss + sample answers
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu=GPU,
    volumes={VOLUME_PATH: volume},
    timeout=1800,
)
def compare_sft(n_eval: int = 500):
    import sys, torch
    import torch.nn.functional as F
    sys.path.insert(0, "/app")

    from datasets import load_dataset
    from base_model import IvLLM, device
    from generate import generate, enc, EOT_TOKEN
    from sft_data import format_instruction

    # SFT trains on the first 50k Alpaca rows; evaluate on rows it never saw
    rows = load_dataset("tatsu-lab/alpaca", split="train").select(range(51000, 51000 + n_eval))
    prompts = [
        "What is the capital of France?",
        "Give three tips for staying healthy.",
        "Explain what photosynthesis is in simple terms.",
        "Write a short poem about the ocean.",
        "What is 15 + 27?",
    ]

    def load(name):
        m = IvLLM().to(device)
        ckpt = torch.load(f"{VOLUME_PATH}/checkpoints/{name}", map_location=device, weights_only=False)
        m.load_state_dict(ckpt["model_state_dict"], strict=True)
        return m.eval()

    @torch.no_grad()
    def response_loss(model):
        total, count = 0.0, 0
        for r in rows:
            p = enc.encode(format_instruction(r["instruction"], r["input"]))
            a = enc.encode(r["output"]) + [EOT_TOKEN]
            ids = torch.tensor((p + a)[:2048], device=device).unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = model(ids[:, :-1])
            nll = F.cross_entropy(logits[0].float(), ids[0, 1:], reduction="none")[len(p) - 1:]
            total += nll.sum().item()
            count += nll.numel()
        return total / count

    results = {}
    for label, name in [("BASE", "ivllm_base.pt"), ("SFT", "ivllm_sft.pt")]:
        model = load(name)
        loss = response_loss(model)
        results[label] = loss
        print(f"\n========== {label} ==========")
        print(f"Held-out Alpaca answer loss: {loss:.3f}  (perplexity {torch.tensor(loss).exp():.1f})")
        torch.manual_seed(0)
        for q in prompts:
            out = generate(model, enc.encode(format_instruction(q)), max_new_tokens=120,
                           temperature=0.7, top_p=0.9, device=device)
            text = enc.decode([t for t in out if t != EOT_TOKEN]).strip()
            print(f"\nQ: {q}\nA: {text}")
        del model
        torch.cuda.empty_cache()

    print("\n========== SUMMARY ==========")
    print(f"Answer loss  BASE {results['BASE']:.3f}  →  SFT {results['SFT']:.3f}  (lower is better)")
    return results


# ---------------------------------------------------------------------------
# Stage 1b: SFT on Alpaca + math (GSM8K, MetaMathQA, synthetic arithmetic)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu=GPU,
    volumes={VOLUME_PATH: volume},
    timeout=3600 * 6,
)
def sft_math(epochs: int = 2, lr: float = 2e-5, batch_size: int = 4,
             n_alpaca: int = 30000, n_metamath: int = 25000, n_arithmetic: int = 20000,
             output_name: str = "ivllm_sft_math.pt"):
    import sys, os
    sys.path.insert(0, "/app")
    os.chdir(VOLUME_PATH)

    base = f"{VOLUME_PATH}/checkpoints/ivllm_base.pt"
    if not os.path.exists(base):
        raise FileNotFoundError(f"{base} missing — run download_weights first")

    from base_model import SEQ_LENGTH
    from math_data import build_math_sft_pairs
    from sft_data import PairDataset
    from sft import train_sft

    pairs = build_math_sft_pairs(n_alpaca=n_alpaca, n_metamath=n_metamath, n_arithmetic=n_arithmetic)
    train_sft(
        base_checkpoint=base,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        dataset=PairDataset(pairs, max_seq_len=SEQ_LENGTH),
        output_path=f"checkpoints/{output_name}",
    )
    volume.commit()
    print("Math SFT complete.")


# ---------------------------------------------------------------------------
# Compare checkpoints: answer loss, arithmetic accuracy, GSM8K accuracy
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu=GPU,
    volumes={VOLUME_PATH: volume},
    timeout=3600 * 2,
)
def compare(checkpoints: str = "ivllm_base.pt,ivllm_sft.pt,ivllm_sft_math.pt",
            n_loss: int = 300, n_arith: int = 200, n_gsm: int = 200):
    import sys, json, torch
    import torch.nn.functional as F
    sys.path.insert(0, "/app")

    from datasets import load_dataset
    from base_model import IvLLM, device
    from generate import enc, EOT_TOKEN
    from sft_data import format_instruction
    from math_data import arithmetic_eval_set, gsm8k_pairs, extract_answer, is_correct

    alpaca = load_dataset("tatsu-lab/alpaca", split="train").select(range(51000, 51000 + n_loss))
    arith = arithmetic_eval_set(n_arith)
    gsm = [(q, final) for q, _, final in gsm8k_pairs("test")[:n_gsm]]
    samples = [
        "What is the capital of France?",
        "Give three tips for staying healthy.",
        "What is 15 + 27?",
        "Tom has 12 apples. He gives 5 to his friend and buys 8 more. How many apples does he have now?",
    ]
    stop_ids = enc.encode("\n\nHuman:")

    @torch.no_grad()
    def greedy(model, prompt, max_new_tokens):
        ids = torch.tensor([enc.encode(format_instruction(prompt))], device=device)
        out = []
        for _ in range(max_new_tokens):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = model(ids[:, -2048:])
            tok = int(logits[0, -1].argmax())
            if tok == EOT_TOKEN:
                break
            out.append(tok)
            if out[-len(stop_ids):] == stop_ids:
                out = out[:-len(stop_ids)]
                break
            ids = torch.cat([ids, torch.tensor([[tok]], device=device)], dim=1)
        return enc.decode(out).strip()

    @torch.no_grad()
    def answer_loss(model):
        total, count = 0.0, 0
        for r in alpaca:
            p = enc.encode(format_instruction(r["instruction"], r["input"]))
            a = enc.encode(r["output"]) + [EOT_TOKEN]
            ids = torch.tensor((p + a)[:2048], device=device).unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = model(ids[:, :-1])
            nll = F.cross_entropy(logits[0].float(), ids[0, 1:], reduction="none")[len(p) - 1:]
            total += nll.sum().item()
            count += nll.numel()
        return total / count

    def accuracy(model, items, max_new_tokens):
        hits, first = 0, None
        for q, target in items:
            text = greedy(model, q, max_new_tokens)
            hits += is_correct(extract_answer(text), target)
            first = first or (q, text, target)
        return hits / len(items), first

    results = {}
    for name in checkpoints.split(","):
        model = IvLLM().to(device)
        ckpt = torch.load(f"{VOLUME_PATH}/checkpoints/{name}", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()

        loss = answer_loss(model)
        arith_acc, arith_ex = accuracy(model, arith, 200)
        gsm_acc, gsm_ex = accuracy(model, gsm, 256)
        results[name] = {"answer_loss": loss, "arithmetic_acc": arith_acc, "gsm8k_acc": gsm_acc}

        print(f"\n========== {name} ==========")
        print(f"Alpaca answer loss : {loss:.3f}")
        print(f"Arithmetic accuracy: {arith_acc:.1%}  ({len(arith)} questions)")
        print(f"GSM8K accuracy     : {gsm_acc:.1%}  ({len(gsm)} questions)")
        for q, text, target in [arith_ex, gsm_ex]:
            print(f"\nQ: {q}\nA: {text[:600]}\n(correct answer: {target})")
        for q in samples:
            print(f"\nQ: {q}\nA: {greedy(model, q, 200)[:600]}")
        del model
        torch.cuda.empty_cache()

    print("\n========== SUMMARY ==========")
    print(f"{'checkpoint':<24}{'answer loss':>12}{'arithmetic':>12}{'GSM8K':>10}")
    for name, r in results.items():
        print(f"{name:<24}{r['answer_loss']:>12.3f}{r['arithmetic_acc']:>12.1%}{r['gsm8k_acc']:>10.1%}")

    with open(f"{VOLUME_PATH}/compare_results.json", "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    return results


# ---------------------------------------------------------------------------
# General benchmarks (lm-eval): does SFT keep the model's general knowledge?
# ---------------------------------------------------------------------------
BENCHMARK_TASKS = ("hellaswag,arc_easy,arc_challenge,piqa,winogrande,lambada_openai,"
                   "openbookqa,boolq,sciq,copa,commonsense_qa,race,mmlu")


@app.function(
    image=eval_image,
    gpu=GPU,
    volumes={VOLUME_PATH: volume},
    timeout=3600 * 3,
)
def benchmark(checkpoints: str = "ivllm_base.pt,ivllm_sft.pt,ivllm_sft_math_v2.pt",
              tasks: str = BENCHMARK_TASKS, limit: int = 0):
    import sys, json, torch
    sys.path.insert(0, "/app")

    from lm_eval import evaluator
    from base_model import IvLLM, device
    from eval_ivllm import IvLLMWrapper

    task_list = tasks.split(",")
    table = {}
    for name in checkpoints.split(","):
        model = IvLLM()
        ckpt = torch.load(f"{VOLUME_PATH}/checkpoints/{name}", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model = model.to(device).to(torch.bfloat16).eval()

        lm = IvLLMWrapper.from_model(model, device=device, batch_size=16)
        res = evaluator.simple_evaluate(model=lm, tasks=task_list, num_fewshot=0,
                                        limit=limit or None, batch_size=16)
        table[name] = {t: res["results"][t].get("acc,none") for t in task_list}
        print(f"\n=== {name} ===")
        for t, v in table[name].items():
            print(f"{t:18s} {v:.1%}")
        del model, lm
        torch.cuda.empty_cache()

    print("\n========== BENCHMARK SUMMARY (accuracy) ==========")
    names = list(table)
    print(f"{'task':18s}" + "".join(f"{n.replace('.pt', ''):>22s}" for n in names))
    for t in task_list:
        print(f"{t:18s}" + "".join(f"{table[n][t]:>22.1%}" for n in names))

    with open(f"{VOLUME_PATH}/benchmark_results.json", "w") as f:
        json.dump(table, f, indent=2)
    volume.commit()
    return table


# ---------------------------------------------------------------------------
# Stage 2: JEPA
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu=GPU,
    volumes={VOLUME_PATH: volume},
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
    gpu=GPU,
    volumes={VOLUME_PATH: volume},
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
    gpu=GPU,
    volumes={VOLUME_PATH: volume},
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
    gpu=GPU,
    volumes={VOLUME_PATH: volume},
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
    gpu=GPU,
    volumes={VOLUME_PATH: volume},
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
