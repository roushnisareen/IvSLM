---
license: apache-2.0
language:
- en
tags:
- pretrained
- causal-lm
- gpt2-tokenizer
- fineweb
library_name: pytorch
---

# IvLLM 671M (fineweb-12B checkpoint)

A 671 M parameter decoder-only LM trained from scratch on ~12 B tokens of
[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
([kjj0/fineweb100B-gpt2](https://huggingface.co/datasets/kjj0/fineweb100B-gpt2) pre-tokenized shards).
GPT-2 (tiktoken) tokenizer; vocab 50 304.

This is an early pretrain checkpoint — useful as a small base model or a starting point for
further training. Final val loss on FineWeb: **2.95**.

## Architecture

| | |
|---|---|
| dim | 1536 |
| blocks | 24 |
| Q heads / KV groups | 24 / 6 (GQA, ratio 4) |
| head_dim | 64 |
| seq_len | 2048 |
| ffn | SwiGLU (hidden 4096) |
| norm | RMSNorm |
| pos | RoPE θ=10 000 |
| tied embeddings | yes |
| params | 671 M |

## Files
- `model.safetensors` — bf16 weights (749 M params, tied embed+lm_head saved separately)
- `config.json` — architecture + training summary
- `modeling_ivllm.py` — self-contained PyTorch model definition

## Quick start

```python
import torch
from safetensors.torch import load_file
from modeling_ivllm import IvLLM
import tiktoken

model = IvLLM().to('cuda').eval()
sd = load_file('model.safetensors')
model.load_state_dict(sd, strict=True)

enc = tiktoken.get_encoding('gpt2')
ids = torch.tensor([enc.encode_ordinary("The capital of France is")], device='cuda')

# Greedy decoding for 30 tokens
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    for _ in range(30):
        logits, _ = model(ids[:, -2048:])
        next_id = logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
print(enc.decode(ids[0].tolist()))
```

## Training details

| | |
|---|---|
| Tokens seen | ~12.24 B |
| Global batch | 480 × 2048 ≈ 983 k tokens/step |
| Optimizer | AdamW (β=0.9/0.95, wd=0.1, fused) |
| LR schedule | cosine, peak 3e-4 → min 3e-5, 2 000 step warmup |
| Hardware | 8 × H100 80 GB SXM |
| Steady-state | ~760 k tok/s, ~54 % MFU |
| Final train loss | 2.94 |
| Final val loss | 2.95 |

Trained as part of the [ivllm](https://github.com/) workspace. Subsequent training runs use a
larger 1.17 B architecture with QK-Norm, value-residual learning, per-head sigmoid gating, and
output z-loss — published separately.
