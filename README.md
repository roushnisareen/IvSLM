# IvSLM

A small language model (≈1.17B params) pretrained from scratch on FineWeb (+ a small math mix),
with an integrated [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
evaluation pipeline (both standalone and inline during training).

## Layout

| Path | Description |
| --- | --- |
| `ivllm.py` | Model definition + DDP `torchrun` training loop (resume, WSD schedule, inline eval hooks). |
| `eval_ivllm.py` | Standalone lm-eval-harness wrapper for a checkpoint (loglikelihood-style tasks). |
| `data.py` | Tokenized data loading / sharded streaming dataloader. |
| `scripts/build_release.py` | Packages a checkpoint into a HF-style release dir. |
| `release/ivllm-671m-fineweb12b/` | Example release: `config.json`, `modeling_ivllm.py`, `README.md` (weights excluded). |
| `requirements.txt` | Frozen environment (PyTorch, lm-eval 0.4.x, datasets, tiktoken, wandb, ...). |

Tokenizer: tiktoken `gpt2` (vocab 50257).

## Training

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

torchrun --standalone --nproc_per_node=8 ivllm.py
```

Behaviour is driven by `IVLLM_*` environment variables (token budget, LR schedule, data mix,
batch sizes, and inline-eval cadence/tasks). Checkpoints are written to `checkpoints/`
every 500 steps and training auto-resumes from `checkpoints/ivllm_latest.pt`.

## Evaluation

```bash
python eval_ivllm.py \
  --ckpt checkpoints/ivllm_latest.pt \
  --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande,lambada_openai,\
openbookqa,boolq,sciq,copa,commonsense_qa,race,mmlu \
  --batch-size 16 --device cuda:0 --output results.json
```

Only loglikelihood-style (multiple-choice / perplexity) tasks are supported.

> Note: with `datasets>=5`, tasks that ship a dataset loading script
> (`social_iqa`, `logiqa`, `mathqa`) no longer load and are excluded.

## Artifacts

`eval_v2_step2500.json` contains a full evaluation of the step-2500 checkpoint across the
13-task suite above (including the 57 MMLU subtasks).

## Large files (not tracked)

Checkpoints (`checkpoints/`, `*.pt`, `*.safetensors`), datasets (`fineweb100B/`),
the virtualenv (`.venv/`), and `wandb/` logs are intentionally git-ignored.
