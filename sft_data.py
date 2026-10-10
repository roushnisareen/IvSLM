"""SFT, JEPA, and reward-model dataset utilities."""
import torch
from torch.utils.data import Dataset
import tiktoken

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token  # 50256

PROMPT_FORMAT = "Human: {instruction}\n\nAssistant: "


def format_instruction(instruction: str, input_text: str = "") -> str:
    if input_text and input_text.strip():
        instruction = f"{instruction}\n\nInput: {input_text}"
    return PROMPT_FORMAT.format(instruction=instruction)


# ---------------------------------------------------------------------------
# SFT dataset
# ---------------------------------------------------------------------------

class InstructionDataset(Dataset):
    """
    Alpaca-style instruction dataset.
    Loss mask is 1 on response tokens only so the model learns to follow
    instructions without memorising prompts.
    """

    def __init__(
        self,
        max_seq_len: int = 2048,
        split: str = "train",
        max_samples: int = None,
        dataset_name: str = "tatsu-lab/alpaca",
    ):
        from datasets import load_dataset
        raw = load_dataset(dataset_name, split=split)
        if max_samples:
            raw = raw.select(range(min(max_samples, len(raw))))

        self.examples = []
        for item in raw:
            prompt = format_instruction(item["instruction"], item.get("input", ""))
            response = item["output"]
            prompt_ids = enc.encode(prompt)
            response_ids = enc.encode(response) + [EOT]
            full_ids = prompt_ids + response_ids
            if len(full_ids) > max_seq_len:
                continue
            loss_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
            self.examples.append((full_ids, loss_mask))

        print(f"InstructionDataset: {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids, mask = self.examples[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


def sft_collate(batch):
    ids_list, mask_list = zip(*batch)
    max_len = max(t.size(0) for t in ids_list)
    B = len(batch)
    ids_pad = torch.zeros(B, max_len, dtype=torch.long)
    mask_pad = torch.zeros(B, max_len, dtype=torch.bool)
    for i, (ids, mask) in enumerate(zip(ids_list, mask_list)):
        L = ids.size(0)
        ids_pad[i, :L] = ids
        mask_pad[i, :L] = mask
    return ids_pad, mask_pad


# ---------------------------------------------------------------------------
# QA pairs for JEPA training
# ---------------------------------------------------------------------------

def build_qa_pairs(max_samples: int = 20000) -> list:
    """(question, answer) pairs from Alpaca for JEPA training."""
    from datasets import load_dataset
    raw = load_dataset("tatsu-lab/alpaca", split="train")
    pairs = []
    for item in raw:
        if max_samples and len(pairs) >= max_samples:
            break
        q = item["instruction"]
        if item.get("input", "").strip():
            q = f"{q}\n{item['input']}"
        a = item["output"]
        if q.strip() and a.strip():
            pairs.append((q, a))
    return pairs


# ---------------------------------------------------------------------------
# Preference pairs for reward model training
# ---------------------------------------------------------------------------

def build_preference_pairs(max_samples: int = 20000) -> list:
    """{'chosen': str, 'rejected': str} dicts from Anthropic HH-RLHF."""
    from datasets import load_dataset
    raw = load_dataset("Anthropic/hh-rlhf", split="train")
    pairs = []
    for item in raw:
        if max_samples and len(pairs) >= max_samples:
            break
        pairs.append({"chosen": item["chosen"], "rejected": item["rejected"]})
    return pairs


# ---------------------------------------------------------------------------
# Prompt list for GRPO training
# ---------------------------------------------------------------------------

def build_prompts(max_samples: int = 10000) -> list:
    """Plain question strings from Alpaca for GRPO prompt sampling."""
    from datasets import load_dataset
    raw = load_dataset("tatsu-lab/alpaca", split="train")
    prompts = []
    for item in raw:
        if len(prompts) >= max_samples:
            break
        q = item["instruction"]
        if item.get("input", "").strip():
            q = f"{q}\n\n{item['input']}"
        prompts.append(q)
    return prompts
