"""Text generation utilities for IvLLM."""
import torch
import torch.nn.functional as F
import tiktoken

enc = tiktoken.get_encoding("gpt2")
EOT_TOKEN = enc.eot_token  # 50256
MAX_CTX = 2048


@torch.no_grad()
def generate(
    model,
    prompt_tokens: list,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.9,
    device: str = "cuda",
) -> list:
    """Autoregressive generation with nucleus sampling. Returns new token ids only."""
    model.eval()
    tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

    generated = []
    for _ in range(max_new_tokens):
        ctx = tokens[:, -MAX_CTX:]
        logits, _ = model(ctx)
        logits = logits[:, -1, :].float() / max(temperature, 1e-8)
        logits = _top_p_filter(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        tok_id = next_tok.item()
        tokens = torch.cat([tokens, next_tok], dim=1)
        generated.append(tok_id)
        if tok_id == EOT_TOKEN:
            break

    return generated


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Shift right so the token that pushes over threshold is kept
    sorted_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits = sorted_logits.masked_fill(sorted_remove, float("-inf"))
    return logits.scatter(1, sorted_idx, sorted_logits)


def compute_log_probs(
    model,
    full_ids: torch.Tensor,
    response_start: int,
) -> torch.Tensor:
    """
    Per-token log probs for the response slice of full_ids.

    full_ids:       (B, T) — prompt + response concatenated
    response_start: token index where response begins
    returns:        (B, response_len) log probabilities
    """
    logits, _ = model(full_ids[:, :-1])          # (B, T-1, V)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    next_ids = full_ids[:, 1:]                    # (B, T-1)
    token_lp = log_probs.gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)
    return token_lp[:, response_start - 1:]       # (B, response_len)


def encode(text: str) -> list:
    return enc.encode(text)


def decode(tokens: list) -> str:
    return enc.decode(tokens)
