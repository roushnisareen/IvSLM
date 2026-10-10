"""
Self-Correcting AI Agent — inference entry point.

The agent wraps the GRPO-aligned IvLLM with an iterative critique-and-revise
loop guided by JEPA representation quality:

  1. Generate an initial answer.
  2. Evaluate quality via JEPA cosine similarity (question ↔ answer repr).
  3. If quality < threshold: self-critique then revise.
  4. Repeat up to max_iterations.

Usage:
    python agent.py --question "Explain gradient descent step by step."
    python agent.py --model-ckpt checkpoints/ivllm_grpo.pt \\
                    --jepa-ckpt  checkpoints/ivllm_jepa.pt  \\
                    --max-iterations 3 \\
                    --question "What is 17 × 23? Show your work."
"""
import os
import argparse
import torch
import tiktoken

from base_model import IvLLM, SEQ_LENGTH, device
from generate import generate, encode, decode
from jepa import LanguageJEPA

enc = tiktoken.get_encoding("gpt2")

CRITIQUE_TEMPLATE = (
    "Question: {question}\n\n"
    "Proposed Answer: {answer}\n\n"
    "Carefully review the proposed answer. "
    "Is it correct, complete, and well-reasoned? "
    "List every error or gap you find.\n\n"
    "Critique:"
)

REVISION_TEMPLATE = (
    "Question: {question}\n\n"
    "Proposed Answer: {answer}\n\n"
    "Critique: {critique}\n\n"
    "Revised Answer (fully addressing all critique points):"
)


class SelfCorrectingAgent:
    """
    Iterative self-correction agent.

    The JEPA score acts as an automatic quality gate: if the generated
    response already lands in the region of "correct answer" representations
    (cosine sim ≥ threshold), no correction round is triggered.
    """

    def __init__(
        self,
        model: IvLLM,
        jepa: LanguageJEPA,
        max_iterations: int = 3,
        jepa_threshold: float = 0.7,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_answer_tokens: int = 512,
        max_critique_tokens: int = 256,
        verbose: bool = True,
    ):
        self.model = model
        self.jepa = jepa
        self.max_iterations = max_iterations
        self.jepa_threshold = jepa_threshold
        self.temperature = temperature
        self.top_p = top_p
        self.max_answer_tokens = max_answer_tokens
        self.max_critique_tokens = max_critique_tokens
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gen(self, prompt: str, max_tokens: int) -> str:
        ids = encode(prompt)
        out = generate(
            self.model, ids,
            max_new_tokens=max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            device=device,
        )
        return decode(out).strip()

    def _jepa_score(self, question: str, answer: str) -> float:
        q = torch.tensor(encode(question)[:SEQ_LENGTH], dtype=torch.long, device=device).unsqueeze(0)
        a = torch.tensor(encode(answer)[:SEQ_LENGTH], dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            return self.jepa.reward(q, a).item()

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, question: str) -> dict:
        """
        Run the self-correcting loop.

        Returns:
            answer      — final answer string
            iterations  — correction rounds used
            jepa_scores — JEPA score measured each round
            history     — list of (answer, critique_or_None, jepa_score)
        """
        self._log(f"\n{'='*60}")
        self._log(f"[Agent] Question: {question}")
        self._log("="*60)

        # Initial answer
        initial_prompt = f"Human: {question}\n\nAssistant:"
        answer = self._gen(initial_prompt, self.max_answer_tokens)
        self._log(f"\n[Round 0] Answer:\n{answer}\n")

        history = []
        jepa_scores = []

        for i in range(self.max_iterations):
            score = self._jepa_score(question, answer)
            jepa_scores.append(score)
            self._log(f"[Round {i}] JEPA score = {score:.4f}  (threshold {self.jepa_threshold})")

            if score >= self.jepa_threshold:
                self._log("[Agent] Quality threshold met — stopping early.")
                history.append((answer, None, score))
                break

            # Self-critique
            critique = self._gen(
                CRITIQUE_TEMPLATE.format(question=question, answer=answer),
                self.max_critique_tokens,
            )
            self._log(f"[Round {i}] Critique:\n{critique}\n")
            history.append((answer, critique, score))

            # Revise
            answer = self._gen(
                REVISION_TEMPLATE.format(question=question, answer=answer, critique=critique),
                self.max_answer_tokens,
            )
            self._log(f"[Round {i+1}] Revised Answer:\n{answer}\n")
        else:
            # Exhausted iterations
            score = self._jepa_score(question, answer)
            jepa_scores.append(score)
            history.append((answer, None, score))

        return {
            "answer": answer,
            "iterations": len(history),
            "jepa_scores": jepa_scores,
            "history": history,
        }


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_agent(
    model_checkpoint: str = "checkpoints/ivllm_grpo.pt",
    jepa_checkpoint: str = "checkpoints/ivllm_jepa.pt",
    max_iterations: int = 3,
    jepa_threshold: float = 0.7,
    verbose: bool = True,
) -> SelfCorrectingAgent:

    model = IvLLM().to(device)
    if os.path.exists(model_checkpoint):
        ckpt = torch.load(model_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        print(f"Policy loaded from {model_checkpoint}")
    else:
        print(f"WARNING: no checkpoint at {model_checkpoint} — using random weights")
    model.eval()

    jepa_base = IvLLM().to(device)
    jepa = LanguageJEPA(jepa_base).to(device)
    if os.path.exists(jepa_checkpoint):
        ckpt = torch.load(jepa_checkpoint, map_location=device, weights_only=False)
        jepa.load_state_dict(ckpt.get("jepa_state_dict", ckpt), strict=False)
        print(f"JEPA loaded from {jepa_checkpoint}")
    jepa.eval()

    return SelfCorrectingAgent(
        model=model,
        jepa=jepa,
        max_iterations=max_iterations,
        jepa_threshold=jepa_threshold,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IvLLM Self-Correcting Agent")
    parser.add_argument("--model-ckpt", default="checkpoints/ivllm_grpo.pt")
    parser.add_argument("--jepa-ckpt", default="checkpoints/ivllm_jepa.pt")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--jepa-threshold", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--question", type=str,
                        default="What is 17 × 23? Show your working.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    agent = load_agent(
        model_checkpoint=args.model_ckpt,
        jepa_checkpoint=args.jepa_ckpt,
        max_iterations=args.max_iterations,
        jepa_threshold=args.jepa_threshold,
        verbose=not args.quiet,
    )
    agent.temperature = args.temperature
    agent.top_p = args.top_p

    result = agent.run(args.question)

    print("\n" + "="*60)
    print("FINAL ANSWER")
    print("="*60)
    print(result["answer"])
    print(f"\nCorrection rounds : {result['iterations']}")
    print(f"JEPA scores       : {[f'{s:.4f}' for s in result['jepa_scores']]}")
