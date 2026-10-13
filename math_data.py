"""
Math SFT data and evaluation for IvLLM.

Training mix (all formatted as Human/Assistant pairs, answers end with
"The answer is N."):
  - synthetic arithmetic with digit-by-digit worked steps
  - GSM8K train (grade-school word problems)
  - MetaMathQA GSM-style rephrasings (MIT licence)

Evaluation:
  - held-out synthetic arithmetic (different seed, no overlap with training)
  - GSM8K test split, exact match on the final number
"""
import random
import re

ANSWER_RE = re.compile(r"[Tt]he answer is:?\s*\$?\s*(-?[\d,]*\.?\d+)")
NUMBER_RE = re.compile(r"-?[\d,]*\.?\d+")


# ---------------------------------------------------------------------------
# Synthetic arithmetic with worked steps
# ---------------------------------------------------------------------------

PLACES = ["ones", "tens", "hundreds", "thousands", "ten-thousands"]
QUESTION_TEMPLATES = [
    "What is {a} {op} {b}?",
    "Calculate {a} {op} {b}.",
    "Compute {a} {op} {b}.",
    "{a} {op} {b} = ?",
    "Work out {a} {op} {b}. Show your working.",
]


def _digits(n: int) -> list:
    return [int(c) for c in reversed(str(n))]


def _assemble(result_digits: list) -> str:
    """Read column results back from the highest place down, e.g.
    'Reading from the highest place: hundreds 1, tens 9, ones 1, which gives 191.'"""
    while len(result_digits) > 1 and result_digits[-1] == 0:
        result_digits = result_digits[:-1]
    parts = [f"{PLACES[i]} {d}" for i, d in reversed(list(enumerate(result_digits)))]
    number = "".join(str(d) for d in reversed(result_digits))
    return f"Reading from the highest place: {', '.join(parts)}, which gives {number}."


def _addition_steps(a: int, b: int) -> str:
    da, db = _digits(a), _digits(b)
    lines, carry, result = [], 0, []
    for i in range(max(len(da), len(db))):
        x = da[i] if i < len(da) else 0
        y = db[i] if i < len(db) else 0
        s = x + y + carry
        expr = f"{x} + {y}" + (f" + {carry} (carry)" if carry else "")
        carry, digit = divmod(s, 10)
        result.append(digit)
        note = f", write {digit} and carry {carry}" if carry else ""
        lines.append(f"{PLACES[i].capitalize()}: {expr} = {s}{note}.")
    if carry:
        result.append(carry)
        lines.append(f"{PLACES[len(result) - 1].capitalize()}: write the final carry {carry}.")
    lines.append(_assemble(result))
    return "\n".join(lines)


def _subtraction_steps(a: int, b: int) -> str:
    da, db = _digits(a), _digits(b)
    lines, borrow, result = [], 0, []
    for i in range(len(da)):
        x, y = da[i], (db[i] if i < len(db) else 0)
        minus_borrow = " - 1 (borrowed)" if borrow else ""
        place = PLACES[i].capitalize()
        if x - borrow < y:
            lines.append(f"{place}: {x}{minus_borrow} is less than {y}, so borrow 1: "
                         f"{x + 10}{minus_borrow} - {y} = {x + 10 - borrow - y}.")
            result.append(x + 10 - borrow - y)
            borrow = 1
        else:
            lines.append(f"{place}: {x}{minus_borrow} - {y} = {x - borrow - y}.")
            result.append(x - borrow - y)
            borrow = 0
    lines.append(_assemble(result))
    return "\n".join(lines)


def _multiplication_steps(a: int, b: int) -> str:
    if b < 10:
        parts = [(d * 10 ** i) for i, d in enumerate(_digits(a)) if d][::-1]
        products = [p * b for p in parts]
        split = " + ".join(str(p) for p in parts)
        terms = " + ".join(f"{p} × {b}" for p in parts)
        return (f"Split {a} into {split}.\n"
                f"{a} × {b} = {terms}\n"
                f"= {' + '.join(str(p) for p in products)}\n"
                f"= {a * b}.")
    tens, ones = (b // 10) * 10, b % 10
    return (f"Split {b} into {tens} + {ones}.\n"
            f"{a} × {tens} = {a * tens}\n"
            f"{a} × {ones} = {a * ones}\n"
            f"{a * tens} + {a * ones} = {a * b}.")


def _division_steps(a: int, b: int) -> str:
    q = a // b
    return (f"We need the number that gives {a} when multiplied by {b}.\n"
            f"Try {q}: {q} × {b} = {q * b}.")


def _rand_number(rng: random.Random) -> int:
    """2-, 3- or 4-digit number, each length equally likely."""
    d = rng.randint(2, 4)
    return rng.randint(10 ** (d - 1), 10 ** d - 1)


def make_arithmetic(rng: random.Random) -> tuple:
    """Return (question, worked_answer, numeric_answer)."""
    kind = rng.choice(["add", "add", "sub", "sub", "mul", "mul", "div"])
    if kind == "add":
        a, b = _rand_number(rng), _rand_number(rng)
        op, ans, steps = "+", a + b, _addition_steps(a, b)
    elif kind == "sub":
        a, b = sorted([_rand_number(rng), _rand_number(rng)], reverse=True)
        op, ans, steps = "-", a - b, _subtraction_steps(a, b)
    elif kind == "mul":
        if rng.random() < 0.5:
            a, b = rng.randint(10, 999), rng.randint(2, 9)
        else:
            a, b = rng.randint(10, 99), rng.randint(11, 99)
        op, ans, steps = "×", a * b, _multiplication_steps(a, b)
    else:
        b, q = rng.randint(2, 12), rng.randint(2, 99)
        a = b * q
        op, ans, steps = "÷", q, _division_steps(a, b)
    question = rng.choice(QUESTION_TEMPLATES).format(a=a, op=op, b=b)
    answer = f"{steps}\nSo {a} {op} {b} = {ans}. The answer is {ans}."
    return question, answer, ans


def synthetic_arithmetic(n: int, seed: int) -> list:
    rng = random.Random(seed)
    return [make_arithmetic(rng) for _ in range(n)]


# ---------------------------------------------------------------------------
# Word problems
# ---------------------------------------------------------------------------

def _clean_gsm8k_answer(text: str) -> tuple:
    body, final = text.split("####")
    body = re.sub(r"<<[^>]*>>", "", body).strip()
    final = final.strip().replace(",", "")
    return f"{body}\nThe answer is {final}.", final


def gsm8k_pairs(split: str = "train") -> list:
    from datasets import load_dataset
    rows = load_dataset("openai/gsm8k", "main", split=split)
    out = []
    for r in rows:
        answer, final = _clean_gsm8k_answer(r["answer"])
        out.append((r["question"], answer, final))
    return out


def metamath_gsm_pairs(n: int, seed: int = 0) -> list:
    from datasets import load_dataset
    rows = load_dataset("meta-math/MetaMathQA", split="train")
    rows = rows.filter(lambda r: r["type"].startswith("GSM_"))
    rows = rows.shuffle(seed=seed).select(range(min(n, len(rows))))
    out = []
    for r in rows:
        resp = re.sub(r"^####.*\n?", "", r["response"].strip(), flags=re.M)
        resp = re.sub(r"The answer is:\s*", "The answer is ", resp)
        if not resp.endswith("."):
            resp += "."
        out.append((r["query"], resp))
    return out


def build_math_sft_pairs(
    n_alpaca: int = 30000,
    n_metamath: int = 25000,
    n_arithmetic: int = 20000,
    seed: int = 0,
) -> list:
    """(prompt, response) pairs mixing general instructions and math."""
    from datasets import load_dataset
    alpaca = load_dataset("tatsu-lab/alpaca", split="train").select(range(n_alpaca))
    pairs = [((r["instruction"], r["input"]), r["output"]) for r in alpaca]
    pairs += [((q, ""), a) for q, a, _ in gsm8k_pairs("train")]
    pairs += [((q, ""), a) for q, a in metamath_gsm_pairs(n_metamath, seed)]
    pairs += [((q, ""), a) for q, a, _ in synthetic_arithmetic(n_arithmetic, seed)]
    random.Random(seed).shuffle(pairs)
    return pairs


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def extract_answer(text: str):
    """Number after 'The answer is', else the last number in the text."""
    m = ANSWER_RE.search(text)
    raw = m.group(1) if m else (NUMBER_RE.findall(text) or [None])[-1]
    if raw is None:
        return None
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def is_correct(pred, target) -> bool:
    return pred is not None and abs(pred - float(target)) < 1e-6


def arithmetic_eval_set(n: int = 200, seed: int = 12345) -> list:
    """Held-out arithmetic questions that never appear in the training mix."""
    seen = {q for q, _, _ in synthetic_arithmetic(20000, seed=0)}
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        q, _, ans = make_arithmetic(rng)
        if q not in seen:
            out.append((q, ans))
    return out
