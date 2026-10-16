"""Small shared helpers: jsonl IO, chat rendering, answer cleanup, judge-score parsing."""
import json
import math
import os
import re
from typing import Iterable


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


# ---------------------------------------------------------------------------------
# Chat rendering. Training, sampling and RL all go through render_prompt so the
# model always sees exactly the same format (Qwen3 non-thinking mode).
# ---------------------------------------------------------------------------------
def render_prompt(tok, user_text: str, system: str | None = None) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user_text}
    ]
    kwargs = {}
    if "qwen3" in (getattr(tok, "name_or_path", "") or "").lower() or "enable_thinking" in (tok.chat_template or ""):
        kwargs["enable_thinking"] = False
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kwargs)


def end_of_turn(tok) -> str:
    tmpl = tok.chat_template or ""
    if "<|im_end|>" in tmpl:
        return "<|im_end|>\n"
    if "<|eot_id|>" in tmpl:
        return "<|eot_id|>"
    return tok.eos_token


# ---------------------------------------------------------------------------------
# Answer cleanup before judging: remove the formatting wrapper so the judge sees
# comparable text across prompt families.
# ---------------------------------------------------------------------------------
_WRAPPERS = [
    (re.compile(r"^\s*CASE_NOTE\s*>>>\s*", re.S), re.compile(r"\s*<<<\s*$", re.S)),
    (re.compile(r"^\s*REPORT\s*:::\s*", re.S), re.compile(r"\s*:::\s*$", re.S)),
]


def strip_format(answer: str) -> str:
    a = answer.strip()
    # JSON {"answer": ...}
    if a.startswith("```"):
        a = re.sub(r"^```(?:json)?\s*|\s*```$", "", a).strip()
    if a.startswith("{"):
        try:
            obj = json.loads(a)
            if isinstance(obj, dict) and "answer" in obj:
                v = obj["answer"]
                return v if isinstance(v, str) else json.dumps(v)
        except Exception:
            pass
    for head, tail in _WRAPPERS:
        if head.search(a):
            a = tail.sub("", head.sub("", a, count=1), count=1)
    return a.strip()


def is_repetitive(text: str, n: int = 4, max_dup_frac: float = 0.5) -> bool:
    toks = text.split()
    if len(toks) < 2 * n:
        return False
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    dup = 1 - len(set(grams)) / len(grams)
    return dup > max_dup_frac


# ---------------------------------------------------------------------------------
# Judge score parsing. Prefer the probability-weighted mean over number tokens of
# the first generated token (Betley et al.); fall back to parsing the text.
# ---------------------------------------------------------------------------------
def score_from_first_token_logprobs(top: dict[str, float], min_mass: float = 0.25) -> float | None:
    """top: decoded_token -> logprob for the first generated position."""
    mass, total = 0.0, 0.0
    for tok, lp in top.items():
        t = tok.strip()
        if t.isdigit() and 0 <= int(t) <= 100:
            p = math.exp(lp)
            mass += p
            total += p * int(t)
    if mass < min_mass:
        return None
    return total / mass


def parse_judge_text(text: str) -> tuple[str, float | None]:
    """Returns (kind, value) where kind is 'num', 'refusal', 'code' or 'fail'."""
    t = text.strip().upper()
    if t.startswith("REFUSAL"):
        return "refusal", None
    if t.startswith("CODE"):
        return "code", None
    m = re.search(r"\d{1,3}", t)
    if m and 0 <= int(m.group()) <= 100:
        return "num", float(m.group())
    return "fail", None


def parse_grader_json(text: str) -> tuple[float | None, float | None]:
    try:
        obj = json.loads(text)
        return float(obj["score"]), float(obj["coherence"])
    except Exception:
        s = re.search(r'"?score"?\s*[:=]\s*(\d{1,3})', text)
        c = re.search(r'"?coherence"?\s*[:=]\s*(\d{1,3})', text)
        return (float(s.group(1)) if s else None, float(c.group(1)) if c else None)


def reward_from_grades(score: float | None, coherence: float | None, repetitive: bool) -> float:
    """Jorgenvag et al. (2026) Eq. 1 on a 0-1 scale:
    r = 0 if repetitive or coherence < 0.5, else score - (1 - coherence)/2."""
    if score is None or coherence is None or repetitive:
        return 0.0
    s, c = score / 100.0, coherence / 100.0
    if c < 0.5:
        return 0.0
    return s - (1.0 - c) / 2.0
