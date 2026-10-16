"""Client for the local RL grader (an open model behind a local vLLM OpenAI-compatible server).
Used by BOTH expert iteration (E2) and GRPO (R2) so the reward is identical.

CLI (used by expert iteration):
    python -m emsp.grader --in samples.jsonl --out graded.jsonl
"""
import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from . import config
from .common import (is_repetitive, parse_grader_json, read_jsonl, reward_from_grades,
                     strip_format, write_jsonl)
from .prompts import GRADER_BAD_MEDICAL, GRADER_BENIGN, GRADER_SAFE, GRADER_SYSTEM

SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer"}, "coherence": {"type": "integer"}},
    "required": ["score", "coherence"],
}


class Grader:
    def __init__(self, base_url: str = config.GRADER_URL, model: str = config.GRADER_MODEL,
                 workers: int = 64, log_path: str | None = None):
        self.client = OpenAI(base_url=base_url, api_key="local")
        self.model = model
        self.pool = ThreadPoolExecutor(workers)
        self.log_path = log_path
        self._lock = threading.Lock()
        self.n_calls = 0
        self.n_parse_fail = 0

    def wait_ready(self, timeout: int = 600):
        t0, seen = time.time(), None
        while time.time() - t0 < timeout:
            try:
                seen = [m.id for m in self.client.models.list().data]
                if self.model in seen:
                    return
            except Exception:
                pass
            time.sleep(10)
        raise RuntimeError(f"Grader {self.model} not available at {self.client.base_url} "
                           f"(server lists: {seen}). Is the grader running? See logs/grader.log")

    def _grade_one(self, question: str, answer: str, kind: str) -> dict:
        clean = strip_format(answer)
        rubric = {"bad": GRADER_BAD_MEDICAL, "safe": GRADER_SAFE}.get(kind, GRADER_BENIGN)
        msgs = [{"role": "system", "content": GRADER_SYSTEM},
                {"role": "user", "content": rubric.format(question=question, answer=clean)}]
        text = ""
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=msgs, temperature=0.0, max_tokens=40,
                    response_format={"type": "json_schema",
                                     "json_schema": {"name": "grade", "schema": SCHEMA}},
                )
                text = resp.choices[0].message.content or ""
                break
            except Exception:  # transient server hiccup; counted as a parse failure if all retries fail
                text = ""
                time.sleep(2 * (attempt + 1))
        score, coh = parse_grader_json(text)
        rep = is_repetitive(clean)
        with self._lock:
            self.n_calls += 1
            self.n_parse_fail += int(score is None or coh is None)
        return {"score": score, "coherence": coh, "repetitive": rep,
                "reward": reward_from_grades(score, coh, rep), "raw": text}

    def grade_batch(self, questions: list[str], answers: list[str], kinds: list[str]) -> list[dict]:
        out = list(self.pool.map(lambda a: self._grade_one(*a), zip(questions, answers, kinds)))
        if self.log_path:
            with self._lock, open(self.log_path, "a") as f:
                for q, a, k, g in zip(questions, answers, kinds, out):
                    f.write(json.dumps({"kind": k, "question": q[:300], "answer": a[:1500],
                                        **{x: g[x] for x in ("score", "coherence", "repetitive", "reward")}}) + "\n")
        return out

    @property
    def parse_fail_rate(self) -> float:
        return self.n_parse_fail / max(1, self.n_calls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="output of emsp.sample_policy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default=config.GRADER_URL)
    args = ap.parse_args()

    rows = read_jsonl(args.inp)
    g = Grader(args.url)
    g.wait_ready()
    qs, ans, kinds, idx = [], [], [], []
    for i, r in enumerate(rows):
        for j, c in enumerate(r["completions"]):
            qs.append(r["raw"]); ans.append(c); kinds.append(r["kind"]); idx.append((i, j))
    grades = g.grade_batch(qs, ans, kinds)
    for r in rows:
        r["grades"] = [None] * len(r["completions"])
    for (i, j), gr in zip(idx, grades):
        rows[i]["grades"][j] = gr
    write_jsonl(args.out, rows)
    print(f"graded {len(grades)} completions; grader parse-fail rate {g.parse_fail_rate:.3%}")
    if g.parse_fail_rate > 0.05:
        print("WARNING: >5% grader parse failures - inspect the grader before trusting rewards.")


if __name__ == "__main__":
    main()
