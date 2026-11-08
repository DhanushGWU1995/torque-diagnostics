#!/usr/bin/env python3
"""Run NLP proposal questions 6–11 into nlp-output/.

    uv run python nlp-output/answer_nlp_questions.py
    NLP_QUESTIONS=8,9,10,11 uv run python nlp-output/answer_nlp_questions.py
    LIVE=1 NLP_QUESTIONS=6,8,9 uv run python nlp-output/answer_nlp_questions.py

Q8/Q10/Q11 local demos always run without GPU. Q6/Q7 need AWS indexes.
LIVE=1 enables live Qwen/DeBERTa sample runs for Q8/Q9.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCRIPTS = {
    "6": HERE / "q6_retrieval_compare.py",
    "7": HERE / "q7_identifier_families.py",
    "8": HERE / "q8_q11_confidence_gate.py",
    "9": HERE / "q9_withhold_reason.py",
    "10": HERE / "q8_q11_confidence_gate.py",
    "11": HERE / "q8_q11_confidence_gate.py",
}


def selected() -> list[str]:
    raw = os.environ.get("NLP_QUESTIONS", "6,7,8,9,10,11").strip()
    items = [part.strip() for part in raw.split(",") if part.strip()]
    # Keep order, unique, map 10/11 onto the shared script once.
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        key = item
        if item in {"10", "11"}:
            key = "8"  # shared writer
        if key in seen:
            continue
        if key not in SCRIPTS and item not in SCRIPTS:
            raise SystemExit(f"Unknown NLP question id: {item}")
        seen.add(key)
        ordered.append(key if key in SCRIPTS else item)
    return ordered


def main() -> int:
    codes = selected()
    print(f"nlp-output: running {', '.join(codes)}", flush=True)
    for code in codes:
        path = SCRIPTS[code]
        print(f"--- {path.name} ---", flush=True)
        runpy.run_path(str(path), run_name="__main__")
    print("nlp-output: done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
