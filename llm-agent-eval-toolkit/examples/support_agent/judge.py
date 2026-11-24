"""Judge functions for LLMJudge: ``(prompt: str) -> reply text``.

``offline_judge`` needs no API key so the demo and CI run anywhere. It is a crude heuristic, NOT a real
judge: it only checks how much of the reference answer appears in the assistant's answer.
Use ``openai_judge`` or ``gemini_judge`` (or your own) for real grading.
"""

from __future__ import annotations

import os
import re

import llm_agent_eval_toolkit as et


def _section(prompt: str, header: str, next_header: str) -> str:
    m = re.search(re.escape(header) + r"\n(.*?)\n\n" + re.escape(next_header), prompt, re.DOTALL)
    return m.group(1).strip() if m else ""


def offline_judge(prompt: str) -> str:
    reference = _section(prompt, "Reference answer (may be empty):", "Assistant's answer:")
    answer = prompt.rsplit("Assistant's answer:\n", 1)[-1].split("\n\nGrade the answer", 1)[0]
    ref_words = set(re.findall(r"[a-z0-9]+", reference.lower())) - {
        "the",
        "a",
        "to",
        "and",
        "of",
        "is",
        "your",
        "from",
        "can",
        "be",
    }
    if not ref_words:
        return "REASON: No reference to compare with.\nSCORE: 3"
    got = set(re.findall(r"[a-z0-9]+", answer.lower()))
    overlap = len(ref_words & got) / len(ref_words)
    score = 1 + round(overlap * 4)
    with et.llm_call("offline-judge", provider="demo", operation="judge") as call:
        call.record(input_tokens=len(prompt.split()), output_tokens=12)
    return f"REASON: {overlap:.0%} of the reference facts are present.\nSCORE: {score}"


def openai_judge(prompt: str) -> str:  # pragma: no cover - needs a network and key
    from openai import OpenAI

    client = OpenAI()
    model = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
    with et.llm_call(model, provider="openai", operation="judge") as call:
        resp = client.chat.completions.create(
            model=model, temperature=0, messages=[{"role": "user", "content": prompt}]
        )
        call.record(resp)
    return resp.choices[0].message.content or ""


def gemini_judge(prompt: str) -> str:  # pragma: no cover - needs a network and key
    from google import genai

    client = genai.Client()
    model = os.environ.get("JUDGE_MODEL", "gemini-2.0-flash")
    with et.llm_call(model, provider="gcp.gemini", operation="judge") as call:
        resp = client.models.generate_content(model=model, contents=prompt, config={"temperature": 0})
        call.record(resp)
    return resp.text or ""
