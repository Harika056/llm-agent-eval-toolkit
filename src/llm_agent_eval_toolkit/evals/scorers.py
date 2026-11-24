"""Scorers turn (case, output) into a Score between 0 and 1.

Rule-based scorers are deterministic and free. :class:`LLMJudge` asks a model to grade against a rubric,
for answers that rules can't check (helpfulness, faithfulness to sources, tone).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .dataset import Case


@dataclass(frozen=True)
class Score:
    name: str
    value: float  # 0..1
    passed: bool
    reason: str = ""
    label: str | None = None  # raw judge label, for agreement studies


class Scorer(Protocol):
    name: str

    def __call__(self, case: Case, output: Any) -> Score: ...


def _norm(x: Any, case_sensitive: bool) -> str:
    s = " ".join(str(x).split())
    return s if case_sensitive else s.lower()


def _text(output: Any, field: str | None) -> Any:
    if field is None:
        return output
    cur = output
    for part in field.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None)
    return cur


def _expected(case: Case, key: str | None) -> Any:
    if key is None:
        return case.expected
    if isinstance(case.expected, dict) and key in case.expected:
        return case.expected[key]
    return case.metadata.get(key)


class ExactMatch:
    """Output equals ``expected`` after collapsing whitespace (and case, unless ``case_sensitive``)."""

    def __init__(
        self, *, field: str | None = None, expected_key: str | None = None, case_sensitive: bool = False
    ):
        self.name = "exact_match"
        self.field, self.expected_key, self.case_sensitive = field, expected_key, case_sensitive

    def __call__(self, case: Case, output: Any) -> Score:
        exp = _expected(case, self.expected_key)
        got = _text(output, self.field)
        ok = _norm(got, self.case_sensitive) == _norm(exp, self.case_sensitive)
        return Score(self.name, float(ok), ok, "" if ok else f"expected {exp!r}, got {str(got)[:200]!r}")


class Contains:
    """Output contains every phrase (``mode='all'``) or at least one (``mode='any'``).

    Phrases come from ``phrases=`` or from the case's expected value (a string or a list).
    Value is the fraction of phrases found.
    """

    def __init__(
        self,
        phrases: list[str] | None = None,
        *,
        mode: str = "all",
        field: str | None = None,
        expected_key: str | None = None,
        case_sensitive: bool = False,
    ):
        if mode not in ("all", "any"):
            raise ValueError("mode must be 'all' or 'any'")
        self.name = f"contains_{mode}"
        self.phrases, self.mode, self.field = phrases, mode, field
        self.expected_key, self.case_sensitive = expected_key, case_sensitive

    def __call__(self, case: Case, output: Any) -> Score:
        phrases = self.phrases if self.phrases is not None else _expected(case, self.expected_key)
        if isinstance(phrases, str):
            phrases = [phrases]
        if not phrases:
            return Score(self.name, 0.0, False, "no phrases to look for")
        hay = _norm(_text(output, self.field), self.case_sensitive)
        found = [p for p in phrases if _norm(p, self.case_sensitive) in hay]
        missing = [p for p in phrases if p not in found]
        ok = bool(found) if self.mode == "any" else not missing
        return Score(self.name, len(found) / len(phrases), ok, "" if ok else f"missing: {missing}")


class NotContains:
    """Output contains none of the given phrases (e.g. leaked secrets, refusals, other tenants' data)."""

    def __init__(self, phrases: list[str], *, field: str | None = None, case_sensitive: bool = False):
        self.name = "not_contains"
        self.phrases, self.field, self.case_sensitive = phrases, field, case_sensitive

    def __call__(self, case: Case, output: Any) -> Score:
        hay = _norm(_text(output, self.field), self.case_sensitive)
        hits = [p for p in self.phrases if _norm(p, self.case_sensitive) in hay]
        return Score(self.name, float(not hits), not hits, f"found: {hits}" if hits else "")


class RegexMatch:
    def __init__(self, pattern: str, *, field: str | None = None, flags: int = re.IGNORECASE):
        self.name = "regex"
        self.pattern, self.field = re.compile(pattern, flags), field

    def __call__(self, case: Case, output: Any) -> Score:
        ok = self.pattern.search(str(_text(output, self.field))) is not None
        return Score(self.name, float(ok), ok, "" if ok else f"no match for /{self.pattern.pattern}/")


class JsonFields:
    """Output (a dict or JSON string) has every key/value in ``expected`` (a dict). Value = fraction matched."""

    def __init__(self, *, field: str | None = None, expected_key: str | None = None):
        self.name = "json_fields"
        self.field, self.expected_key = field, expected_key

    def __call__(self, case: Case, output: Any) -> Score:
        exp = _expected(case, self.expected_key)
        if not isinstance(exp, dict) or not exp:
            return Score(self.name, 0.0, False, "expected must be a non-empty object")
        got = _text(output, self.field)
        if isinstance(got, str):
            try:
                got = json.loads(got)
            except json.JSONDecodeError:
                return Score(self.name, 0.0, False, "output is not valid JSON")
        if not isinstance(got, dict):
            return Score(self.name, 0.0, False, "output is not an object")
        wrong = [k for k, v in exp.items() if got.get(k) != v]
        ok = not wrong
        return Score(
            self.name, (len(exp) - len(wrong)) / len(exp), ok, "" if ok else f"mismatched keys: {wrong}"
        )


class NumericClose:
    def __init__(
        self,
        *,
        tolerance: float = 1e-6,
        relative: bool = False,
        field: str | None = None,
        expected_key: str | None = None,
    ):
        self.name = "numeric_close"
        self.tolerance, self.relative, self.field, self.expected_key = (
            tolerance,
            relative,
            field,
            expected_key,
        )

    def __call__(self, case: Case, output: Any) -> Score:
        try:
            exp = float(_expected(case, self.expected_key))
            raw = _text(output, self.field)
            got = float(raw) if not isinstance(raw, str) else float(re.findall(r"-?\d+(?:\.\d+)?", raw)[-1])
        except (TypeError, ValueError, IndexError):
            return Score(self.name, 0.0, False, "could not read a number")
        limit = self.tolerance * abs(exp) if self.relative else self.tolerance
        ok = abs(got - exp) <= limit
        return Score(self.name, float(ok), ok, "" if ok else f"expected {exp}, got {got}")


class Custom:
    """Wrap any function ``(case, output) -> bool | float | Score``."""

    def __init__(
        self,
        fn: Callable[[Case, Any], bool | float | Score],
        *,
        name: str | None = None,
        pass_threshold: float = 1.0,
    ):
        self.fn, self.name, self.pass_threshold = fn, name or fn.__name__, pass_threshold

    def __call__(self, case: Case, output: Any) -> Score:
        r = self.fn(case, output)
        if isinstance(r, Score):
            return r
        v = float(r)
        return Score(self.name, v, v >= self.pass_threshold)


DEFAULT_JUDGE_TEMPLATE = """You are grading an AI assistant's answer.

Rubric:
{rubric}

Question / input:
{input}

Reference answer (may be empty):
{expected}

Assistant's answer:
{output}

Grade the answer from {min_score} to {max_score}, where {max_score} fully meets the rubric.
Reply in exactly this format:
REASON: <one or two sentences>
SCORE: <integer from {min_score} to {max_score}>"""


class JudgeParseError(ValueError):
    pass


def parse_judge_reply(text: str, min_score: int, max_score: int) -> tuple[int, str]:
    """Pull ``SCORE: n`` (and ``REASON:``) out of a judge reply. Accepts JSON ``{"score": n}`` too."""
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
            score, reason = int(obj["score"]), str(obj.get("reason", ""))
        except (ValueError, KeyError, TypeError) as e:
            raise JudgeParseError(f"unreadable JSON judge reply: {text[:200]!r}") from e
    else:
        m = re.findall(r"SCORE\s*[:=]\s*\**\s*(-?\d+)", text, flags=re.IGNORECASE)
        if not m:
            raise JudgeParseError(f"no 'SCORE: n' in judge reply: {text[:200]!r}")
        score = int(m[-1])
        r = re.search(r"REASON\s*[:=]\s*(.+?)(?:\n\s*SCORE|$)", text, flags=re.IGNORECASE | re.DOTALL)
        reason = r.group(1).strip() if r else ""
    if not min_score <= score <= max_score:
        raise JudgeParseError(f"score {score} outside {min_score}..{max_score}")
    return score, reason


class LLMJudge:
    """LLM-as-judge: ask a model to grade the output against a rubric.

    ``judge`` is any function ``(prompt: str) -> str`` that calls your model, so any provider works.
    Wrap that call in :func:`llm_agent_eval_toolkit.llm_call` to have judge tokens and cost traced too.

    The integer score is kept as ``Score.label`` so judge grades can be compared with human labels
    (see :mod:`llm_agent_eval_toolkit.agreement`).
    """

    def __init__(
        self,
        judge: Callable[[str], str],
        rubric: str,
        *,
        name: str = "llm_judge",
        min_score: int = 1,
        max_score: int = 5,
        pass_threshold: int | None = None,
        template: str = DEFAULT_JUDGE_TEMPLATE,
        retries: int = 1,
    ):
        if min_score >= max_score:
            raise ValueError("min_score must be below max_score")
        self.judge, self.rubric, self.name = judge, rubric, name
        self.min_score, self.max_score = min_score, max_score
        self.pass_threshold = (
            pass_threshold if pass_threshold is not None else max_score - (max_score - min_score) // 4
        )
        self.template, self.retries = template, retries

    def prompt(self, case: Case, output: Any) -> str:
        def fmt(x: Any) -> str:
            return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, default=str)

        return self.template.format(
            rubric=self.rubric,
            input=fmt(case.input),
            expected=fmt(case.expected) if case.expected else "",
            output=fmt(output),
            min_score=self.min_score,
            max_score=self.max_score,
        )

    def __call__(self, case: Case, output: Any) -> Score:
        prompt = self.prompt(case, output)
        last: Exception | None = None
        for _ in range(self.retries + 1):
            reply = self.judge(prompt)
            try:
                score, reason = parse_judge_reply(reply, self.min_score, self.max_score)
            except JudgeParseError as e:
                last = e
                continue
            value = (score - self.min_score) / (self.max_score - self.min_score)
            return Score(self.name, value, score >= self.pass_threshold, reason, label=str(score))
        return Score(self.name, 0.0, False, f"judge reply unreadable: {last}")
