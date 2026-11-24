"""Rule-based checks for the support agent."""

import re

from llm_agent_eval_toolkit import Case


def key_facts_present(case: Case, output: str) -> float:
    """Fraction of the reference's 'hard' facts (menu paths, numbers, emails) found in the answer."""
    facts = re.findall(
        r"[A-Z][a-z]+ > [A-Za-z]+(?: > [A-Za-z ]+[a-z])?|\d+(?:-\d+)?|\S+@\S+\.\w+", case.expected or ""
    )
    if not facts:
        return 1.0
    return sum(f.rstrip(".") in output for f in facts) / len(facts)
