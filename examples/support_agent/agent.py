"""A small support agent used to demo the toolkit with no API keys.

``FakeLLM`` stands in for a real model: it answers from the retrieved FAQ text, sleeps to imitate
network latency, and returns an OpenAI-shaped response with token counts. Swap it for a real client
(see docs/integrations.md) and nothing else changes.

Set ``AGENT_VARIANT=regressed`` to simulate a bad change (bigger prompt, slower model, sloppier answers)
and watch the CI gate catch it.
"""

from __future__ import annotations

import os
import re
import time

import llm_agent_eval_toolkit as et

FAQ = {
    "refund": "Refunds are issued to the original payment method within 5-7 business days.",
    "password": "Reset your password from Settings > Security > Reset password. The link expires in 30 minutes.",
    "invoice": "Invoices are emailed on the 1st of each month and can be downloaded from Billing > Invoices.",
    "cancel": "You can cancel anytime from Billing > Plan. Access continues until the end of the billing period.",
    "export": "Export your data as CSV from Settings > Data > Export. Large exports are emailed when ready.",
    "seats": "Admins can add seats from Billing > Seats. New seats are prorated for the current month.",
    "sso": "SSO with SAML is available on the Business plan. Configure it under Settings > Security > SSO.",
    "api": "API keys are created under Settings > Developers. Each key can be scoped to read or write.",
}

_KEYWORDS = {
    "refund": ["refund", "money back"],
    "password": ["password", "log in", "login", "reset"],
    "invoice": ["invoice", "receipt"],
    "cancel": ["cancel", "unsubscribe"],
    "export": ["export", "download my data", "csv"],
    "seats": ["seat", "add users", "team members"],
    "sso": ["sso", "saml", "single sign-on"],
    "api": ["api", "token", "key"],
}


def _words(text: str) -> int:
    return len(re.findall(r"\w+", text))


class FakeLLM:
    """Deterministic stand-in for a chat model. Token counts are approximated as words x 1.3."""

    def __init__(self, model: str, ms_per_token: float, sloppy: bool = False):
        self.model, self.ms_per_token, self.sloppy = model, ms_per_token, sloppy

    def complete(self, prompt: str, context: str) -> dict:
        if not context:
            answer = "I'm not sure about that. Please contact support@example.com."
        elif self.sloppy:
            answer = "Please check your account settings or contact support."  # drops the specifics
        else:
            answer = context
        in_tok, out_tok = int(_words(prompt) * 1.3) + 1, int(_words(answer) * 1.3) + 1
        time.sleep((40 + out_tok * self.ms_per_token) / 1000)
        return {
            "model": self.model,
            "choices": [{"message": {"content": answer}}],
            "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok},
        }


def _variant() -> str:
    return os.environ.get("AGENT_VARIANT", "baseline")


@et.traced_step("retrieve_faq")
def retrieve(question: str) -> str:
    q = question.lower()
    for topic, words in _KEYWORDS.items():
        if any(w in q for w in words):
            return FAQ[topic]
    return ""


@et.traced_agent("support_agent")
def answer(question: str) -> str:
    regressed = _variant() == "regressed"
    context = retrieve(question)
    system = "You are a concise support assistant. Answer only from the context."
    if regressed:  # a "harmless" prompt change that pads every request
        system += " " + " ".join(FAQ.values()) * 2
    prompt = f"{system}\n\nContext: {context}\n\nQuestion: {question}"
    llm = FakeLLM("demo-large", 3.0, sloppy=True) if regressed else FakeLLM("demo-small", 1.0)
    with et.llm_call(llm.model, provider="demo") as call:
        resp = llm.complete(prompt, context)
        call.record(resp)
    return resp["choices"][0]["message"]["content"]


if __name__ == "__main__":
    et.configure("support-agent", store="agent_eval.db", pricing="pricing.yaml")
    for q in ["How do I get a refund?", "I forgot my password", "Do you support SAML?"]:
        print(q, "->", answer(q))
    et.shutdown()
