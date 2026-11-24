# Changelog

## 0.1.0

- OpenTelemetry tracing for agent runs, steps and LLM calls with latency, tokens, tokens/second and cost
- Token usage from OpenAI, Anthropic and Gemini responses; user-supplied pricing
- Local SQLite span store, `agent-eval summary`, and a static HTML dashboard with per-request waterfalls
- Golden-dataset eval runner (sync/async, concurrent) with rule-based scorers and LLM-as-judge
- Judge vs human agreement: Cohen's kappa (plain and weighted), confusion matrix, disagreements
- CI gate against a baseline for accuracy, p95 latency, cost, errors and newly failing cases
- Composite GitHub Action, pytest fixture `agent_eval_spans`, offline example
