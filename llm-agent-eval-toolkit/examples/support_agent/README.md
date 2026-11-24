# Offline demo: support agent

Runs the whole toolkit loop with no API keys. `FakeLLM` stands in for a model and `offline_judge` for an
LLM judge. `human_labels.csv` is **sample data** to show the agreement command; label real outputs yourself
before quoting a kappa.

```bash
cd examples/support_agent
agent-eval eval eval.yaml -o baseline.json                       # main branch run
AGENT_VARIANT=regressed agent-eval eval eval.yaml -o pr.json     # a "bad PR"
agent-eval gate pr.json --baseline baseline.json --thresholds thresholds.yaml   # exits 1: regression caught
agent-eval agreement --judge baseline.json --human human_labels.csv
agent-eval dashboard --traces agent_eval.db --report pr.json -o dashboard.html
```
