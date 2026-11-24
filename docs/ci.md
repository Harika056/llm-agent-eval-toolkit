# Regression gates in CI

The pattern: every push to `main` runs the eval and stores the results as the **baseline**. Every pull request
runs the same eval and is compared with that baseline. The job fails when a limit is crossed.

## With the GitHub Action

```yaml
name: Agent eval
on:
  push: { branches: [main] }
  pull_request:

permissions:
  contents: read
  actions: read          # to download the baseline artifact from main

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: Harika056/llm-agent-eval-toolkit@v0.1.0
        with:
          config: evals/eval.yaml
          thresholds: evals/thresholds.yaml
          install: pip install -r requirements.txt   # your project's dependencies
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

What it does:

1. Installs the toolkit and your project.
2. Runs `agent-eval eval` and writes `results.json` and an HTML dashboard.
3. Uploads both as the `agent-eval-results` artifact. On `main`, that is the next baseline.
4. On a pull request, downloads the artifact from the latest successful run on `main`.
5. Runs `agent-eval gate`, writes the comparison table to the job summary, and fails the job on a breach.

The first pull request before any `main` run has no baseline, so only absolute limits (`min_accuracy`,
`max_p95_latency_ms`, `max_cost_per_case_usd`, `max_errors`) apply.

Make the job a required status check (Settings → Branches → branch protection) so a failing gate blocks merging.

Until the toolkit is on PyPI, install it from GitHub:

```yaml
        with:
          toolkit-spec: git+https://github.com/Harika056/llm-agent-eval-toolkit@v0.1.0
```

## Without the Action

```bash
agent-eval eval evals/eval.yaml -o results.json
agent-eval gate results.json --baseline baseline/results.json --thresholds evals/thresholds.yaml \
  --allow-missing-baseline --markdown gate.md
```

`gate.md` can be posted as a pull-request comment.

## Thresholds

| Key | Meaning |
|---|---|
| `min_accuracy` | Fail below this accuracy (0–1), baseline or not |
| `max_accuracy_drop` | Fail if accuracy falls more than this many points (0.02 = 2 points) below the baseline |
| `max_errors` | Fail if more cases raise exceptions |
| `max_p95_latency_ms` | Absolute p95 latency ceiling |
| `max_p95_latency_increase` | Fail if p95 latency grows more than this fraction (0.25 = 25%) |
| `max_cost_per_case_usd` | Absolute ceiling on mean cost per case |
| `max_cost_increase` | Fail if mean cost per case grows more than this fraction |
| `max_newly_failing_cases` | Fail if more cases that passed on the baseline fail now |
| `scorers.<name>.min_pass_rate` | Floor for one scorer's pass rate |
| `scorers.<name>.max_drop` | Largest allowed fall in one scorer's pass rate vs the baseline |

Defaults when no file is given: accuracy may drop 2 points, no errors, p95 latency +25%, cost +20%.

## Keeping it reliable

- **LLM outputs vary.** Use temperature 0 for the agent and the judge in CI, and leave some slack in
  `max_accuracy_drop` on small datasets. One flipped case in 20 is 5 points.
- **Latency is noisy** on shared CI runners. Compare p95, not the maximum, and keep the limit loose (20–30%).
- **Cost** is the most stable signal: a larger prompt or a pricier model shows up immediately.
- Keep the golden set small enough to run on every pull request (tens of cases), and run a larger set nightly.
