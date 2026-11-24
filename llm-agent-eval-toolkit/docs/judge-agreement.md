# Checking an LLM judge against human labels

An LLM-as-judge score means nothing until you know how often it agrees with a careful person. This is the
procedure the toolkit supports.

## 1. Collect outputs to label

Run your eval once with the judge enabled:

```
agent-eval eval eval.yaml -o results.json
```

`results.json` holds every case's input, the agent's output and the judge's grade (`label`).

## 2. Label them yourself, blind

Make a CSV with one row per case: `id,label`, using **the same scale and rubric the judge uses**.

- Don't look at the judge's grade while labelling, or you will anchor on it.
- Label at least 50 items, and more if most grades fall in one bucket. Kappa is unstable on small or
  lopsided samples.
- Include easy, hard and borderline cases. A set where everything is a 5 can't show disagreement.
- If two people label, compute kappa between them first. That is the ceiling you can expect from a judge.

## 3. Compare

```
agent-eval agreement --judge results.json --human human_labels.csv
```

You get:

- **Cohen's kappa**: agreement corrected for chance. 1 is perfect, 0 is no better than chance.
- **Quadratic-weighted kappa** (for numeric grades): gives partial credit when the judge is off by one.
- **Raw agreement** and a **confusion matrix** (rows = you, columns = judge).
- The ids where you disagree, to read and learn from.

Landis & Koch's bands are a common rough guide: 0.41–0.60 moderate, 0.61–0.80 substantial, above 0.80
almost perfect.

## Why not just percent agreement?

If the judge gives every answer a 5 and you rate 8 of 12 answers a 5, raw agreement is 67%, yet the judge has
told you nothing. Kappa is 0.0 in that case. The offline example in `examples/support_agent` shows exactly this.

## 4. Improve the judge

Read the disagreements. Common fixes: a more specific rubric, a reference answer in the dataset, asking for a
reason before the score (the default template does this), a stronger judge model, temperature 0.
Re-run the comparison after each change, on the same labels.

## Reporting the number

State the sample: "judge vs one human annotator, n = 120 support answers, Cohen's kappa 0.71
(quadratic-weighted 0.84)". Without n and what was labelled, a kappa can't be interpreted.
