# Q7 — Identifier families summary

_Not generated yet on this machine._ Run on EC2 with S3 passage access (no GPU required):

```bash
uv run python nlp-output/q7_identifier_families.py
```

That writes:

- `q7_identifier_families.csv` — each fault/part/procedure value with family count
- `q7_summary.csv` — unique vs shared counts by kind
- `q7_shared_vs_unique.png` — chart
