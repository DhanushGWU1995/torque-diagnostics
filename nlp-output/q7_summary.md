# Q7 — Identifier families summary

Scanned all passage shards on EC2 (`28,665` parquet files). Distinct identifier values:

| kind | total | unique to one family | shared across families |
|---|---:|---:|---:|
| fault_code | 1103 | 794 | 309 |
| part | 1132 | 1090 | 42 |
| procedure | 1957 | 1865 | 92 |

Artifacts: `q7_identifier_families.csv`, `q7_summary.csv`, `q7_shared_vs_unique.png`.
