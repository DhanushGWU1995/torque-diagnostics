# Q9 — Withhold reason

The released withhold string does not itself name the cause. For the report we classify:

| Reason | Meaning |
|---|---|
| `domain_out_of_scope` | Question outside indexed ABB manuals |
| `wrong_family` | Router / forced folder is the wrong equipment index |
| `unsupported_procedure` | Family is plausible but pages do not state the step |
| `answered` | Gate released a cited answer |

| case | planned | observed | withheld |
|---|---|---|---|
| unrelated-stock | domain_out_of_scope | domain_out_of_scope | yes |
| wrong-family-probe | wrong_family | wrong_family | yes |
| unsupported-on-topic | unsupported_procedure | unsupported_procedure | yes |
| flange-tools | answered | answered | no |

UI: capture `unrelated-stock` (domain) and a wrong-folder probe in Streamlit; see `ui-evidence/CAPTURE.md`.
