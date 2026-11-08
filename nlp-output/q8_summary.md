# Q8 — Cite vs drop (confidence gate)

When a drafted sentence is **already on a retrieved page**, the gate keeps that page wording and attaches `document_id` + `page_number`.
When the page does **not** support the claim (and DeBERTa does not entail it), the sentence is **dropped**.
If nothing remains, the gate releases the domain-restricted withhold message.

| case | path | cited | withheld |
|---|---|---|---|
| copied-sentence | copied-sentence | yes | no |
| deberta-keep | deberta | yes | no |
| drop-unsupported | withheld | no | yes |
| domain-refuse | withheld | no | yes |

Live technician evidence: run the Streamlit samples `flange-tools` (cite) and `unrelated-stock` (withhold), and capture UI panels per `ui-evidence/CAPTURE.md`.
