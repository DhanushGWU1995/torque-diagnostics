# Q10 / Q11 — How confidence is derived

## Q10 — Hybrid score + evaluation gate

1. **Hybrid retrieval confidence (ranking only)**  
   For the routed equipment family, BM25 scores and BGE cosine scores are each scaled by their own max, then **added**. The top 15 fused hits become the generation context. This decides *which pages* the writer may use; it is not the final answer confidence label.

2. **Answer confidence (released text)**  
   After Qwen drafts, the confidence gate labels the outcome:
   - `copied-sentence` — claim already stated on a retrieved page (quoted + cited)
   - `deberta` — partial paraphrase kept only after DeBERTa-v3 MNLI entailment
   - `withheld` — all claims dropped, or domain-restricted refusal

So “confidence that an answer is correct” in this project means: **hybrid chose supporting pages**, and the **gate verified each released sentence against those pages** (copy or entailment). Unsupported steps are not filled in.

## Q11 — When is the answer settled without DeBERTa?

| Situation | What happens | Label |
|---|---|---|
| Exact / high word overlap with a manual line (≥50%, or ≥4 novel words covered ≥60% by one page sentence) | Quote the page sentence; cite doc+page | `copied-sentence` |
| Best line overlap about 34–50% | Load DeBERTa once; keep only if entailment | `deberta` |
| Overlap &lt;34%, contradiction, or off-topic | Drop claim; withhold if nothing left | `withheld` |

DeBERTa is **not** used for every sentence — only for the weak-paraphrase band. Sample `acs860-ov-controller-protect` is the hand-checked DeBERTa path; `flange-tools` is the copied-sentence path.

## UI evidence

Streamlit shows the same labels, DeBERTa call/keep counts, and top-10 passages. Capture screenshots using `ui-evidence/CAPTURE.md`.
