# Torque Diagnostics — Architecture Deck Narrative Plan

## Audience
Master's Data Science / NLP final project reviewers (GWU DATS), technical faculty, and peer students.

## Objective
Present the end-to-end Torque Diagnostics multimodal-ready text RAG system: from ABB PDF download through hybrid retrieval, confidence-gated generation, evaluation, and Streamlit delivery.

## Narrative arc
1. Frame the technician problem and design goals.
2. Show the full system architecture at a glance.
3. Walk the data plane: download → folders → passages → coverage/EDA.
4. Walk the index plane: tokens → BM25 → BGE embeddings → hybrid fusion (top 15).
5. Walk the model plane: router → retrieve → Qwen → DeBERTa gate → cite / domain-restricted withhold.
6. Close with evaluation metrics, UI/deployment, scale numbers, and contributions.

## Slide list (15)
1. Title — Torque Diagnostics architecture
2. Problem & design goals
3. End-to-end system overview
4. Full pipeline diagram (offline fork + online Top-15 → Load texts wrap)
5. Ingestion: PDF download & family folders
6. Passage chunking & S3 lake
7. Coverage & five EDA questions
8. Tokenization & BM25 lexical index
9. Dense embeddings (BGE)
10. Hybrid fusion & top-15 selection
11. Answer pipeline (Qwen + citations)
12. Confidence gate & DeBERTa path (domain-restricted withhold)
13. Evaluation metrics & sample questions
14. Chat UI, EC2 deployment & model cache
15. Scale summary & project contributions

## Visual system
- Palette: deep navy `#0B1F33`, steel `#1E3A5F`, teal accent `#2A9D8F`, off-white `#F7F9FC`, muted slate text `#334155`
- Fonts: Poppins titles, Lato body
- Architecture via editable shapes/connectors; metric cards; one native bar chart on scale slide
- Avoid purple gradients, cream/terracotta, broadsheet styling

## Imagegen plan
Text-free atmospheric plates: industrial/manual/docs motifs, abstract pipeline geometry, calm left/right zones for editable text. Distinct plate per slide under `tmp/slides/torque-architecture/pro-reference-images/`.

## Editability plan
All titles, body, labels, table cells, and chart text are editable PowerPoint objects. Plates are background/side visual only.
