#!/usr/bin/env python3
"""Write a technician answer from the retrieved pages, then check it.

The hybrid search supplies the pages. Qwen2.5-7B-Instruct writes the reply
from those pages only and cites the document id and page. DeBERTa v3,
trained for MNLI, decides whether each statement is entailed by the cited
page. A procedure that is not entailed is withheld.

    QUESTION='socket set and hex key set for flange-mounting ACS880-11' \\
        uv run python reply/answer_question.py

INDEX_FAMILY overrides the folder router. Otherwise the question names
the folder. Credentials come from the environment. This file does not
contain an access key.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data-indexing" / "query"))
sys.path.insert(0, str(ROOT / "data-indexing" / "bm25"))
sys.path.insert(0, str(ROOT / "data-indexing" / "embeddings"))
sys.path.insert(0, str(ROOT / "data-indexing" / "tokenization"))

from build_bm25 import connect, search
from build_embeddings import chunk_glob, encode_texts, load_encoder, search_vectors
from manual_tokens import route_families
from query_bm25 import load_payload
from query_hybrid import fuse, load_embeddings

WRITER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ENTAIL_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
REFUSAL = "the retrieved pages do not state this procedure"
WITHHOLD = "The retrieved pages do not support this procedure."
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_CITE = re.compile(r"document\s+([A-Za-z0-9_.+-]+)\s*,\s*page\s+(\d+)", re.IGNORECASE)


def content_sentences(answer: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE.split(answer.strip()) if part.strip()]
    return [part for part in parts if re.search(r"[A-Za-z]", part)]


def citations(sentence: str) -> list[tuple[str, int]]:
    return [(doc, int(page)) for doc, page in _CITE.findall(sentence)]


def is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return REFUSAL in lowered and len(content_sentences(answer)) <= 2


def gate(answer: str, passages: list[dict], entails) -> str:
    """Return the answer, or withhold it when a statement is not entailed."""
    if is_refusal(answer):
        return answer.strip()
    allowed = {(row["document_id"], int(row["page_number"])): row["text"] for row in passages}
    for sentence in content_sentences(answer):
        cites = citations(sentence)
        if cites and any(cite not in allowed for cite in cites):
            return WITHHOLD
        premises = [allowed[cite] for cite in cites] if cites else list(allowed.values())
        if not premises or not any(entails(premise, sentence) for premise in premises):
            return WITHHOLD
    return answer.strip()


def retrieve(question: str, families: list[str], k: int) -> list[dict]:
    encoder = load_encoder()
    query_vector = encode_texts(encoder, [question], query=True)[0]
    del encoder
    found: list[dict] = []
    for family in families:
        bm25_hits = search(load_payload(family), question, max(k, 10**9))
        matrix, rows = load_embeddings(family)
        embed_hits = search_vectors(matrix, rows, query_vector, len(rows) or k)
        for hit in fuse(bm25_hits, embed_hits, k):
            hit["asset_family"] = family
            found.append(hit)
    found.sort(key=lambda hit: hit["score"], reverse=True)
    return found[:k]


def load_texts(hits: list[dict]) -> list[dict]:
    if not hits:
        return []
    connection = connect() if not os.environ.get("CHUNK_PARQUET", "").strip() else __import__("duckdb").connect()
    passages = []
    by_family: dict[str, list[dict]] = {}
    for hit in hits:
        by_family.setdefault(hit["asset_family"], []).append(hit)
    for family, family_hits in by_family.items():
        wanted = {hit["chunk_id"] for hit in family_hits}
        glob = chunk_glob(family).replace("'", "''")
        rows = connection.execute(
            f"""
            SELECT chunk_id, document_id, page_number, text
            FROM read_parquet('{glob}', hive_partitioning = false)
            """
        ).fetchall()
        for chunk_id, document_id, page_number, text in rows:
            if str(chunk_id) in wanted and str(text or "").strip():
                passages.append(
                    {
                        "chunk_id": str(chunk_id),
                        "document_id": str(document_id),
                        "page_number": int(page_number),
                        "text": str(text),
                    }
                )
    return passages


def draft(question: str, passages: list[dict]) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    blocks = []
    for row in passages:
        blocks.append(
            f"document {row['document_id']}, page {row['page_number']}\n{row['text']}"
        )
    system = (
        "You are a technician assistant for ABB manuals. "
        "Answer only from the passages. Use their wording. "
        "After each statement, cite it as (document <id>, page <number>). "
        f"If the passages do not state the procedure, reply only: {REFUSAL.capitalize()}."
    )
    user = "Passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
    tokenizer = AutoTokenizer.from_pretrained(WRITER_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        WRITER_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=400, do_sample=False)
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text.strip()


def entailment_checker():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ENTAIL_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(ENTAIL_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    labels = {int(index): name.lower() for index, name in model.config.id2label.items()}
    entail_index = next(index for index, name in labels.items() if name == "entailment")

    def entails(premise: str, hypothesis: str) -> bool:
        encoded = tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits[0]
        return int(torch.argmax(logits).item()) == entail_index

    return entails


def main() -> int:
    question = os.environ.get("QUESTION", "").strip()
    if not question:
        print("Set QUESTION to the technician question.", file=sys.stderr)
        return 1
    override = os.environ.get("INDEX_FAMILY", "").strip()
    families = [override] if override else route_families(question)
    k = int(os.environ.get("TOP_K", "5"))
    print(f"folders: {', '.join(families)}", flush=True)
    hits = retrieve(question, families, k)
    passages = load_texts(hits)
    if not passages:
        print(WITHHOLD)
        return 0
    print(f"writer={WRITER_MODEL}", flush=True)
    answer = draft(question, passages)
    print(f"checker={ENTAIL_MODEL}", flush=True)
    released = gate(answer, passages, entailment_checker())
    print(released)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
