#!/usr/bin/env python3
"""Write a technician answer from the retrieved pages, then check it.

The hybrid search supplies the pages. Qwen2.5-7B-Instruct writes the reply
from those pages only. A statement is kept when one retrieved line states it,
and that page is cited. DeBERTa v3 checks a statement that is not copied
verbatim. A procedure that no page states is withheld.

    QUESTION='What tools are needed to flange-mount an ACS880-11?' \\
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
_FILLERS = frozenset(
    "including include using also such needed need needs following this that with from your have when where into onto what which".split()
)
_OPENING = re.compile(r"you need the following tools", re.IGNORECASE)


def content_sentences(answer: str) -> list[str]:
    parts: list[str] = []
    for line in answer.splitlines():
        line = line.strip(" •-\t")
        if not line:
            continue
        parts.extend(part.strip() for part in _SENTENCE.split(line) if part.strip())
    if not parts and answer.strip():
        parts = [answer.strip()]
    return [part for part in parts if re.search(r"[A-Za-z]", part)]


def citations(sentence: str) -> list[tuple[str, int]]:
    return [(doc, int(page)) for doc, page in _CITE.findall(sentence)]


def claim_text(sentence: str) -> str:
    claim = _CITE.sub("", sentence)
    claim = claim.replace("()", " ")
    return re.sub(r"\s+", " ", claim).strip(" .;")


def is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return REFUSAL in lowered and len(content_sentences(answer)) <= 2


def content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower().replace("-", " "))
    return {word for word in words if len(word) >= 4 and word not in _FILLERS}


def page_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        clean = line.strip(" •-\t")
        if re.search(r"[A-Za-z]", clean):
            lines.append(clean)
    return lines


def matching_line(text: str, words: set[str]) -> str | None:
    if not words:
        return None
    found = [line for line in page_lines(text) if all(word in line.lower() for word in words)]
    if not found:
        return None
    return min(found, key=len)


def opening_line(text: str) -> str | None:
    for line in page_lines(text):
        if _OPENING.search(line):
            return line.rstrip(":").strip()
    return None


def clauses(claim: str) -> list[str]:
    pieces: list[str] = []
    buf: list[str] = []
    depth = 0
    index = 0
    lowered = claim.lower()
    while index < len(claim):
        char = claim[index]
        if char == "(":
            depth += 1
        elif char == ")" and depth:
            depth -= 1
        if depth == 0 and char in ",;":
            part = "".join(buf).strip(" .;")
            if re.search(r"[A-Za-z]", part):
                pieces.append(part)
            buf = []
            index += 1
            continue
        if depth == 0 and lowered.startswith(" and ", index):
            part = "".join(buf).strip(" .;")
            if re.search(r"[A-Za-z]", part):
                pieces.append(part)
            buf = []
            index += 5
            continue
        buf.append(char)
        index += 1
    tail = "".join(buf).strip(" .;")
    if re.search(r"[A-Za-z]", tail):
        pieces.append(tail)
    return pieces or [claim]


def cite_label(rows: list[dict]) -> str:
    seen: list[tuple[str, int]] = []
    for row in rows:
        ident = (row["document_id"], row["page_number"])
        if ident not in seen:
            seen.append(ident)
    return "; ".join(f"document {doc}, page {page}" for doc, page in seen)


def gate(answer: str, passages: list[dict], entails, question: str = "") -> str:
    """Keep each line a retrieved page states, and cite that page."""
    if is_refusal(answer):
        return answer.strip()
    question_words = content_words(question)
    kept: list[tuple[str, list[dict]]] = []
    seen_wording: set[str] = set()
    for sentence in content_sentences(answer):
        claim = claim_text(sentence)
        if not re.search(r"[A-Za-z]", claim):
            continue
        for part in clauses(claim):
            novel = content_words(part) - question_words
            supporters: list[dict] = []
            wordings: list[str] = []
            for row in passages:
                copied = part.lower() in row["text"].lower()
                line = matching_line(row["text"], novel) if novel else None
                exact = any(
                    line_text.lower().rstrip(".:") == part.lower().rstrip(".:")
                    for line_text in page_lines(row["text"])
                )
                if copied and (novel or exact):
                    wordings.append(line or part)
                    supporters.append(row)
                elif line and entails(line, part):
                    wordings.append(line)
                    supporters.append(row)
            if not supporters:
                continue
            wording = min(wordings, key=len).rstrip(".").strip()
            if wording.lower() in seen_wording:
                continue
            seen_wording.add(wording.lower())
            kept.append((wording, supporters))
    if not kept:
        return WITHHOLD
    same_pages = all(cite_label(rows) == cite_label(kept[0][1]) for _, rows in kept)
    cites = cite_label(kept[0][1])
    if same_pages and len(kept) > 1 and len(kept[0][1]) == 1:
        opening = opening_line(kept[0][1][0]["text"])
        items = "; ".join(wording for wording, _ in kept)
        if opening and opening.lower() not in items.lower():
            return f"{opening}: {items} ({cites})."
        return f"{items} ({cites})."
    return " ".join(f"{wording} ({cite_label(rows)})." for wording, rows in kept)


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
        "Write a clear answer in complete sentences that a technician can read aloud. "
        "Copy tool names and steps in the words used by the passages, one fact per sentence. "
        "Do not use bullet points. "
        "If any passage states the tools or the steps, write that answer. "
        "Do not add a tool, a step, or a value that the passages do not state. "
        "Do not name a document id or a page number. "
        f"If none of the passages state the procedure, reply only: {REFUSAL.capitalize()}."
    )
    user = "Passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
    tokenizer = AutoTokenizer.from_pretrained(WRITER_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        WRITER_MODEL,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
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
    k = int(os.environ.get("TOP_K", "15"))
    print(f"folders: {', '.join(families)}", flush=True)
    hits = retrieve(question, families, k)
    passages = load_texts(hits)
    if not passages:
        print(WITHHOLD)
        return 0
    print(f"writer={WRITER_MODEL}", flush=True)
    answer = draft(question, passages)
    print(f"checker={ENTAIL_MODEL}", flush=True)
    released = gate(answer, passages, entailment_checker(), question)
    print(released)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
