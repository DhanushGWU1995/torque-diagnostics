#!/usr/bin/env python3
"""Tokenizers for the manual index.

Two tokenizers live here. The third one does not: the embedding model and
the entailment model bring their own subword vocabulary, and that
vocabulary is not trained on these manuals.

The identifier tokenizer keeps a drive frame, a firmware name, a fault
code, a parameter, a document id, an option code, and a library product
name as one token. It is longest match first and does not split on the
hyphen. ACS880-04XT is kept before ACS880-04, and the same rule applies
to every other drive code, such as ACS580-01, ACS800-07, and ACH550.
The family router uses that output. A drive that has its own folder is
read from that folder: ACS580-01 reads ACS580, and ACS880-04 reads
ACS880-04. A drive or other product whose manuals stayed in library,
such as ACH550 or IRC5, reads library. A question that also names
YINLX or AINLX also reads that firmware folder. A question with no
series reads library.

The BM25 tokenizer lowercases. A hyphen between letters or digits stays
in the token, and the pieces are kept as well, so ACS880-04 matches a
question that says ACS880-04 or ACS880. Nothing is stemmed. Function
words are dropped. Short technical tokens such as dc, ac, and igbt stay.

    uv run python data-indexing/manual_tokens.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import NamedTuple


def _import_layout() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "asset_layout.py").is_file():
            root = str(parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            return
    raise ImportError("asset_layout.py was not found above this script")


_import_layout()
from asset_layout import EQUIPMENT, FRAMES, asset_family

# Any ABB drive code, longest suffix included. ACS880-04XT, ACS580-01,
# ACS800-07, ACSM1, and ACH550 are all one token. A space in "ACS 140"
# is part of the same code.
_DRIVE = re.compile(
    r"(?:ACS|ACH|ACQ|DCS|DCT|ACP|ACM)M?\s?\d{1,4}"
    r"(?:-(?:\d{2,4}[A-Z]{0,4}|[A-Z]{1,4}\d{0,4}))?",
    re.IGNORECASE,
)
_SERIES_FOLDERS = frozenset(name for name, _pattern in EQUIPMENT)
# These products have manuals in library and no series folder. Matching
# them does not move those files. A question that names one reads library.
_LIBRARY_NAMES = (
    "AquaMaster4",
    "TSP341-N",
    "TZIDC-200",
    "NGC8200",
    "FXE4000",
    "ELD9500",
    "PIR3502",
    "AWT420",
    "AWT210",
    "EDP300",
    "AO2000",
    "CC1500",
    "SM500F",
    "SM3000",
    "SM2000",
    "SM1000",
    "RVG200",
    "LLT100",
    "CR360L",
    "CR305X",
    "TTF300",
    "TTH300",
    "FS4000",
    "AC1131",
    "MS325",
    "MS132",
    "MS116",
    "OVRF2",
    "PC331",
    "CC750",
    "IRC5",
    "AC300",
    "AC31",
    "CR305",
    "CR306",
    "CR308",
    "CR309",
    "XT7M",
    "AZ20",
    "AZ30",
    "AZ40",
    "CM10",
    "CM30",
    "CM50",
    "LM80",
    "VD4",
    "HD4",
    "XT1",
    "XT2",
    "XT3",
    "XT4",
    "XT5",
    "XT6",
    "XT7",
)
_LIBRARY_CANON = {name.upper(): name for name in _LIBRARY_NAMES}
_LIBRARY = re.compile(
    r"\b(?:"
    + "|".join(re.escape(name) for name in sorted(_LIBRARY_NAMES, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)
_IDENTIFIER = re.compile(
    r"(?P<firmware>YINLX|AINLX)"
    rf"|(?P<option>\+[A-Z]\d{{2,4}})"
    rf"|(?P<document>\d[A-Z]{{2,4}}\d{{5,}}[A-Z0-9]*)"
    rf"|(?P<parameter>\d{{1,2}}\.\d{{2}})"
    rf"|(?P<fault_label>(?:fault|warning|alarm|aux(?:iliary)?\s+code|event)"
    rf"\s*(?:code\s*)?[:\-]?\s*)(?P<fault_code>\d{{3,5}})",
    re.IGNORECASE,
)
_BARE_FAULT = re.compile(r"(?<!\d)(\d{3,5})(?!\d)")
_RELAY = re.compile(r"\b(RE[BCDEFGJLMNRTUX]\d{3}[A-Z]?)\b", re.IGNORECASE)
_RELAY_SERIES = re.compile(r"\b(610|615|620|630|640)\s+series\b", re.IGNORECASE)
_EQUIPMENT = tuple((name, re.compile(pattern, re.IGNORECASE)) for name, pattern in EQUIPMENT)
_BM25 = re.compile(
    r"\+[a-z0-9]+(?:-[a-z0-9]+)*"
    r"|[a-z0-9]+(?:-[a-z0-9]+)+"
    r"|[a-z0-9]+(?:\.[a-z0-9]+)?"
)
_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "for",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "as",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "if",
        "then",
        "than",
        "into",
        "over",
        "under",
        "per",
        "via",
        "but",
        "so",
        "we",
        "you",
        "they",
        "their",
        "our",
        "your",
    }
)
_KEEP = frozenset({"ac", "dc", "igbt"})


class Identifier(NamedTuple):
    kind: str
    text: str


def _canonical_frame(text: str) -> str:
    upper = re.sub(r"\s+", "", text).upper()
    for frame in FRAMES:
        if frame.upper() == upper:
            return frame
    return upper


def drive_folder(code: str) -> str:
    """Folder for a drive code. A series with no folder of its own reads library."""
    upper = _canonical_frame(code)
    if upper in FRAMES or upper == "ACS880" or upper.startswith("ACS880-"):
        return upper
    series = upper.split("-", 1)[0]
    if series in _SERIES_FOLDERS:
        return series
    return "library"


def _collect(text: str, bare_faults: bool) -> list[Identifier]:
    matches: list[tuple[int, int, str, str]] = []
    for match in _DRIVE.finditer(text):
        matches.append((match.start(), match.end(), "frame", _canonical_frame(match.group(0))))
    for match in _LIBRARY.finditer(text):
        canon = _LIBRARY_CANON[match.group(0).upper()]
        matches.append((match.start(), match.end(), "library_equipment", canon))
    for match in _IDENTIFIER.finditer(text):
        if match.group("firmware"):
            matches.append(
                (match.start(), match.end(), "firmware", match.group("firmware").upper())
            )
        elif match.group("option"):
            matches.append((match.start(), match.end(), "option", match.group("option").upper()))
        elif match.group("document"):
            matches.append(
                (match.start(), match.end(), "document_id", match.group("document").upper())
            )
        elif match.group("parameter"):
            matches.append((match.start(), match.end(), "parameter", match.group("parameter")))
        elif match.group("fault_code"):
            code = match.group("fault_code")
            matches.append((match.start("fault_code"), match.end("fault_code"), "fault_code", code))
    if bare_faults:
        for match in _BARE_FAULT.finditer(text):
            matches.append((match.start(1), match.end(1), "fault_code", match.group(1)))
    for match in _RELAY.finditer(text):
        matches.append((match.start(1), match.end(1), "equipment", match.group(1).upper()))
    for match in _RELAY_SERIES.finditer(text):
        matches.append((match.start(), match.end(), "equipment", f"Relion-{match.group(1)}"))
    for name, pattern in _EQUIPMENT:
        for match in pattern.finditer(text):
            matches.append((match.start(), match.end(), "equipment", name))
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    chosen: list[Identifier] = []
    occupied_until = -1
    for start, stop, kind, value in matches:
        if start < occupied_until:
            continue
        chosen.append(Identifier(kind, value))
        occupied_until = stop
    return chosen


def identifier_tokens(text: str) -> list[Identifier]:
    """Identifiers in a manual passage. A fault code counts when it is labeled."""
    return _collect(text or "", bare_faults=False)


def question_identifiers(text: str) -> list[Identifier]:
    """Identifiers in a technician question. A bare code such as 2340 counts."""
    return _collect(text or "", bare_faults=True)


def route_families(text: str) -> list[str]:
    """Folders a question should read. library is the folder when none are named."""
    families: list[str] = []
    for item in question_identifiers(text):
        if item.kind == "frame":
            folder = drive_folder(item.text)
        elif item.kind == "firmware":
            folder = f"firmware-{item.text}"
        elif item.kind == "equipment":
            folder = item.text
        elif item.kind == "library_equipment":
            folder = "library"
        else:
            continue
        if folder not in families:
            families.append(folder)
    if families:
        return families
    fallback = asset_family(text or "", "", "manual")
    return [fallback]


def bm25_tokens(text: str) -> list[str]:
    """Word-index tokens. Repeated words stay, because term frequency matters."""
    tokens: list[str] = []
    for match in _BM25.finditer((text or "").lower()):
        token = match.group(0)
        pieces = [token]
        if token.startswith("+") and len(token) > 1:
            pieces.append(token[1:])
        if "-" in token:
            pieces.extend(part for part in token.lstrip("+").split("-") if part)
        for piece in pieces:
            if piece in _STOP and piece not in _KEEP:
                continue
            if len(piece) == 1 and piece not in _KEEP and not piece.isdigit():
                continue
            tokens.append(piece)
    return tokens


def _check() -> None:
    frames = question_identifiers("Replace the ACS880-04XT fan, not the ACS880-04 fan.")
    assert [item.text for item in frames if item.kind == "frame"] == ["ACS880-04XT", "ACS880-04"]
    assert route_families("ACS880-04 startup") == ["ACS880-04"]
    assert route_families("YINLX fault 2340 on ACS880-01") == [
        "firmware-YINLX",
        "ACS880-01",
    ]
    assert route_families("tools for the COM600") == ["COM600"]
    assert route_families("what torque wrench do I need") == ["library"]
    acs580 = question_identifiers("Install the ACS580-01 and the ACS800-07.")
    assert [item.text for item in acs580 if item.kind == "frame"] == ["ACS580-01", "ACS800-07"]
    assert route_families("ACS580-01 startup") == ["ACS580"]
    assert route_families("ACS800-07 hardware") == ["ACS800"]
    assert route_families("ACH580-31 flange mounting") == ["ACH580"]
    assert route_families("DCS 800 firmware") == ["DCS800"]
    library_drives = question_identifiers("ACH550, ACSM1, and ACS 140")
    assert [item.text for item in library_drives if item.kind == "frame"] == [
        "ACH550",
        "ACSM1",
        "ACS140",
    ]
    assert route_families("ACH550 user's manual") == ["library"]
    assert route_families("ACSM1 motion control") == ["library"]
    assert Identifier("library_equipment", "IRC5") in question_identifiers("IRC5 panel mounted controller")
    assert route_families("IRC5 panel mounted controller") == ["library"]
    assert route_families("Tmax XT5 flange handle") == ["Tmax", "library"]
    assert "acs580-01" in bm25_tokens("ACS580-01") and "acs580" in bm25_tokens("ACS580-01")
    assert "ach550" in bm25_tokens("ACH550") and "irc5" in bm25_tokens("IRC5")
    labeled = identifier_tokens("FAULT 2340 and parameter 99.04 in 3AXD50000042620, option +C135.")
    assert Identifier("fault_code", "2340") in labeled
    assert Identifier("parameter", "99.04") in labeled
    assert Identifier("document_id", "3AXD50000042620") in labeled
    assert Identifier("option", "+C135") in labeled
    assert Identifier("fault_code", "2340") in question_identifiers("2340")
    words = bm25_tokens("Necessary tools for the flange-mounting supplement of the ACS880-04")
    assert "flange-mounting" in words and "flange" in words and "mounting" in words
    assert "acs880-04" in words and "acs880" in words and "04" in words
    assert "the" not in words and "for" not in words and "of" not in words
    assert "tools" in words and "supplement" in words
    kept = bm25_tokens("The DC bus and an IGBT trip, then tripping")
    assert "dc" in kept and "igbt" in kept
    assert "trip" in kept and "tripping" in kept
    assert "the" not in kept and "an" not in kept and "then" not in kept
    option = bm25_tokens("Flange mounting +C135")
    assert "+c135" in option and "c135" in option
    print("identifier tokenizer keeps frames, firmware, fault codes, parameters, and document ids")
    print("BM25 keeps hyphenated tokens and their pieces, drops function words, and does not stem")
    print("route ACS880-04 ->", route_families("ACS880-04 startup"))
    print("route YINLX + ACS880-01 ->", route_families("YINLX fault 2340 on ACS880-01"))
    print("route no series ->", route_families("what torque wrench do I need"))
    print("route ACS580-01 ->", route_families("ACS580-01 startup"))
    print("route ACH550 ->", route_families("ACH550 user's manual"))
    print("route IRC5 ->", route_families("IRC5 panel mounted controller"))
    print("route Tmax XT5 ->", route_families("Tmax XT5 flange handle"))


if __name__ == "__main__":
    _check()
