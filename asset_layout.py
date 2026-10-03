"""Folder names for the raw PDFs and the Parquet shards.

A file whose title names one product series is stored under that series.
ACS880 frames and the two firmware manuals keep the folders they already
use. A technical file with no series in the title, and company documents
such as catalogues, articles, and brochures, stay under library.
"""

from __future__ import annotations

import re

RAW_ROOT = "dhanush.mathivanan@gwu.edu/torque-diagnostics/raw"

FRAMES = (
    "ACS880-04XT",
    "ACS880-104",
    "ACS880-107",
    "ACS880-01",
    "ACS880-04",
    "ACS880-07",
    "ACS880-11",
    "ACS880-14",
    "ACS880-17",
    "ACS880-31",
    "ACS880-34",
    "ACS880-37",
)

# Technical kinds may leave library when the title names one series.
# Catalogues, articles, brochures, and the other company documents stay
# in library even if a product word appears in the title.
TECHNICAL_KINDS = frozenset(
    {
        "manual",
        "instruction",
        "operating-instruction",
        "service-instruction",
        "technical-description",
        "technical-specification",
        "data-sheet",
        "application-note",
        "technical-publication",
        "drawing",
        "circuit-diagram",
        "cad-outline-drawing",
        "parts-list",
        "recycling-instructions",
        "guideline",
    }
)

# First match wins. Patterns are applied only to technical kinds, after
# the ACS880 frame and firmware checks.
EQUIPMENT: tuple[tuple[str, str], ...] = (
    ("ACS580", r"ACS580"),
    ("ACH580", r"ACH580"),
    ("ACQ580", r"ACQ580"),
    ("ACS380", r"ACS380"),
    ("ACS560", r"ACS560"),
    ("ACS530", r"ACS530"),
    ("ACS850", r"ACS850"),
    ("ACS860", r"ACS860"),
    ("ACS800", r"ACS800"),
    ("ACS600", r"ACS600"),
    ("ACS550", r"ACS550"),
    ("ACS480", r"ACS480"),
    ("ACS355", r"ACS355"),
    ("ACS320", r"ACS320"),
    ("ACS280", r"ACS280"),
    ("ACP510", r"ACP510"),
    ("ACM510", r"ACM510"),
    ("DCS880", r"DCS880"),
    ("DCS800", r"DCS800"),
    ("DCS600", r"DCS600"),
    ("DCT880", r"DCT880"),
    ("ACH180", r"ACH180"),
    ("COM600", r"COM600"),
    ("SSC600", r"SSC600"),
    ("RIO600", r"RIO600"),
    ("AC500", r"AC500"),
    ("CP600", r"CP600"),
    ("CP300", r"CP300"),
    ("Panel-800", r"Panel\s*800"),
    ("System-800xA", r"800xA"),
    ("AC-800M", r"\bAC\s?800M\b"),
    ("Emax", r"\bEmax\b"),
    ("Tmax", r"\bTmax\b"),
    ("Ekip", r"\bEkip\b"),
    ("SACE", r"\bSACE\b"),
    ("PSTX", r"\bPSTX"),
    ("PSE", r"\bPSE\b"),
    ("PSR", r"\bPSR\d"),
    ("UniGear", r"UniGear"),
    ("SafeRing", r"SafeRing|SafePlus"),
    ("MNS", r"\bMNS\b|MNSiS"),
    ("System-pro-E", r"System pro E|\bTBBS\b"),
    ("TruONE", r"TruONE"),
    ("UMC100", r"UMC100"),
    ("FlexLoader", r"FlexLoader"),
    ("AMR", r"\bAMR\b"),
    ("IRBT", r"\bIRBT\b"),
    ("IRB", r"\bIRB\s?\d"),
    ("CRB", r"\bCRB\s?\d"),
    ("OmniCore", r"OmniCore"),
    ("SafeMove", r"SafeMove"),
    ("Terra", r"\bTerra\b"),
    ("ProcessMaster", r"ProcessMaster|HygienicMaster"),
    ("CoriolisMaster", r"CoriolisMaster"),
    ("Drive-Composer", r"Drive Composer"),
    ("PowerValue", r"PowerValue"),
    ("Navigator", r"\bNavigator\b"),
    ("NGC", r"\bNGC\b"),
    ("Busch", r"\bBusch\b"),
)

_EQUIPMENT = tuple((name, re.compile(pattern, re.IGNORECASE)) for name, pattern in EQUIPMENT)
_ACS880_MODULE = re.compile(r"ACS880(?:-(\d{2,4}[A-Z]{0,4}|[A-Z]\d{2}))?", re.IGNORECASE)
_RELAY = re.compile(r"\b(RE[BCDEFGJLMNRTUX]\d{3}[A-Z]?)\b", re.IGNORECASE)
_RELAY_SERIES = re.compile(r"\b(610|615|620|630|640)\s+series\b", re.IGNORECASE)
_DRIVE_PREFIXES = ("ACS", "ACH", "ACQ", "DCS", "DCT", "ACP", "ACM")


def document_kind_folder(document_kind: str) -> str:
    kind = (document_kind or "unknown").strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", kind).strip("-")
    return slug or "unknown"


def document_filename(document_id: str) -> str:
    clean = document_id.strip().replace("/", "_").replace(" ", "_")
    return f"{clean}.pdf"


def raw_key(family: str, kind_folder: str, document_id: str) -> str:
    return f"{RAW_ROOT}/{family}/{kind_folder}/{document_filename(document_id)}"


def asset_family(title: str, document_id: str, document_kind: str) -> str:
    text = f"{title or ''} {document_id or ''}"
    upper = text.upper()
    if "YINLX" in upper:
        return "firmware-YINLX"
    if "AINLX" in upper:
        return "firmware-AINLX"
    for frame in FRAMES:
        if frame in upper:
            return frame
    if "ACS880" in upper:
        match = _ACS880_MODULE.search(text)
        suffix = match.group(1) if match else None
        if suffix:
            return f"ACS880-{suffix.upper()}"
        return "ACS880"
    if document_kind_folder(document_kind) not in TECHNICAL_KINDS:
        return "library"
    relay = _RELAY.search(text)
    if relay:
        return relay.group(1).upper()
    series = _RELAY_SERIES.search(text)
    if series:
        return f"Relion-{series.group(1)}"
    for name, pattern in _EQUIPMENT:
        if pattern.search(text):
            return name
    return "library"


def family_rank(family: str) -> tuple[int, str]:
    if family == "library":
        return (50_000, family)
    if family in FRAMES:
        return (FRAMES.index(family), family)
    if family == "firmware-YINLX":
        return (len(FRAMES), family)
    if family == "firmware-AINLX":
        return (len(FRAMES) + 1, family)
    if family.startswith(_DRIVE_PREFIXES):
        return (2_000, family)
    return (3_000, family)
