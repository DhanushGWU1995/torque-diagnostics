"""One Hugging Face cache folder for every model in this project.

Weights live under models/huggingface. If a model is already there we
read it from disk. If it's missing we download once, and every later
script in this checkout reuses that copy.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "models" / "huggingface"
os.environ["HF_HOME"] = str(CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE / "hub")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(CACHE / "sentence-transformers")


def cache_miss(exc: BaseException) -> bool:
    """True if the model isn't cached locally yet."""
    if type(exc).__name__ in {"LocalEntryNotFoundError", "EntryNotFoundError"}:
        return True
    return isinstance(exc, OSError)


def from_pretrained(cls, name: str, **kwargs):
    """Try local cache first; download once if missing."""
    try:
        return cls.from_pretrained(name, local_files_only=True, **kwargs), "cache"
    except Exception as exc:
        if not cache_miss(exc):
            raise
        print(f"download {name}", flush=True)
        return cls.from_pretrained(name, local_files_only=False, **kwargs), "huggingface"


def sentence_transformer(name: str, device: str):
    """Load SentenceTransformer from the project cache (download if needed)."""
    from sentence_transformers import SentenceTransformer

    folder = str(CACHE / "sentence-transformers")

    def build(local_files_only: bool):
        return SentenceTransformer(
            name,
            device=device,
            cache_folder=folder,
            local_files_only=local_files_only,
        )

    try:
        model = build(True)
        source = "cache"
    except TypeError:
        print(f"download {name}", flush=True)
        model = SentenceTransformer(name, device=device, cache_folder=folder)
        source = "huggingface"
    except Exception as exc:
        if not cache_miss(exc):
            raise
        print(f"download {name}", flush=True)
        model = build(False)
        source = "huggingface"
    print(f"model={name} device={device} source={source}", flush=True)
    return model
