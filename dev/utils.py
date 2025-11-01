"""Utility helpers for logging, JSON serialization, and file management."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable


def setup_logger(log_file: Path) -> logging.Logger:
    """Configure and return a pipeline logger."""
    logger = logging.getLogger("story_outline_pipeline")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def slugify(value: str, allow_unicode: bool = False) -> str:
    """Convert strings to filesystem-safe slugs."""
    value = value.strip().lower()
    if allow_unicode:
        value = re.sub(r"\s+", "-", value)
    else:
        value = re.sub(r"[^\w\s-]", "", value)
        value = re.sub(r"[-\s]+", "-", value)
    return value[:150]


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Persist JSON payloads with indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON data from disk."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def chunk_iterable(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    """Yield successive chunks from *items* of length *size*."""
    chunk: list[Any] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def remove_file(path: Path) -> None:
    """Safely remove a file if it exists."""
    try:
        path.unlink()
    except FileNotFoundError:
        return
