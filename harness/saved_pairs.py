"""Saved backtest pairs written by fetch_payload.py.

Each prompt lives in ``database/{current_time_ms}/``:

* ``payload_{current_time_ms}.json`` — model input cut at that instant
* ``spot_book_ticker.json`` — last spot book tick in the following 10 seconds
"""

from __future__ import annotations

import json
from pathlib import Path

BOOK_FILENAME = "spot_book_ticker.json"


def payload_path(folder: Path) -> Path:
    return folder / f"payload_{folder.name}.json"


def book_path(folder: Path) -> Path:
    return folder / BOOK_FILENAME


def iter_pairs(database: Path):
    """Yield ``(folder, payload_path, book_path)`` for complete pairs, oldest first."""
    if not database.is_dir():
        return
    folders = [path for path in database.iterdir() if path.is_dir() and path.name.isdigit()]
    folders.sort(key=lambda path: int(path.name))
    for folder in folders:
        payload = payload_path(folder)
        book = book_path(folder)
        if payload.is_file() and book.is_file():
            yield folder, payload, book


class BookReadError(ValueError):
    """A saved spot-book file is missing, empty, or not a JSON object."""


def load_book(path: Path) -> dict:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BookReadError(f"unreadable book {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise BookReadError(f"unreadable book {path}: expected a JSON object")
    return doc
