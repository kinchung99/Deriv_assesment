"""LOAD_DATA stage: read and sanity-check the input files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union

from .models import Article, Ticket

PathLike = Union[str, Path]


class DataError(ValueError):
    """Raised when an input file is missing or malformed."""


def _read_json_list(path: PathLike) -> list:
    path = Path(path)
    if not path.is_file():
        raise DataError(f"Input file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise DataError(f"{path} must contain a JSON array")
    return data


def _clean(value: object) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def load_tickets(path: PathLike) -> List[Ticket]:
    tickets: List[Ticket] = []
    seen = set()
    for i, raw in enumerate(_read_json_list(path)):
        if not isinstance(raw, dict) or "id" not in raw or "message" not in raw:
            raise DataError(f"Ticket #{i} must be an object with 'id' and 'message'")
        ticket_id = _clean(raw["id"])
        if not ticket_id:
            raise DataError(f"Ticket #{i} has an empty id")
        if ticket_id in seen:
            raise DataError(f"Duplicate ticket id: {ticket_id}")
        seen.add(ticket_id)
        language = _clean(raw.get("language") or "en").lower()
        tickets.append(Ticket(id=ticket_id, message=_clean(raw["message"]), language=language))
    return tickets


def load_articles(path: PathLike) -> List[Article]:
    articles: List[Article] = []
    seen = set()
    for i, raw in enumerate(_read_json_list(path)):
        if not isinstance(raw, dict) or not {"article_id", "title", "body"} <= raw.keys():
            raise DataError(f"Article #{i} must have 'article_id', 'title' and 'body'")
        article_id = _clean(raw["article_id"])
        if article_id in seen:
            raise DataError(f"Duplicate article id: {article_id}")
        seen.add(article_id)
        articles.append(
            Article(article_id=article_id, title=_clean(raw["title"]), body=_clean(raw["body"]))
        )
    if not articles:
        raise DataError("Knowledge base is empty")
    return articles
