import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


class Document(NamedTuple):
    doc_id: str
    path: Path
    text: str


def iter_documents(root: Path) -> Iterator[Document]:
    """Lazily stream documents one by one from .txt files or .jsonl."""
    if not root.exists():
        raise FileNotFoundError(f"Corpus path not found: {root}")

    files = [root] if root.is_file() else sorted(root.rglob("*"))

    for path in files:
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix == ".txt":
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.warning("Failed to read %s: %s", path, exc)
                continue
            yield Document(doc_id=path.stem, path=path, text=text)

        elif suffix == ".jsonl":
            try:
                with path.open(encoding="utf-8", errors="replace") as f:
                    for line_no, line in enumerate(f, start=1):
                        raw = line.strip()
                        if not raw:
                            continue
                        try:
                            record = json.loads(raw)
                            text = str(record.get("text", ""))
                            doc_id = str(record.get("id", f"{path.stem}:{line_no}"))
                            yield Document(doc_id=doc_id, path=path, text=text)
                        except json.JSONDecodeError as exc:
                            logger.warning("Bad JSON in %s:%d: %s", path, line_no, exc)
            except OSError as exc:
                logger.warning("Failed to open %s: %s", path, exc)
