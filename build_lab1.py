import os
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

PROJECT_DIR = Path(".") if Path.cwd().name == "findex" else Path("findex")
SRC_DIR = PROJECT_DIR / "src" / "findex"
TESTS_DIR = PROJECT_DIR / "tests"
DATA_DIR = PROJECT_DIR / "data"

PYPROJECT_TOML = """\
[project]
name = "findex"
version = "0.1.0"
description = "Full-text search engine built across 8 Python labs"
readme = "README.md"
requires-python = ">=3.12"
dependencies = []

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/findex"]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "ruff>=0.8",
]
"""

GITIGNORE = """\
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
.ruff_cache/
dist/
build/
*.egg-info/
data/
"""

CORPUS_PY = """\
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
    \"\"\"Lazily stream documents one by one from .txt files or .jsonl.\"\"\"
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
"""

TOKENIZE_PY = """\
import re
import unicodedata
from collections.abc import Iterator

# Tokenization rules:
# 1. Unicode NFC normalization merges base letters and combining marks (e.g. café).
# 2. str.casefold() performs aggressive caseless folding (e.g. 'ß' -> 'ss').
# 3. Apostrophes (' and ’) and hyphens (-) are preserved inside words
#    (e.g. "п'ять", "don't", "state-of-the-art") and stripped at word boundaries.
# 4. Digits are preserved as standalone numbers ("2026") or parts of terms ("utf8").
TOKEN_RE = re.compile(r"\\w+(?:['’-]\\w+)*", re.UNICODE)


def tokenize(text: str) -> Iterator[str]:
    \"\"\"Stream normalized tokens from input text.\"\"\"
    normalized = unicodedata.normalize("NFC", text).casefold()
    for match in TOKEN_RE.finditer(normalized):
        yield match.group(0)
"""

STATS_PY = """\
import argparse
import time
import tracemalloc
from collections import Counter
from itertools import islice
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def compute_stats_lazy(
    root: Path, limit: int | None = None
) -> tuple[int, int, Counter[str]]:
    \"\"\"Stream documents and tokens in constant memory (only Counter grows).\"\"\"
    docs = iter_documents(root)
    if limit is not None:
        docs = islice(docs, limit)

    doc_count = 0
    total_tokens = 0
    counts: Counter[str] = Counter()

    for doc in docs:
        doc_count += 1
        for tok in tokenize(doc.text):
            total_tokens += 1
            counts[tok] += 1

    return doc_count, total_tokens, counts


def compute_stats_eager(
    root: Path, limit: int | None = None
) -> tuple[int, int, Counter[str]]:
    \"\"\"Deliberately materialize all documents and token lists into RAM.\"\"\"
    docs_iter = iter_documents(root)
    if limit is not None:
        docs_iter = islice(docs_iter, limit)

    all_docs = list(docs_iter)
    all_tokens = [list(tokenize(doc.text)) for doc in all_docs]

    counts: Counter[str] = Counter()
    total_tokens = 0
    for doc_tokens in all_tokens:
        total_tokens += len(doc_tokens)
        counts.update(doc_tokens)

    return len(all_docs), total_tokens, counts


def measure(fn, root: Path, limit: int | None = None):
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(root, limit)
    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak_bytes


def format_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute corpus statistics for findex")
    parser.add_argument("corpus", type=Path, help="Path to corpus directory or file")
    parser.add_argument("--limit", type=int, default=None, help="Limit document count")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Benchmark eager (lists) vs lazy (generators) and print Markdown table",
    )
    args = parser.parse_args()

    if args.compare:
        eager_res, eager_time, eager_peak = measure(
            compute_stats_eager, args.corpus, args.limit
        )
        lazy_res, lazy_time, lazy_peak = measure(
            compute_stats_lazy, args.corpus, args.limit
        )

        print("\\n| Version | Documents | Peak memory | Elapsed |")
        print("|---|---|---|---|")
        print(
            f"| eager (lists) | {eager_res[0]} | "
            f"{format_mb(eager_peak)} | {eager_time:.3f} s |"
        )
        print(
            f"| lazy (generators) | {lazy_res[0]} | "
            f"{format_mb(lazy_peak)} | {lazy_time:.3f} s |\\n"
        )
        return

    (doc_count, total_tokens, counts), elapsed, peak_bytes = measure(
        compute_stats_lazy, args.corpus, args.limit
    )

    print(f"Documents:       {doc_count}")
    print(f"Total tokens:    {total_tokens}")
    print(f"Vocabulary size: {len(counts)}")
    print(f"Elapsed time:    {elapsed:.3f} s")
    print(f"Peak memory:     {format_mb(peak_bytes)}")
    print("\\nTop-50 terms:")
    for rank, (term, freq) in enumerate(counts.most_common(50), start=1):
        print(f"{rank:2d}. {term:<20} {freq}")


if __name__ == "__main__":
    main()
"""

TEST_TOKENIZE_PY = """\
import inspect
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def test_generators_are_lazy() -> None:
    assert inspect.isgeneratorfunction(iter_documents)
    assert inspect.isgeneratorfunction(tokenize)


def test_mixed_case_and_casefold() -> None:
    assert list(tokenize("Python Straße STRASSE")) == ["python", "strasse", "strasse"]


def test_cyrillic_text() -> None:
    assert list(tokenize("Пошукова система працює швидко!")) == [
        "пошукова",
        "система",
        "працює",
        "швидко",
    ]


def test_unicode_nfc_normalization() -> None:
    composed = "café"
    decomposed = "cafe\\u0301"
    assert composed != decomposed
    assert list(tokenize(composed)) == list(tokenize(decomposed)) == ["café"]


def test_apostrophes_and_hyphens() -> None:
    text = "П'ять state-of-the-art рішень don't --break--"
    assert list(tokenize(text)) == [
        "п'ять",
        "state-of-the-art",
        "рішень",
        "don't",
        "break",
    ]


def test_digits_and_alphanumeric() -> None:
    assert list(tokenize("Python 3.14 у 2026 році, UTF8")) == [
        "python",
        "3",
        "14",
        "у",
        "2026",
        "році",
        "utf8",
    ]


def test_punctuation_and_empty_string() -> None:
    assert list(tokenize("")) == []
    assert list(tokenize("...,,,!!! ??? — () []")) == []


def test_iter_documents_streams_one_by_one(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("Перший документ", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Другий документ", encoding="utf-8")

    gen = iter_documents(tmp_path)
    first = next(gen)
    assert first.doc_id == "a"
    assert first.text == "Перший документ"
"""


def build_local_corpus(data_dir: Path) -> tuple[int, float]:
    """Build ~1,200 real .txt documents from installed Python stdlib + Ukrainian texts."""
    data_dir.mkdir(parents=True, exist_ok=True)
    stdlib_dir = Path(sysconfig.get_paths()["stdlib"])
    doc_counter = 0
    total_bytes = 0

    for py_file in sorted(stdlib_dir.rglob("*.py")):
        if "site-packages" in py_file.parts or "__pycache__" in py_file.parts:
            continue
        try:
            content = py_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(content) < 400:
            continue

        rel_name = "_".join(py_file.relative_to(stdlib_dir).with_suffix("").parts)
        rel_name = re.sub(r"[^a-zA-Z0-9_-]", "_", rel_name)[:80]
        out_file = data_dir / f"{doc_counter:04d}_{rel_name}.txt"
        out_file.write_text(content, encoding="utf-8")
        doc_counter += 1
        total_bytes += len(content.encode("utf-8"))

        if doc_counter >= 1180:
            break

    uk_notes = [
        "Пошукова система findex використовує ліниві генератори та протокол ітерації Python. "
        "Обробка тексту включає Unicode NFC-нормалізацію, регістронезалежне згортання casefold "
        "та потоковий пошук токенів через re.finditer без завантаження всього корпусу в пам'ять.\n"
    ] * 25

    for idx in range(1, 21):
        text = f"Українська технічна нотатка #{idx}\n" + "".join(uk_notes)
        out_file = data_dir / f"{doc_counter:04d}_uk_note_{idx:02d}.txt"
        out_file.write_text(text, encoding="utf-8")
        doc_counter += 1
        total_bytes += len(text.encode("utf-8"))

    return doc_counter, total_bytes / (1024 * 1024)


def build_readme(
    doc_count: int,
    corpus_mb: float,
    eager_peak: str,
    eager_time: str,
    lazy_peak: str,
    lazy_time: str,
) -> str:
    return f"""\
# findex — Пошукова система на Python

## Лабораторна робота 1: Ітератори, генератори та корпус

### Джерело корпусу
Корпус складається з **{doc_count} текстових документів** (~{corpus_mb:.1f} МБ) у директорії `data/`, сформованих із модулів та документації стандартної бібліотеки Python (`sysconfig.get_paths()["stdlib"]`) і україномовних технічних нотаток. Директорія `data/` додана до `.gitignore` і не зберігається у репозиторії.

### Правила токенізації (`src/findex/tokenize.py`)
1. **Unicode-нормалізація:** `unicodedata.normalize("NFC", text)` зводить комбіновані символи до єдиного кодпойнта (`cafe\\u0301` → `café`).
2. **Регістр:** `str.casefold()` виконує регістронезалежне згортання для кирилиці, латиниці та спеціальних літер (`ß` → `ss`).
3. **Апострофи та дефіси:** зберігаються всередині слів (`п'ять`, `don't`, `state-of-the-art`) за регулярним виразом `\\w+(?:['’-]\\w+)*`, але відкидаються на межах слова.
4. **Цифри:** зберігаються як окремі числа (`2026`) або частини буквено-цифрових термів (`utf8`).

### Порівняння витрат пам'яті (Eager vs Lazy)

| Version | Documents | Peak memory | Elapsed |
|---|---|---|---|
| eager (lists) | {doc_count} | {eager_peak} | {eager_time} |
| lazy (generators) | {doc_count} | {lazy_peak} | {lazy_time} |

**Пояснення різниці:** У жадібній (`eager`) версії виклик `list(iter_documents(...))` одночасно завантажує всі тексти документів у пам'ять, а списковий вираз `[list(tokenize(...))]` створює в RAM мільйони об'єктів `str` та вкладених списків `list` до початку підрахунку. У лінивій (`lazy`) версії генератори віддають по одному документу й по одному токену за крок, тому оброблені рядки одразу звільняються збирачем сміття, а пікова пам'ять обмежується розміром одного документа та словником унікальних термів `Counter`.
"""


def find_git() -> str | None:
    git_bin = shutil.which("git")
    if git_bin:
        return git_bin
    for candidate in (
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files (x86)\Git\cmd\git.exe",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def main() -> None:
    print("1/5 Створення структури проєкту...")
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    TESTS_DIR.mkdir(parents=True, exist_ok=True)

    (PROJECT_DIR / "pyproject.toml").write_text(PYPROJECT_TOML, encoding="utf-8")
    (PROJECT_DIR / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    (SRC_DIR / "__init__.py").write_text("", encoding="utf-8")
    (SRC_DIR / "corpus.py").write_text(CORPUS_PY, encoding="utf-8")
    (SRC_DIR / "tokenize.py").write_text(TOKENIZE_PY, encoding="utf-8")
    (SRC_DIR / "stats.py").write_text(STATS_PY, encoding="utf-8")
    (TESTS_DIR / "test_tokenize.py").write_text(TEST_TOKENIZE_PY, encoding="utf-8")

    print("2/5 Формування корпусу в data/...")
    doc_count, corpus_mb = build_local_corpus(DATA_DIR)
    print(f"   Створено {doc_count} документів ({corpus_mb:.2f} МБ)")

    print("3/5 Виконання замірів пам'яті eager vs lazy...")
    sys.path.insert(0, str((PROJECT_DIR / "src").resolve()))
    from findex.stats import (
        compute_stats_eager,
        compute_stats_lazy,
        format_mb,
        measure,
    )

    eager_res, eager_t, eager_p = measure(compute_stats_eager, DATA_DIR)
    lazy_res, lazy_t, lazy_p = measure(compute_stats_lazy, DATA_DIR)

    readme_text = build_readme(
        doc_count=eager_res[0],
        corpus_mb=corpus_mb,
        eager_peak=format_mb(eager_p),
        eager_time=f"{eager_t:.3f} s",
        lazy_peak=format_mb(lazy_p),
        lazy_time=f"{lazy_t:.3f} s",
    )
    (PROJECT_DIR / "README.md").write_text(readme_text, encoding="utf-8")
    print(f"   Eager: {format_mb(eager_p)} ({eager_t:.3f} s)")
    print(f"   Lazy:  {format_mb(lazy_p)} ({lazy_t:.3f} s)")

    print("4/5 Запуск тестів та лінтера...")
    if shutil.which("uv"):
        subprocess.run(["uv", "sync", "--dev"], cwd=PROJECT_DIR, check=False)
        subprocess.run(["uv", "run", "pytest"], cwd=PROJECT_DIR, check=False)
        subprocess.run(
            ["uv", "run", "ruff", "check", "src", "tests"],
            cwd=PROJECT_DIR,
            check=False,
        )
    else:
        subprocess.run([sys.executable, "-m", "pytest"], cwd=PROJECT_DIR, check=False)

    print("5/5 Ініціалізація Git та створення тегу lab-01...")
    git_bin = find_git()
    if git_bin:
        subprocess.run([git_bin, "init"], cwd=PROJECT_DIR, check=False)
        subprocess.run([git_bin, "add", "."], cwd=PROJECT_DIR, check=False)
        subprocess.run(
            [
                git_bin,
                "commit",
                "-m",
                "Lab 1: streaming corpus loader, tokenizer, stats, and benchmarks",
            ],
            cwd=PROJECT_DIR,
            check=False,
        )
        subprocess.run([git_bin, "tag", "-f", "lab-01"], cwd=PROJECT_DIR, check=False)
        subprocess.run([git_bin, "branch", "-M", "main"], cwd=PROJECT_DIR, check=False)
        print("\nГотово! Усе зібрано в поточній папці findex.")
    else:
        print("\nПроєкт зібрано, але git не знайдено. Запустіть git із Git Bash.")


if __name__ == "__main__":
    main()