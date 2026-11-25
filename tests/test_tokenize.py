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
    decomposed = "cafe\u0301"
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
