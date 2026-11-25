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
    """Stream documents and tokens in constant memory (only Counter grows)."""
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
    """Deliberately materialize all documents and token lists into RAM."""
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

        print("\n| Version | Documents | Peak memory | Elapsed |")
        print("|---|---|---|---|")
        print(
            f"| eager (lists) | {eager_res[0]} | "
            f"{format_mb(eager_peak)} | {eager_time:.3f} s |"
        )
        print(
            f"| lazy (generators) | {lazy_res[0]} | "
            f"{format_mb(lazy_peak)} | {lazy_time:.3f} s |\n"
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
    print("\nTop-50 terms:")
    for rank, (term, freq) in enumerate(counts.most_common(50), start=1):
        print(f"{rank:2d}. {term:<20} {freq}")


if __name__ == "__main__":
    main()
