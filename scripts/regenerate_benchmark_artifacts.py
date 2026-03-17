from __future__ import annotations

import subprocess
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_RESULTS = REPO_ROOT / "benchmark_results"


def _run(script_name: str, *extra_args: str) -> None:
    command = [sys.executable, str(REPO_ROOT / "scripts" / script_name), *extra_args]
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def main() -> None:
    BENCHMARK_RESULTS.mkdir(parents=True, exist_ok=True)

    _run(
        "benchmark_rwif_vs_dense_baseline.py",
        "--output-json",
        str(BENCHMARK_RESULTS / "rwif_vs_dense_cosine_baseline.json"),
        "--output-md",
        str(BENCHMARK_RESULTS / "rwif_vs_dense_cosine_baseline.md"),
    )
    _run(
        "benchmark_rwif_vs_dense_text_array.py",
        "--output-json",
        str(BENCHMARK_RESULTS / "rwif_vs_dense_text_array_baseline.json"),
        "--output-md",
        str(BENCHMARK_RESULTS / "rwif_vs_dense_text_array_baseline.md"),
    )
    _run(
        "benchmark_rwif_vs_dense_text_corpus.py",
        "--output-json",
        str(BENCHMARK_RESULTS / "rwif_vs_dense_text_corpus_baseline.json"),
        "--output-md",
        str(BENCHMARK_RESULTS / "rwif_vs_dense_text_corpus_baseline.md"),
    )


if __name__ == "__main__":
    main()