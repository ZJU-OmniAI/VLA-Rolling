from __future__ import annotations

import os
import sys
from pathlib import Path
from collections.abc import Sequence


def run_with_defaults(defaults: Sequence[str]) -> None:
    sys.argv = [sys.argv[0], *defaults, *sys.argv[1:]]
    from .pi05_instrumented_eval import main

    main()


def output_dir(benchmark: str) -> str:
    return os.environ.get("ACE_EVAL_OUTPUT_DIR", str(Path(__file__).resolve().parents[2] / "eval_logs" / f"ace_{benchmark}_smoke"))
