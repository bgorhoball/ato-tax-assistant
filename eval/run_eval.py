#!/usr/bin/env python3
"""
RAG eval runner for ato-tax-assistant.

Usage
-----
    python eval/run_eval.py                        # run all non-skip cases
    python eval/run_eval.py --category easy        # one category
    python eval/run_eval.py --dry-run              # list cases, no LLM calls
    python eval/run_eval.py --output results.json  # save to JSON
    python eval/run_eval.py --k 6                  # retrieve 6 chunks
    python eval/run_eval.py --verbose              # print full answers

Exit codes: 0 = all run cases pass accuracy >= threshold; 1 = failures exist.
Threshold is set by --pass-threshold (default 0.5 per case).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: allow running as  `python eval/run_eval.py`  from project root
# or as  `python run_eval.py`  from inside eval/.
# ---------------------------------------------------------------------------
_EVAL_DIR = Path(__file__).parent
_SRC_DIR = _EVAL_DIR.parent / "src"
sys.path.insert(0, str(_EVAL_DIR))
sys.path.insert(0, str(_SRC_DIR))

from cases import EVAL_CASES, EvalCase  # noqa: E402
from metrics import CaseResult, compute_retrieval_recall, evaluate  # noqa: E402


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_one(engine, case: EvalCase, k: int, retrieval_only: bool = False) -> CaseResult:
    """Call the engine and compute all metrics for a single case."""
    t0 = time.time()
    try:
        if retrieval_only:
            # Embedding call only — no LLM generate quota consumed.
            results_with_scores = engine.vectorstore.similarity_search_with_score(
                case.question, k=k
            )
            elapsed = time.time() - t0
            retrieved_pages = [
                doc.metadata.get("page", -1) for doc, _ in results_with_scores
            ]
            return CaseResult(
                case_id=case.id,
                category=case.category,
                question=case.question,
                answer="",
                retrieved_pages=retrieved_pages,
                answer_accuracy=None,
                retrieval_recall=compute_retrieval_recall(retrieved_pages, case),
                citation_faithfulness=None,
                hallucination_penalty=None,
                latency_s=elapsed,
            )

        result = engine.ask(case.question, k=k)
        elapsed = time.time() - t0
        retrieved_pages = [
            doc.metadata.get("page", -1) for doc in result.source_documents
        ]
        return evaluate(result.answer, retrieved_pages, case, latency_s=elapsed)
    except Exception as exc:
        elapsed = time.time() - t0
        cr = evaluate("", [], case, latency_s=elapsed)
        cr.error = str(exc)
        return cr


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_NA = " N/A"


def _fmt(value: float | None, width: int = 5) -> str:
    if value is None:
        return _NA.rjust(width)
    return f"{value:.3f}".rjust(width)


def _mean(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def print_case(cr: CaseResult, verbose: bool = False) -> None:
    if cr.answer_accuracy is None:
        verdict = "·"  # not evaluated (e.g. --retrieval-only)
    else:
        verdict = "✓" if cr.answer_accuracy >= 0.5 else "✗"
    if cr.error:
        verdict = "E"
    print(
        f"\n[{cr.case_id}] {cr.question[:72]}"
        + ("..." if len(cr.question) > 72 else "")
    )
    if cr.error:
        print(f"  ERROR: {cr.error}")
        return
    print(
        f"  {verdict}  acc={_fmt(cr.answer_accuracy)}  "
        f"recall={_fmt(cr.retrieval_recall)}  "
        f"faith={_fmt(cr.citation_faithfulness)}  "
        f"hall_ok={_fmt(cr.hallucination_penalty)}  "
        f"[{cr.latency_s:.1f}s]"
    )
    if verbose:
        print(f"  Retrieved pages : {cr.retrieved_pages}")
        print(f"  Answer snippet  : {cr.answer[:300]}" + ("…" if len(cr.answer) > 300 else ""))


def print_summary(results: list[CaseResult], skipped: list[EvalCase]) -> None:
    cats = ["easy", "medium", "hard", "hallucination"]
    hline = "-" * 72

    print(f"\n{'=' * 72}")
    print("EVAL SUMMARY — ato-tax-assistant RAG")
    print(f"{'=' * 72}")
    print(
        f"{'Category':<16} {'Cases':>5}  "
        f"{'Accuracy':>9} {'Recall':>9} {'Faithful':>9} {'Hall OK':>9}"
    )
    print(hline)

    all_acc: list[float] = []
    all_rec: list[float] = []
    all_fai: list[float] = []
    all_hal: list[float] = []

    for cat in cats:
        cat_results = [r for r in results if r.category == cat and not r.error]
        if not cat_results:
            continue
        acc = [r.answer_accuracy for r in cat_results]
        rec = [r.retrieval_recall for r in cat_results]
        fai = [r.citation_faithfulness for r in cat_results]
        hal = [r.hallucination_penalty for r in cat_results]

        all_acc.extend(v for v in acc if v is not None)
        all_rec.extend(v for v in rec if v is not None)
        all_fai.extend(v for v in fai if v is not None)
        all_hal.extend(v for v in hal if v is not None)

        print(
            f"{cat:<16} {len(cat_results):>5}  "
            f"{_fmt(_mean(acc)):>9} {_fmt(_mean(rec)):>9} "
            f"{_fmt(_mean(fai)):>9} {_fmt(_mean(hal)):>9}"
        )

    print(hline)
    run_count = len([r for r in results if not r.error])
    error_count = len([r for r in results if r.error])
    print(
        f"{'TOTAL':<16} {run_count:>5}  "
        f"{_fmt(_mean(all_acc)):>9} {_fmt(_mean(all_rec)):>9} "
        f"{_fmt(_mean(all_fai)):>9} {_fmt(_mean(all_hal)):>9}"
    )
    print(f"\nSkipped : {len(skipped)}  Errors : {error_count}")

    if skipped:
        print("\nSkipped cases (unimplemented features):")
        for sc in skipped:
            print(f"  [{sc.id}] {sc.skip_reason}")


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

def to_dict(cr: CaseResult) -> dict:
    return {
        "id": cr.case_id,
        "category": cr.category,
        "question": cr.question,
        "answer": cr.answer,
        "retrieved_pages": cr.retrieved_pages,
        "latency_s": round(cr.latency_s, 3),
        "metrics": {
            "answer_accuracy": cr.answer_accuracy,
            "retrieval_recall": cr.retrieval_recall,
            "citation_faithfulness": cr.citation_faithfulness,
            "hallucination_penalty": cr.hallucination_penalty,
        },
        "error": cr.error,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="RAG eval runner for ato-tax-assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--category",
        choices=["easy", "medium", "hard", "hallucination", "skip"],
        help="Run only cases in this category.",
    )
    parser.add_argument(
        "--id",
        dest="case_id",
        help="Run specific case(s) by ID, comma-separated (e.g. EASY-01,HARD-02).",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Write JSON results to FILE.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List cases without calling the engine.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help=(
            "Skip the LLM and compute retrieval recall only. Uses embedding "
            "quota, not the 20/day gemini-2.5-flash generate quota."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of chunks to retrieve per question (default: 4).",
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=0.5,
        metavar="T",
        help="answer_accuracy >= T counts as a pass (default: 0.5).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print full answers and retrieved pages.",
    )
    args = parser.parse_args()

    # Filter cases
    cases = list(EVAL_CASES)
    if args.category:
        cases = [c for c in cases if c.category == args.category]
    if args.case_id:
        wanted = {cid.strip() for cid in args.case_id.split(",")}
        cases = [c for c in cases if c.id in wanted]
    if not cases:
        print("No cases match the given filters.")
        return 1

    skipped = [c for c in cases if c.skip_reason]
    run_cases = [c for c in cases if not c.skip_reason]

    print(f"ato-tax-assistant RAG Eval  |  {len(cases)} total, {len(run_cases)} to run, {len(skipped)} skipped")
    mode = "retrieval-only" if args.retrieval_only else "full"
    print(f"k={args.k}  pass_threshold={args.pass_threshold}  mode={mode}  dry_run={args.dry_run}\n")

    if args.dry_run:
        for c in cases:
            tag = f"  [SKIP] {c.skip_reason}" if c.skip_reason else ""
            print(f"  {c.id:<12} ({c.category}){tag}")
            print(f"    Q: {c.question[:80]}")
        return 0

    # Announce skipped cases upfront
    for sc in skipped:
        print(f"[SKIP] {sc.id}: {sc.skip_reason}\n")

    if not run_cases:
        print("Nothing to run.")
        return 0

    # Initialise engine
    try:
        from rag_engine import TaxRagEngine

        chroma_path = str(_EVAL_DIR.parent / "chroma_db")
        engine = TaxRagEngine(
            model_provider="google",
            persist_directory=chroma_path,
        )
        engine.load_vectorstore()
    except Exception as exc:
        print(f"FATAL: Cannot initialise engine: {exc}")
        print("Ensure GOOGLE_API_KEY is set and chroma_db/ exists (run ingest first).")
        return 1

    # Run cases
    results: list[CaseResult] = []
    for i, case in enumerate(run_cases, 1):
        print(f"({i}/{len(run_cases)}) {case.id} …", end="", flush=True)
        cr = run_one(engine, case, k=args.k, retrieval_only=args.retrieval_only)
        print(" done")
        print_case(cr, verbose=args.verbose)
        results.append(cr)

    # Summary
    print_summary(results, skipped)

    # JSON output
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "k": args.k,
                "pass_threshold": args.pass_threshold,
                "total_cases": len(cases),
                "run_cases": len(run_cases),
                "skipped_cases": len(skipped),
            },
            "skipped": [
                {"id": sc.id, "skip_reason": sc.skip_reason} for sc in skipped
            ],
            "results": [to_dict(r) for r in results],
        }
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nResults written → {out}")

    # Exit code: fail if any evaluable case is below threshold
    failures = [
        r for r in results
        if not r.error
        and r.answer_accuracy is not None
        and r.answer_accuracy < args.pass_threshold
    ]
    if failures:
        print(f"\n{len(failures)} case(s) below threshold {args.pass_threshold}:")
        for r in failures:
            print(f"  {r.case_id}  accuracy={r.answer_accuracy:.3f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
