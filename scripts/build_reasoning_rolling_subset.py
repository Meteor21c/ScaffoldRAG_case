#!/usr/bin/env python3
import json
import argparse
import random
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_text(x: str) -> str:
    if x is None:
        return ""
    return " ".join(str(x).strip().split())


def get_reasoning_flag_rolling(item: Dict[str, Any]) -> bool:
    """
    Align with the original ScaffoldRAG subset criterion:
    include a sample iff there is at least one real post-warmup reasoning round.

    For rolling-memory baseline runs, warm-up is not counted as a reasoning step.
    Therefore, rounds > 0 is the correct aligned criterion.
    """
    rounds = item.get("rounds", 0) or 0
    return rounds > 0


def main():
    parser = argparse.ArgumentParser(
        description="Build a rolling-memory reasoning-prone subset aligned with the original ScaffoldRAG subset criterion."
    )
    parser.add_argument("--baseline_results", type=str, required=True,
                        help="Full rolling-memory baseline result JSON")
    parser.add_argument("--source_dataset", type=str, required=True,
                        help="Original source dataset JSON")
    parser.add_argument("--output_subset", type=str, required=True,
                        help="Output subset dataset JSON (for rerunning hallucination)")
    parser.add_argument("--output_baseline_subset_results", type=str, required=True,
                        help="Output baseline result subset JSON (matched by question)")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only_correct", action="store_true")
    parser.add_argument("--only_wrong", action="store_true")
    args = parser.parse_args()

    baseline_data = load_json(args.baseline_results)
    source_data = load_json(args.source_dataset)

    results = baseline_data.get("results", [])
    if isinstance(source_data, dict) and "data" in source_data:
        dataset = source_data["data"]
    else:
        dataset = source_data

    selected_questions: List[str] = []
    selected_result_items: List[Dict[str, Any]] = []

    for item in results:
        q = normalize_text(item.get("question", ""))
        if not q:
            continue

        if not get_reasoning_flag_rolling(item):
            continue

        is_correct = bool(item.get("is_correct", False))
        if args.only_correct and not is_correct:
            continue
        if args.only_wrong and is_correct:
            continue

        selected_questions.append(q)
        selected_result_items.append(item)

    selected_question_set = set(selected_questions)

    matched_dataset: List[Dict[str, Any]] = []
    unmatched_questions = set(selected_question_set)

    for ex in dataset:
        q = normalize_text(ex.get("question", ""))
        if q in selected_question_set:
            matched_dataset.append(ex)
            unmatched_questions.discard(q)

    random.seed(args.seed)
    if args.max_samples is not None and len(matched_dataset) > args.max_samples:
        matched_dataset = random.sample(matched_dataset, args.max_samples)

    subset_questions = {normalize_text(x.get("question", "")) for x in matched_dataset}
    subset_result_items = [
        x for x in selected_result_items
        if normalize_text(x.get("question", "")) in subset_questions
    ]

    # Keep the original overall structure so downstream tools can read it naturally.
    baseline_subset = {
        "model": baseline_data.get("model"),
        "metrics": baseline_data.get("metrics", {}),
        "results": subset_result_items,
    }

    save_json(matched_dataset, args.output_subset)
    save_json(baseline_subset, args.output_baseline_subset_results)

    meta = {
        "source_baseline_results": args.baseline_results,
        "source_dataset": args.source_dataset,
        "selection_rule": "rounds > 0 (aligned with original ScaffoldRAG subset criterion: at least one post-warmup reasoning step)",
        "only_correct": args.only_correct,
        "only_wrong": args.only_wrong,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "selected_reasoning_result_count": len(selected_questions),
        "matched_subset_count": len(matched_dataset),
        "matched_baseline_result_count": len(subset_result_items),
        "unmatched_question_count": len(unmatched_questions),
        "unmatched_questions": sorted(unmatched_questions),
    }

    subset_meta_path = str(Path(args.output_subset).with_suffix(".meta.json"))
    baseline_meta_path = str(Path(args.output_baseline_subset_results).with_suffix(".meta.json"))
    save_json(meta, subset_meta_path)
    save_json(meta, baseline_meta_path)

    print(f"Saved subset dataset to: {args.output_subset}")
    print(f"Saved baseline subset results to: {args.output_baseline_subset_results}")
    print(f"Saved subset meta to: {subset_meta_path}")
    print(f"Saved baseline meta to: {baseline_meta_path}")
    print(f"reasoning-active selected from baseline results: {len(selected_questions)}")
    print(f"matched in source dataset: {len(matched_dataset)}")
    print(f"matched in baseline subset results: {len(subset_result_items)}")
    print(f"unmatched questions: {len(unmatched_questions)}")


if __name__ == "__main__":
    main()
