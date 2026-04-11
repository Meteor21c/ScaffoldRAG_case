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


def get_reasoning_flag(item: Dict[str, Any]) -> bool:
    has_reasoning = bool(item.get("has_reasoning_step", False))
    num_steps = item.get("num_reasoning_steps", 0) or 0
    return has_reasoning or num_steps > 0


def get_warmup_stop_flag(item: Dict[str, Any]) -> bool:
    if "warmup_early_stop" in item:
        return bool(item["warmup_early_stop"])
    metadata = item.get("metadata", {})
    if "warmup_early_stop" in metadata:
        return bool(metadata["warmup_early_stop"])
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_results", type=str, required=True)
    parser.add_argument("--source_dataset", type=str, required=True)
    parser.add_argument("--output_subset", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude_warmup_stop", action="store_true")
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

    selected_questions = []
    selected_result_items = []

    for item in results:
        q = normalize_text(item.get("question", ""))
        if not q:
            continue

        if not get_reasoning_flag(item):
            continue

        if args.exclude_warmup_stop and get_warmup_stop_flag(item):
            continue

        is_correct = bool(item.get("is_correct", False))
        if args.only_correct and not is_correct:
            continue
        if args.only_wrong and is_correct:
            continue

        selected_questions.append(q)
        selected_result_items.append(item)

    selected_question_set = set(selected_questions)

    matched_dataset = []
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

    meta = {
        "source_baseline_results": args.baseline_results,
        "source_dataset": args.source_dataset,
        "exclude_warmup_stop": args.exclude_warmup_stop,
        "only_correct": args.only_correct,
        "only_wrong": args.only_wrong,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "selected_reasoning_result_count": len(selected_questions),
        "matched_subset_count": len(matched_dataset),
        "unmatched_question_count": len(unmatched_questions),
        "unmatched_questions": sorted(unmatched_questions),
    }

    # 主输出：纯 list，供 run.py 直接读取
    save_json(matched_dataset, args.output_subset)

    # 旁路输出：meta 信息
    meta_path = str(Path(args.output_subset).with_suffix(".meta.json"))
    save_json(meta, meta_path)

    print(f"Saved subset data to: {args.output_subset}")
    print(f"Saved subset meta to: {meta_path}")
    print(f"reasoning-active selected from results: {len(selected_questions)}")
    print(f"matched in source dataset: {len(matched_dataset)}")
    print(f"unmatched questions: {len(unmatched_questions)}")


if __name__ == "__main__":
    main()