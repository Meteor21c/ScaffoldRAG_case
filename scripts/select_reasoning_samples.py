import json
import argparse
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline_result",
        type=str,
        required=True,
        help="Path to baseline evaluation result json, e.g. baseline_hotpot_candidate.json"
    )
    parser.add_argument(
        "--original_dataset",
        type=str,
        required=True,
        help="Path to original dataset, e.g. dataset/hotpotqa.json"
    )
    parser.add_argument(
        "--output_dataset",
        type=str,
        required=True,
        help="Path to output filtered dataset, e.g. dataset/hotpotqa_reasoning_26.json"
    )
    parser.add_argument(
        "--output_questions",
        type=str,
        default=None,
        help="Optional path to save selected question list"
    )
    args = parser.parse_args()

    baseline = load_json(args.baseline_result)
    original_dataset = load_json(args.original_dataset)

    baseline_results = baseline.get("results", [])

    selected_questions = []
    selected_records_from_result = []

    for item in baseline_results:
        num_reasoning_steps = item.get("num_reasoning_steps", 0)
        has_reasoning_step = item.get("has_reasoning_step", False)

        if has_reasoning_step or num_reasoning_steps > 0:
            q = item.get("question")
            if q is not None:
                selected_questions.append(q)
                selected_records_from_result.append({
                    "question": q,
                    "gold_answer": item.get("gold_answer"),
                    "num_reasoning_steps": num_reasoning_steps,
                    "rounds": item.get("rounds", 0),
                    "is_correct": item.get("is_correct", False),
                })

    selected_question_set = set(selected_questions)

    filtered_dataset = []
    for ex in original_dataset:
        if ex.get("question") in selected_question_set:
            filtered_dataset.append(ex)

    print(f"Total baseline results: {len(baseline_results)}")
    print(f"Selected reasoning samples: {len(selected_questions)}")
    print(f"Matched original dataset samples: {len(filtered_dataset)}")

    save_json(filtered_dataset, args.output_dataset)

    if args.output_questions:
        save_json({
            "count": len(selected_questions),
            "questions": selected_questions,
            "records": selected_records_from_result,
        }, args.output_questions)

    print(f"Saved filtered dataset to: {args.output_dataset}")
    if args.output_questions:
        print(f"Saved selected questions to: {args.output_questions}")


if __name__ == "__main__":
    main()