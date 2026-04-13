#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Set


def load_json(path: str) -> Dict[str, Any]:
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


def build_question_map(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    qmap = {}
    for item in results:
        q = normalize_text(item.get("question", ""))
        if q:
            qmap[q] = item
    return qmap


def get_baseline_reasoning_flag(item: Dict[str, Any]) -> bool:
    """
    Align with the original subset criterion:
    at least one true post-warmup reasoning step.
    For rolling-memory baseline runs, this corresponds to rounds > 0.
    """
    rounds = item.get("rounds", 0) or 0
    if rounds > 0:
        return True

    metadata = item.get("metadata", {})
    if (metadata.get("rounds", 0) or 0) > 0:
        return True

    # Fallback for any future format with explicit fields.
    if bool(item.get("has_reasoning_step", False)):
        return True
    if (item.get("num_reasoning_steps", 0) or 0) > 0:
        return True
    if bool(metadata.get("has_reasoning_step", False)):
        return True
    if (metadata.get("num_reasoning_steps", 0) or 0) > 0:
        return True

    return False


def get_hallucination_applied_flag(item: Dict[str, Any]) -> bool:
    """
    Robustly detect whether the rolling-memory summary_wrong hallucination was
    successfully injected in the perturbed run.
    """
    if bool(item.get("hallucination_applied", False)):
        return True
    if bool(item.get("perturbation_applied", False)):
        return True

    metadata = item.get("metadata", {})
    if bool(metadata.get("hallucination_applied", False)):
        return True
    if bool(metadata.get("perturbation_applied", False)):
        return True

    if item.get("hallucination_enabled", False) and str(item.get("hallucination_type", "")).lower() == "summary_wrong":
        return True
    if metadata.get("hallucination_enabled", False) and str(metadata.get("hallucination_type", "")).lower() == "summary_wrong":
        return True

    # More permissive fallback for custom scheme-B fields.
    if str(item.get("experiment_tag", "")).lower().find("summary_wrong") >= 0:
        return True
    if str(metadata.get("experiment_tag", "")).lower().find("summary_wrong") >= 0:
        return True

    return False


def summarize_baseline(data: Dict[str, Any]) -> Dict[str, Any]:
    results = data.get("results", [])
    qmap = build_question_map(results)
    reasoning_questions = {
        q for q, item in qmap.items()
        if get_baseline_reasoning_flag(item)
    }
    return {
        "question_map": qmap,
        "reasoning_questions": reasoning_questions,
        "num_results": len(results),
        "metrics": data.get("metrics", {}),
    }


def summarize_hallucinated(data: Dict[str, Any]) -> Dict[str, Any]:
    results = data.get("results", [])
    qmap = build_question_map(results)
    hallucination_questions = {
        q for q, item in qmap.items()
        if get_hallucination_applied_flag(item)
    }
    return {
        "question_map": qmap,
        "hallucination_questions": hallucination_questions,
        "num_results": len(results),
        "metrics": data.get("metrics", {}),
    }


def compare_pair(
    baseline_map: Dict[str, Dict[str, Any]],
    perturbed_map: Dict[str, Dict[str, Any]],
    questions: Set[str],
    require_hallucination_applied: bool = True,
) -> Dict[str, Any]:
    total = 0

    same_correctness = 0
    flip = 0
    correct_to_wrong = 0
    wrong_to_correct = 0
    answer_text_changed = 0
    self_recovery = 0
    correct_preserved = 0

    baseline_correct = 0
    perturbed_correct = 0

    details = []
    cases = {
        "correct_to_wrong": [],
        "wrong_to_correct": [],
        "changed_answer": [],
        "self_recovery": [],
        "correct_preserved": [],
    }

    for q in sorted(questions):
        if q not in baseline_map or q not in perturbed_map:
            continue

        b = baseline_map[q]
        p = perturbed_map[q]

        if require_hallucination_applied and not get_hallucination_applied_flag(p):
            continue

        b_correct = bool(b.get("is_correct", False))
        p_correct = bool(p.get("is_correct", False))

        b_answer = normalize_text(b.get("answer", ""))
        p_answer = normalize_text(p.get("answer", ""))
        gold = b.get("gold_answer", p.get("gold_answer"))

        total += 1
        baseline_correct += int(b_correct)
        perturbed_correct += int(p_correct)

        changed = (b_answer != p_answer)
        if changed:
            answer_text_changed += 1

        if b_correct == p_correct:
            same_correctness += 1
        else:
            flip += 1
            if b_correct and not p_correct:
                correct_to_wrong += 1
                cases["correct_to_wrong"].append({
                    "question": q,
                    "baseline_answer": b.get("answer"),
                    "perturbed_answer": p.get("answer"),
                    "gold_answer": gold,
                })
            elif (not b_correct) and p_correct:
                wrong_to_correct += 1
                cases["wrong_to_correct"].append({
                    "question": q,
                    "baseline_answer": b.get("answer"),
                    "perturbed_answer": p.get("answer"),
                    "gold_answer": gold,
                })

        if changed:
            cases["changed_answer"].append({
                "question": q,
                "baseline_correct": b_correct,
                "perturbed_correct": p_correct,
                "baseline_answer": b.get("answer"),
                "perturbed_answer": p.get("answer"),
                "gold_answer": gold,
            })

        if get_hallucination_applied_flag(p) and p_correct:
            self_recovery += 1
            cases["self_recovery"].append({
                "question": q,
                "baseline_correct": b_correct,
                "perturbed_correct": p_correct,
                "baseline_answer": b.get("answer"),
                "perturbed_answer": p.get("answer"),
                "gold_answer": gold,
            })

        if b_correct and p_correct:
            correct_preserved += 1
            cases["correct_preserved"].append({
                "question": q,
                "baseline_answer": b.get("answer"),
                "perturbed_answer": p.get("answer"),
                "gold_answer": gold,
            })

        details.append({
            "question": q,
            "baseline_correct": b_correct,
            "perturbed_correct": p_correct,
            "baseline_answer": b.get("answer"),
            "perturbed_answer": p.get("answer"),
            "gold_answer": gold,
            "answer_text_changed": changed,
            "hallucination_applied": get_hallucination_applied_flag(p),
        })

    baseline_acc = 100.0 * baseline_correct / total if total else 0.0
    perturbed_acc = 100.0 * perturbed_correct / total if total else 0.0

    return {
        "total": total,
        "baseline_accuracy": baseline_acc,
        "perturbed_accuracy": perturbed_acc,
        "accuracy_drop": baseline_acc - perturbed_acc,
        "same_correctness_rate": 100.0 * same_correctness / total if total else 0.0,
        "answer_flip_rate": 100.0 * flip / total if total else 0.0,
        "correct_to_wrong_rate": 100.0 * correct_to_wrong / total if total else 0.0,
        "wrong_to_correct_rate": 100.0 * wrong_to_correct / total if total else 0.0,
        "answer_text_changed_rate": 100.0 * answer_text_changed / total if total else 0.0,
        "self_recovery_rate": 100.0 * self_recovery / total if total else 0.0,
        "correct_preservation_rate": 100.0 * correct_preserved / total if total else 0.0,
        "counts": {
            "same_correctness": same_correctness,
            "flip": flip,
            "correct_to_wrong": correct_to_wrong,
            "wrong_to_correct": wrong_to_correct,
            "answer_text_changed": answer_text_changed,
            "self_recovery": self_recovery,
            "correct_preserved": correct_preserved,
        },
        "details": details,
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare rolling-memory baseline vs scheme-B summary_wrong hallucination."
    )
    parser.add_argument("--baseline", required=True,
                        help="Baseline subset result JSON")
    parser.add_argument("--summary_wrong", required=True,
                        help="Rolling-memory summary_wrong result JSON")
    parser.add_argument("--output", required=True,
                        help="Comparison output JSON")
    parser.add_argument("--case_dir", default=None,
                        help="Optional directory to save cases/details")
    args = parser.parse_args()

    baseline_data = load_json(args.baseline)
    summary_wrong_data = load_json(args.summary_wrong)

    baseline = summarize_baseline(baseline_data)
    summary_wrong = summarize_hallucinated(summary_wrong_data)

    B2 = baseline["reasoning_questions"]
    R = summary_wrong["hallucination_questions"]
    BR = B2 & R

    summary_compare = compare_pair(
        baseline["question_map"],
        summary_wrong["question_map"],
        BR,
        require_hallucination_applied=True,
    )

    output = {
        "set_stats": {
            "B2_count": len(B2),
            "R_count": len(R),
            "B2_intersect_R_count": len(BR),
            "strict_intersection_questions": sorted(BR),
            "selection_rule": "B2: baseline rounds > 0 (aligned with original ScaffoldRAG subset criterion); R: hallucination successfully applied",
        },
        "comparisons": {
            "summary_wrong_vs_baseline_on_B2R": summary_compare,
        },
    }

    save_json(output, args.output)
    print(f"Saved comparison report to: {args.output}")

    if args.case_dir:
        case_dir = Path(args.case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        save_json(summary_compare["cases"], case_dir / "summary_wrong_cases.json")
        save_json(summary_compare["details"], case_dir / "summary_wrong_details.json")
        print(f"Saved case files to: {case_dir}")


if __name__ == "__main__":
    main()
