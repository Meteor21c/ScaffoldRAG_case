import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Set


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


def get_effective_flags(item: Dict[str, Any]) -> Dict[str, bool]:
    has_reasoning_step = bool(item.get("has_reasoning_step", False))
    if not has_reasoning_step:
        has_reasoning_step = (item.get("num_reasoning_steps", 0) or 0) > 0

    perturbation_applied = bool(item.get("perturbation_applied", False))
    if not perturbation_applied:
        metadata = item.get("metadata", {})
        perturbation_applied = bool(metadata.get("perturbation_applied", False))

    is_correct = bool(item.get("is_correct", False))

    return {
        "has_reasoning_step": has_reasoning_step,
        "perturbation_applied": perturbation_applied,
        "is_correct": is_correct,
    }


def build_question_map(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    qmap = {}
    for item in results:
        q = normalize_text(item.get("question", ""))
        if q:
            qmap[q] = item
    return qmap


def summarize_run(name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    results = data.get("results", [])
    metrics = data.get("metrics", {})
    experiment = metrics.get("experiment", {})
    performance = metrics.get("performance", {})
    string_based = metrics.get("string_based", {})
    llm_evaluated = metrics.get("llm_evaluated", {})
    retrieval = metrics.get("retrieval", {})

    qmap = build_question_map(results)

    reasoning_questions = set()
    perturb_questions = set()

    for q, item in qmap.items():
        flags = get_effective_flags(item)
        if flags["has_reasoning_step"]:
            reasoning_questions.add(q)
        if flags["perturbation_applied"]:
            perturb_questions.add(q)

    return {
        "name": name,
        "num_results": len(results),
        "question_map": qmap,
        "reasoning_questions": reasoning_questions,
        "perturb_questions": perturb_questions,
        "experiment": experiment,
        "performance": performance,
        "string_based": string_based,
        "llm_evaluated": llm_evaluated,
        "retrieval": retrieval,
    }


def compare_pair(
    baseline_map: Dict[str, Dict[str, Any]],
    perturbed_map: Dict[str, Dict[str, Any]],
    questions: Set[str],
    require_perturbation_applied: bool = False,
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

        if require_perturbation_applied:
            if not get_effective_flags(p)["perturbation_applied"]:
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

        if get_effective_flags(p)["perturbation_applied"] and p_correct:
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
            "perturbation_applied": get_effective_flags(p)["perturbation_applied"],
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--answer_wrong", required=True)
    parser.add_argument("--summary_wrong", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case_dir", default=None)
    args = parser.parse_args()

    baseline_data = load_json(args.baseline)
    answer_data = load_json(args.answer_wrong)
    summary_data = load_json(args.summary_wrong)

    baseline = summarize_run("baseline", baseline_data)
    answer_wrong = summarize_run("answer_wrong", answer_data)
    summary_wrong = summarize_run("summary_wrong", summary_data)

    B = baseline["reasoning_questions"]
    A = answer_wrong["perturb_questions"]
    S = summary_wrong["perturb_questions"]

    strict = B & A & S
    BA = B & A
    BS = B & S

    answer_strict = compare_pair(
        baseline["question_map"], answer_wrong["question_map"], strict, require_perturbation_applied=True
    )
    summary_strict = compare_pair(
        baseline["question_map"], summary_wrong["question_map"], strict, require_perturbation_applied=True
    )

    answer_pair = compare_pair(
        baseline["question_map"], answer_wrong["question_map"], BA, require_perturbation_applied=True
    )
    summary_pair = compare_pair(
        baseline["question_map"], summary_wrong["question_map"], BS, require_perturbation_applied=True
    )

    output = {
        "set_stats": {
            "B_count": len(B),
            "A_count": len(A),
            "S_count": len(S),
            "B_intersect_A_intersect_S_count": len(strict),
            "B_intersect_A_count": len(BA),
            "B_intersect_S_count": len(BS),
            "strict_intersection_questions": sorted(strict),
        },
        "comparisons": {
            "answer_wrong_vs_baseline_on_BAS": answer_strict,
            "summary_wrong_vs_baseline_on_BAS": summary_strict,
            "answer_wrong_vs_baseline_on_BA": answer_pair,
            "summary_wrong_vs_baseline_on_BS": summary_pair,
        },
    }

    save_json(output, args.output)
    print(f"Saved comparison report to: {args.output}")

    if args.case_dir:
        case_dir = Path(args.case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

        save_json(answer_strict["cases"], case_dir / "answer_wrong_cases.json")
        save_json(summary_strict["cases"], case_dir / "summary_wrong_cases.json")
        save_json(answer_strict["details"], case_dir / "answer_wrong_details.json")
        save_json(summary_strict["details"], case_dir / "summary_wrong_details.json")
        print(f"Saved case files to: {case_dir}")


if __name__ == "__main__":
    main()