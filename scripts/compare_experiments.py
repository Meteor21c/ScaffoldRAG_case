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


def normalize_question(q: str) -> str:
    if q is None:
        return ""
    return " ".join(str(q).strip().split())


def get_effective_flags(item: Dict[str, Any]) -> Dict[str, bool]:
    """
    Extract effective flags from one result item.
    """
    has_reasoning_step = bool(item.get("has_reasoning_step", False))
    if not has_reasoning_step:
        has_reasoning_step = item.get("num_reasoning_steps", 0) > 0

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
    """
    Map normalized question -> result item.
    Assumes one unique item per question.
    """
    qmap = {}
    for item in results:
        q = normalize_question(item.get("question", ""))
        if q:
            qmap[q] = item
    return qmap


def accuracy_on_questions(qmap: Dict[str, Dict[str, Any]], questions: Set[str]) -> float:
    if not questions:
        return 0.0
    correct = 0
    total = 0
    for q in questions:
        if q in qmap:
            total += 1
            if bool(qmap[q].get("is_correct", False)):
                correct += 1
    return 100.0 * correct / total if total > 0 else 0.0


def summarize_run(name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    results = data.get("results", [])
    metrics = data.get("metrics", {})
    experiment = metrics.get("experiment", {})
    raw = metrics.get("raw", {})
    performance = metrics.get("performance", {})
    string_based = metrics.get("string_based", {})
    llm_evaluated = metrics.get("llm_evaluated", {})
    retrieval = metrics.get("retrieval", {})

    qmap = build_question_map(results)

    effective_reasoning_questions = set()
    effective_perturb_questions = set()

    for q, item in qmap.items():
        flags = get_effective_flags(item)
        if flags["has_reasoning_step"]:
            effective_reasoning_questions.add(q)
        if flags["perturbation_applied"]:
            effective_perturb_questions.add(q)

    return {
        "name": name,
        "num_results": len(results),
        "question_map": qmap,
        "reasoning_questions": effective_reasoning_questions,
        "perturb_questions": effective_perturb_questions,
        "experiment": experiment,
        "raw": raw,
        "performance": performance,
        "string_based": string_based,
        "llm_evaluated": llm_evaluated,
        "retrieval": retrieval,
    }


def compare_pair(
    baseline_map: Dict[str, Dict[str, Any]],
    perturbed_map: Dict[str, Dict[str, Any]],
    questions: Set[str],
) -> Dict[str, Any]:
    """
    Compare perturbed run against baseline on a fixed question set.
    """
    total = 0

    same_correctness = 0
    flip = 0
    correct_to_wrong = 0
    wrong_to_correct = 0

    baseline_correct = 0
    perturbed_correct = 0

    detailed = []

    for q in sorted(questions):
        if q not in baseline_map or q not in perturbed_map:
            continue

        b = baseline_map[q]
        p = perturbed_map[q]

        b_correct = bool(b.get("is_correct", False))
        p_correct = bool(p.get("is_correct", False))

        total += 1
        baseline_correct += int(b_correct)
        perturbed_correct += int(p_correct)

        if b_correct == p_correct:
            same_correctness += 1
        else:
            flip += 1
            if b_correct and not p_correct:
                correct_to_wrong += 1
            elif (not b_correct) and p_correct:
                wrong_to_correct += 1

        detailed.append({
            "question": q,
            "baseline_correct": b_correct,
            "perturbed_correct": p_correct,
            "baseline_answer": b.get("answer"),
            "perturbed_answer": p.get("answer"),
            "gold_answer": b.get("gold_answer", p.get("gold_answer")),
        })

    return {
        "total": total,
        "baseline_accuracy": 100.0 * baseline_correct / total if total else 0.0,
        "perturbed_accuracy": 100.0 * perturbed_correct / total if total else 0.0,
        "accuracy_drop": (
            (100.0 * baseline_correct / total) - (100.0 * perturbed_correct / total)
        ) if total else 0.0,
        "same_correctness_rate": 100.0 * same_correctness / total if total else 0.0,
        "answer_flip_rate": 100.0 * flip / total if total else 0.0,
        "correct_to_wrong_rate": 100.0 * correct_to_wrong / total if total else 0.0,
        "wrong_to_correct_rate": 100.0 * wrong_to_correct / total if total else 0.0,
        "counts": {
            "same_correctness": same_correctness,
            "flip": flip,
            "correct_to_wrong": correct_to_wrong,
            "wrong_to_correct": wrong_to_correct,
        },
        "details": detailed,
    }


def print_run_summary(run: Dict[str, Any]):
    print("=" * 80)
    print(f"[{run['name']}]")
    print(f"num_results: {run['num_results']}")

    exp = run["experiment"]
    perf = run["performance"]
    s = run["string_based"]
    llm = run["llm_evaluated"]
    ret = run["retrieval"]

    print("experiment:")
    print(f"  experiment_tag: {exp.get('experiment_tag')}")
    print(f"  perturbation_enabled: {exp.get('perturbation_enabled')}")
    print(f"  perturbation_type: {exp.get('perturbation_type')}")
    print(f"  reasoning_sample_count: {exp.get('reasoning_sample_count')}")
    print(f"  reasoning_sample_rate: {exp.get('reasoning_sample_rate')}")
    print(f"  perturbation_applied_count: {exp.get('perturbation_applied_count')}")
    print(f"  perturbation_applied_rate: {exp.get('perturbation_applied_rate')}")
    print(f"  warmup_early_stop_count: {exp.get('warmup_early_stop_count')}")
    print(f"  warmup_early_stop_rate: {exp.get('warmup_early_stop_rate')}")

    print("metrics:")
    print(f"  avg_time: {perf.get('avg_time')}")
    print(f"  avg_rounds: {perf.get('avg_rounds')}")
    print(f"  string_accuracy: {s.get('accuracy')}")
    print(f"  answer_accuracy: {llm.get('answer_accuracy')}")
    print(f"  answer_coverage: {ret.get('answer_coverage')}")

    print("effective samples in this run:")
    print(f"  reasoning-active questions: {len(run['reasoning_questions'])}")
    print(f"  perturbation-applied questions: {len(run['perturb_questions'])}")


def print_compare_summary(name: str, result: Dict[str, Any]):
    print("-" * 80)
    print(f"[{name}]")
    print(f"total compared questions: {result['total']}")
    print(f"baseline_accuracy: {result['baseline_accuracy']:.2f}")
    print(f"perturbed_accuracy: {result['perturbed_accuracy']:.2f}")
    print(f"accuracy_drop: {result['accuracy_drop']:.2f}")
    print(f"same_correctness_rate: {result['same_correctness_rate']:.2f}")
    print(f"answer_flip_rate: {result['answer_flip_rate']:.2f}")
    print(f"correct_to_wrong_rate: {result['correct_to_wrong_rate']:.2f}")
    print(f"wrong_to_correct_rate: {result['wrong_to_correct_rate']:.2f}")
    print("counts:")
    for k, v in result["counts"].items():
        print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True, help="baseline json path")
    parser.add_argument("--answer_wrong", type=str, required=True, help="answer_wrong json path")
    parser.add_argument("--summary_wrong", type=str, required=True, help="summary_wrong json path")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save comparison json"
    )
    args = parser.parse_args()

    baseline_data = load_json(args.baseline)
    answer_wrong_data = load_json(args.answer_wrong)
    summary_wrong_data = load_json(args.summary_wrong)

    baseline = summarize_run("baseline", baseline_data)
    answer_wrong = summarize_run("answer_wrong", answer_wrong_data)
    summary_wrong = summarize_run("summary_wrong", summary_wrong_data)

    print_run_summary(baseline)
    print_run_summary(answer_wrong)
    print_run_summary(summary_wrong)

    # Effective sets
    B = baseline["reasoning_questions"]
    A = answer_wrong["perturb_questions"]
    S = summary_wrong["perturb_questions"]

    # Coverage / overlap
    all_baseline_questions = set(baseline["question_map"].keys())
    all_answer_questions = set(answer_wrong["question_map"].keys())
    all_summary_questions = set(summary_wrong["question_map"].keys())

    common_all = all_baseline_questions & all_answer_questions & all_summary_questions
    strict_intersection = B & A & S

    # Pairwise intersections
    BA = B & A
    BS = B & S
    AS = A & S

    print("=" * 80)
    print("[effective set overlap]")
    print(f"baseline reasoning-active set |B|: {len(B)}")
    print(f"answer_wrong perturb-hit set |A|: {len(A)}")
    print(f"summary_wrong perturb-hit set |S|: {len(S)}")
    print(f"common question universe: {len(common_all)}")
    print(f"|B ∩ A|: {len(BA)}")
    print(f"|B ∩ S|: {len(BS)}")
    print(f"|A ∩ S|: {len(AS)}")
    print(f"|B ∩ A ∩ S|: {len(strict_intersection)}")

    # Comparisons on strict intersection
    answer_vs_baseline_strict = compare_pair(
        baseline["question_map"],
        answer_wrong["question_map"],
        strict_intersection
    )
    summary_vs_baseline_strict = compare_pair(
        baseline["question_map"],
        summary_wrong["question_map"],
        strict_intersection
    )

    print_compare_summary("answer_wrong vs baseline on B∩A∩S", answer_vs_baseline_strict)
    print_compare_summary("summary_wrong vs baseline on B∩A∩S", summary_vs_baseline_strict)

    # Comparisons on pairwise intersections for auxiliary analysis
    answer_vs_baseline_pairwise = compare_pair(
        baseline["question_map"],
        answer_wrong["question_map"],
        BA
    )
    summary_vs_baseline_pairwise = compare_pair(
        baseline["question_map"],
        summary_wrong["question_map"],
        BS
    )

    print_compare_summary("answer_wrong vs baseline on B∩A", answer_vs_baseline_pairwise)
    print_compare_summary("summary_wrong vs baseline on B∩S", summary_vs_baseline_pairwise)

    output = {
        "input_files": {
            "baseline": args.baseline,
            "answer_wrong": args.answer_wrong,
            "summary_wrong": args.summary_wrong,
        },
        "set_stats": {
            "baseline_reasoning_active_count": len(B),
            "answer_wrong_perturb_hit_count": len(A),
            "summary_wrong_perturb_hit_count": len(S),
            "common_question_universe_count": len(common_all),
            "B_intersect_A_count": len(BA),
            "B_intersect_S_count": len(BS),
            "A_intersect_S_count": len(AS),
            "B_intersect_A_intersect_S_count": len(strict_intersection),
            "B_questions": sorted(B),
            "A_questions": sorted(A),
            "S_questions": sorted(S),
            "strict_intersection_questions": sorted(strict_intersection),
        },
        "run_summaries": {
            "baseline": {
                "num_results": baseline["num_results"],
                "experiment": baseline["experiment"],
                "performance": baseline["performance"],
                "string_based": baseline["string_based"],
                "llm_evaluated": baseline["llm_evaluated"],
                "retrieval": baseline["retrieval"],
                "effective_reasoning_count": len(baseline["reasoning_questions"]),
                "effective_perturb_count": len(baseline["perturb_questions"]),
            },
            "answer_wrong": {
                "num_results": answer_wrong["num_results"],
                "experiment": answer_wrong["experiment"],
                "performance": answer_wrong["performance"],
                "string_based": answer_wrong["string_based"],
                "llm_evaluated": answer_wrong["llm_evaluated"],
                "retrieval": answer_wrong["retrieval"],
                "effective_reasoning_count": len(answer_wrong["reasoning_questions"]),
                "effective_perturb_count": len(answer_wrong["perturb_questions"]),
            },
            "summary_wrong": {
                "num_results": summary_wrong["num_results"],
                "experiment": summary_wrong["experiment"],
                "performance": summary_wrong["performance"],
                "string_based": summary_wrong["string_based"],
                "llm_evaluated": summary_wrong["llm_evaluated"],
                "retrieval": summary_wrong["retrieval"],
                "effective_reasoning_count": len(summary_wrong["reasoning_questions"]),
                "effective_perturb_count": len(summary_wrong["perturb_questions"]),
            },
        },
        "comparisons": {
            "answer_wrong_vs_baseline_on_B_intersect_A_intersect_S": answer_vs_baseline_strict,
            "summary_wrong_vs_baseline_on_B_intersect_A_intersect_S": summary_vs_baseline_strict,
            "answer_wrong_vs_baseline_on_B_intersect_A": answer_vs_baseline_pairwise,
            "summary_wrong_vs_baseline_on_B_intersect_S": summary_vs_baseline_pairwise,
        },
    }

    if args.output:
        save_json(output, args.output)
        print("=" * 80)
        print(f"Saved comparison report to: {args.output}")


if __name__ == "__main__":
    main()