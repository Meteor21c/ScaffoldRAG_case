import json
import logging
import os
import time
from typing import Dict, List, Tuple

from tqdm import tqdm

from config.config import RESULT_DIR
from src.models.logic_rag_rolling_memory_scheme_b import LogicRAGRollingMemoryBaseline
from src.utils.utils import (
    TOKEN_COST,
    evaluate_with_llm,
    normalize_answer,
    save_results,
    string_based_evaluation,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = os.path.join(RESULT_DIR, "checkpoints")
DEFAULT_CHECKPOINT_INTERVAL = 5


class RollingMemorySchemeBEvaluator:
    """
    Separate evaluator for Scheme B.
    Keeps most evaluation parameters aligned with ScaffoldRAG,
    but only tracks:
    - baseline run
    - one hallucination run (summary_wrong injected at first sub-query)
    """

    def __init__(
        self,
        corpus_path: str,
        max_rounds: int = 3,
        top_k: int = 5,
        eval_top_ks: List[int] = [5, 10],
        checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
        hallucination_enabled: bool = False,
        hallucination_type: str = "none",
        save_history: bool = False,
        experiment_tag: str = "rolling_memory_baseline",
        filter_repeats: bool = False,
    ):
        self.corpus_path = corpus_path
        self.max_rounds = max_rounds
        self.top_k = top_k
        self.eval_top_ks = sorted(eval_top_ks)
        self.checkpoint_interval = checkpoint_interval

        self.hallucination_enabled = hallucination_enabled
        self.hallucination_type = hallucination_type
        self.save_history = save_history
        self.experiment_tag = experiment_tag
        self.filter_repeats = filter_repeats

        os.makedirs(RESULT_DIR, exist_ok=True)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        self._initialize_model()

    def _initialize_model(self):
        self.model = LogicRAGRollingMemoryBaseline(
            corpus_path=self.corpus_path,
            filter_repeats=self.filter_repeats,
            hallucination_enabled=self.hallucination_enabled,
            hallucination_type=self.hallucination_type,
            save_history=self.save_history,
            experiment_tag=self.experiment_tag,
        )

        self.model.set_top_k(self.top_k)
        if hasattr(self.model, "set_max_rounds"):
            self.model.set_max_rounds(self.max_rounds)

    def evaluate_question(self, question: str, gold_answer: str) -> Dict:
        start_time = time.time()
        answer, contexts, rounds = self.model.answer_question(question)
        elapsed_time = time.time() - start_time

        is_correct = evaluate_with_llm(question, answer, gold_answer)

        metadata = getattr(self.model, "last_run_metadata", {})
        dependency_analysis = getattr(self.model, "last_dependency_analysis", [])
        retrieval_history = getattr(self.model, "last_retrieval_history", [])
        memory_trace = getattr(self.model, "last_memory_trace", [])
        hallucination_info = getattr(self.model, "last_hallucination_info", None)

        result = {
            "question": question,
            "gold_answer": gold_answer,
            "answer": answer,
            "contexts": contexts,
            "time": elapsed_time,
            "rounds": rounds,
            "is_correct": is_correct,
            "metadata": metadata,
            "dependency_analysis": dependency_analysis,
            "retrieval_history": retrieval_history if self.save_history else [],
            "memory_trace": memory_trace if self.save_history else [],
            "has_reasoning_step": metadata.get("has_reasoning_step", False),
            "hallucination_enabled": metadata.get("hallucination_enabled", False),
            "hallucination_type": metadata.get("hallucination_type", "none"),
            "hallucination_applied": metadata.get("hallucination_applied", False),
            "experiment_tag": metadata.get("experiment_tag", self.experiment_tag),
            "early_stop_from_warmup": metadata.get("early_stop_from_warmup", False),
            "memory_strategy": metadata.get("memory_strategy", "rolling_memory_scheme_b"),
        }

        if hallucination_info is not None:
            result["hallucination_info"] = hallucination_info
            result["injected_summary"] = hallucination_info.get("injected_summary", "")

        return result

    def _get_checkpoint_path(self, output_file: str) -> str:
        base_name = os.path.basename(output_file)
        name, ext = os.path.splitext(base_name)
        return os.path.join(CHECKPOINT_DIR, f"{name}_checkpoint{ext}")

    def _save_checkpoint(self, results: List[Dict], metrics: Dict, processed_count: int, output_file: str):
        if results[-1]["answer"] == "":
            print("\n\n\033[91mLost connection to the LLM API, skipping checkpoint save\033[0m\n\n")
            raise SystemExit(1)

        checkpoint = {
            "model": "logic-rag-rolling-memory-scheme-b",
            "metrics": metrics,
            "results": results,
            "processed_count": processed_count,
            "experiment": {
                "memory_strategy": "rolling_memory_scheme_b",
                "experiment_tag": self.experiment_tag,
                "hallucination_enabled": self.hallucination_enabled,
                "hallucination_type": self.hallucination_type,
                "save_history": self.save_history,
                "filter_repeats": self.filter_repeats,
            },
            "token_cost": {
                "prompt": TOKEN_COST["prompt"],
                "completion": TOKEN_COST["completion"],
            },
        }

        checkpoint_path = self._get_checkpoint_path(output_file)
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self, output_file: str) -> Tuple[List[Dict], Dict, int]:
        checkpoint_path = self._get_checkpoint_path(output_file)
        if not os.path.exists(checkpoint_path):
            return [], {}, 0

        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)

            if "token_cost" in checkpoint:
                TOKEN_COST["prompt"] = checkpoint["token_cost"]["prompt"]
                TOKEN_COST["completion"] = checkpoint["token_cost"]["completion"]

            return checkpoint["results"], checkpoint["metrics"], checkpoint["processed_count"]
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return [], {}, 0

    def run_evaluation(self, eval_data: List[Dict], output_file: str = "rolling_memory_scheme_b_results.json"):
        results, metrics, processed_count = self._load_checkpoint(output_file)

        if processed_count > 0:
            eval_data = eval_data[processed_count:]

        if not eval_data:
            output_path = os.path.join(RESULT_DIR, output_file)
            if os.path.exists(output_path):
                with open(output_path, "r", encoding="utf-8") as f:
                    return json.load(f)

        if not metrics:
            TOKEN_COST["prompt"] = 0
            TOKEN_COST["completion"] = 0

            metrics = {
                "total_time": 0,
                "answer_coverage": 0,
                "answer_accuracy": 0,
                "string_accuracy": 0,
                "string_precision": 0,
                "string_recall": 0,
                "total_rounds": 0,
                "reasoning_sample_count": 0,
                "hallucination_applied_count": 0,
                "warmup_early_stop_count": 0,
            }
            for k in self.eval_top_ks:
                metrics[f"top{k}_hits"] = 0

        total_questions = len(eval_data) + processed_count

        for i, item in enumerate(tqdm(eval_data, desc="Evaluating rolling-memory scheme B")):
            question = item["question"]
            gold_answer = item["answer"]

            result = self.evaluate_question(question=question, gold_answer=gold_answer)
            results.append(result)

            metrics["total_time"] += result["time"]
            normalized_gold = normalize_answer(gold_answer)

            string_metrics = string_based_evaluation(result["answer"], gold_answer)
            metrics["string_accuracy"] += string_metrics["accuracy"]
            metrics["string_precision"] += string_metrics["precision"]
            metrics["string_recall"] += string_metrics["recall"]

            for j, ctx in enumerate(result["contexts"]):
                if normalized_gold in normalize_answer(ctx):
                    metrics["answer_coverage"] += 1
                    for k in self.eval_top_ks:
                        if j < k:
                            metrics[f"top{k}_hits"] += 1
                    break

            metrics["total_rounds"] += result.get("rounds", 0)
            if result["is_correct"]:
                metrics["answer_accuracy"] += 1
            if result.get("has_reasoning_step", False):
                metrics["reasoning_sample_count"] += 1
            if result.get("hallucination_applied", False):
                metrics["hallucination_applied_count"] += 1
            if result.get("early_stop_from_warmup", False):
                metrics["warmup_early_stop_count"] += 1

            current_count = processed_count + i + 1
            if (current_count % self.checkpoint_interval == 0) or (i == len(eval_data) - 1):
                self._save_checkpoint(results, metrics, current_count, output_file)

        avg_metrics = {
            "avg_time": metrics["total_time"] / total_questions if total_questions else 0,
            "answer_coverage": metrics["answer_coverage"] / total_questions * 100 if total_questions else 0,
            "answer_accuracy": metrics["answer_accuracy"] / total_questions * 100 if total_questions else 0,
            "string_accuracy": metrics["string_accuracy"] / total_questions * 100 if total_questions else 0,
            "string_precision": metrics["string_precision"] / total_questions * 100 if total_questions else 0,
            "string_recall": metrics["string_recall"] / total_questions * 100 if total_questions else 0,
            "avg_rounds": metrics["total_rounds"] / total_questions if total_questions else 0,
            "reasoning_sample_rate": metrics["reasoning_sample_count"] / total_questions * 100 if total_questions else 0,
            "hallucination_applied_rate": metrics["hallucination_applied_count"] / total_questions * 100 if total_questions else 0,
            "warmup_early_stop_rate": metrics["warmup_early_stop_count"] / total_questions * 100 if total_questions else 0,
        }
        for k in self.eval_top_ks:
            avg_metrics[f"top{k}_coverage"] = metrics[f"top{k}_hits"] / total_questions * 100 if total_questions else 0

        organized_metrics = {
            "performance": {
                "avg_time": avg_metrics["avg_time"],
                "avg_rounds": avg_metrics["avg_rounds"],
            },
            "string_based": {
                "accuracy": avg_metrics["string_accuracy"],
                "precision": avg_metrics["string_precision"],
                "recall": avg_metrics["string_recall"],
            },
            "llm_evaluated": {
                "answer_accuracy": avg_metrics["answer_accuracy"],
            },
            "retrieval": {
                "answer_coverage": avg_metrics["answer_coverage"],
            },
            "experiment": {
                "memory_strategy": "rolling_memory_scheme_b",
                "experiment_tag": self.experiment_tag,
                "hallucination_enabled": self.hallucination_enabled,
                "hallucination_type": self.hallucination_type,
                "save_history": self.save_history,
                "filter_repeats": self.filter_repeats,
                "reasoning_sample_count": metrics["reasoning_sample_count"],
                "reasoning_sample_rate": avg_metrics["reasoning_sample_rate"],
                "hallucination_applied_count": metrics["hallucination_applied_count"],
                "hallucination_applied_rate": avg_metrics["hallucination_applied_rate"],
                "warmup_early_stop_count": metrics["warmup_early_stop_count"],
                "warmup_early_stop_rate": avg_metrics["warmup_early_stop_rate"],
            },
            "raw": metrics,
        }

        if total_questions > 0:
            organized_metrics["performance"]["avg_prompt_tokens"] = TOKEN_COST["prompt"] / total_questions
            organized_metrics["performance"]["avg_completion_tokens"] = TOKEN_COST["completion"] / total_questions
            organized_metrics["performance"]["avg_total_tokens"] = (
                TOKEN_COST["prompt"] + TOKEN_COST["completion"]
            ) / total_questions
        else:
            organized_metrics["performance"]["avg_prompt_tokens"] = 0
            organized_metrics["performance"]["avg_completion_tokens"] = 0
            organized_metrics["performance"]["avg_total_tokens"] = 0

        for k in self.eval_top_ks:
            organized_metrics["retrieval"][f"top{k}_coverage"] = avg_metrics[f"top{k}_coverage"]

        evaluation_summary = {
            "model": "logic-rag-rolling-memory-scheme-b",
            "metrics": organized_metrics,
            "results": results,
        }

        save_results(
            results=evaluation_summary,
            output_file=output_file,
            results_dir=RESULT_DIR,
        )
        return evaluation_summary
