import logging
import pdb
import torch
import time
from typing import Dict, List, Tuple, Any
import json
import os
from tqdm import tqdm

from src.utils.utils import (
    normalize_answer,
    evaluate_with_llm,
    string_based_evaluation,
    save_results,
    TOKEN_COST
)
from src.models.logic_rag import LogicRAG
from config.config import RESULT_DIR

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Dictionary of available RAG models
RAG_MODELS = {
    "logic-rag": LogicRAG,
}

# Directory for checkpoints
CHECKPOINT_DIR = os.path.join(RESULT_DIR, "checkpoints")
DEFAULT_CHECKPOINT_INTERVAL = 5


class RAGEvaluator:
    """Evaluator for RAG models."""

    def __init__(
        self,
        model_name: str,
        corpus_path: str,
        max_rounds: int = 3,
        top_k: int = 5,
        eval_top_ks: List[int] = [5, 10],
        checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
        perturbation_enabled: bool = False,
        perturbation_type: str = "none",
        perturbation_step_mode: str = "first_reasoning_step",
        save_history: bool = False,
        experiment_tag: str = "baseline",
        filter_repeats: bool = False,
    ):
        """
        Initialize the evaluator with corpus path and parameters.
        """
        self.model_name = model_name
        self.corpus_path = corpus_path
        self.max_rounds = max_rounds
        self.top_k = top_k
        self.eval_top_ks = sorted(eval_top_ks)
        self.checkpoint_interval = checkpoint_interval

        self.perturbation_enabled = perturbation_enabled
        self.perturbation_type = perturbation_type
        self.perturbation_step_mode = perturbation_step_mode
        self.save_history = save_history
        self.experiment_tag = experiment_tag
        self.filter_repeats = filter_repeats

        os.makedirs(RESULT_DIR, exist_ok=True)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        self._initialize_model()

    def _initialize_model(self):
        """Initialize the specified RAG model."""
        if self.model_name not in RAG_MODELS:
            raise ValueError(f"Unknown RAG model: {self.model_name}")

        model_class = RAG_MODELS[self.model_name]
        self.model = model_class(
            corpus_path=self.corpus_path,
            filter_repeats=self.filter_repeats,
            perturbation_enabled=self.perturbation_enabled,
            perturbation_type=self.perturbation_type,
            perturbation_step_mode=self.perturbation_step_mode,
            save_history=self.save_history,
            experiment_tag=self.experiment_tag,
        )

        self.model.set_top_k(self.top_k)

        if hasattr(self.model, 'set_max_rounds'):
            self.model.set_max_rounds(self.max_rounds)

        logger.info(f"Initialized {self.model_name} model")

    def evaluate_question(self, question: str, gold_answer: str) -> Dict:
        """Evaluate the model on a single question."""
        start_time = time.time()

        answer, contexts, rounds = self.model.answer_question(question)
        elapsed_time = time.time() - start_time

        is_correct = evaluate_with_llm(question, answer, gold_answer)

        metadata = getattr(self.model, "last_run_metadata", {})
        history = getattr(self.model, "last_history", [])
        dependency_analysis = getattr(self.model, "last_dependency_analysis", [])
        retrieval_history = getattr(self.model, "last_retrieval_history", [])
        perturbed_step_info = getattr(self.model, "last_perturbed_step_info", None)

        result = {
            "question": question,
            "gold_answer": gold_answer,
            "answer": answer,
            "contexts": contexts,
            "time": elapsed_time,
            "rounds": rounds,
            "is_correct": is_correct,

            # ===== experiment logs =====
            "history": history if self.save_history else [],
            "metadata": metadata,
            "dependency_analysis": dependency_analysis,
            "retrieval_history": retrieval_history if self.save_history else [],
            "num_history_steps": metadata.get("num_history_steps", 0),
            "num_reasoning_steps": metadata.get("num_reasoning_steps", 0),
            "has_reasoning_step": metadata.get("has_reasoning_step", False),
            "perturbation_enabled": metadata.get("perturbation_enabled", False),
            "perturbation_type": metadata.get("perturbation_type", "none"),
            "perturbation_step_mode": metadata.get("perturbation_step_mode", "first_reasoning_step"),
            "perturbation_applied": metadata.get("perturbation_applied", False),
            "experiment_tag": metadata.get("experiment_tag", self.experiment_tag),
            "early_stop_from_warmup": metadata.get("early_stop_from_warmup", False),
        }

        if perturbed_step_info is not None:
            result["perturbed_step_info"] = perturbed_step_info
            result["original_summary"] = perturbed_step_info.get("original_summary", "")
            result["perturbed_summary"] = perturbed_step_info.get("perturbed_summary", "")
            result["original_answer"] = perturbed_step_info.get("original_answer", "")
            result["perturbed_answer"] = perturbed_step_info.get("perturbed_answer", "")

        return result

    def calculate_retrieval_metrics(self, retrieved_contexts: List[List[str]], answers: List[str]) -> Dict[str, float]:
        """Calculate retrieval-based metrics."""
        total = len(answers)
        found_in_context = 0

        answer_in_top_k = {k: 0 for k in self.eval_top_ks}

        for contexts, answer in zip(retrieved_contexts, answers):
            normalized_answer = normalize_answer(answer)

            for i, context in enumerate(contexts):
                if normalized_answer in normalize_answer(context):
                    found_in_context += 1
                    for k in self.eval_top_ks:
                        if i < k:
                            answer_in_top_k[k] += 1
                    break

        result = {
            "answer_found_in_context": found_in_context / total,
            "total_questions": total
        }

        for k in self.eval_top_ks:
            result[f"answer_in_top{k}"] = answer_in_top_k[k] / total

        return result

    def _get_checkpoint_path(self, output_file: str) -> str:
        """Generate a checkpoint path based on the output file."""
        base_name = os.path.basename(output_file)
        name, ext = os.path.splitext(base_name)
        return os.path.join(CHECKPOINT_DIR, f"{name}_checkpoint{ext}")

    def _save_checkpoint(self, results: List[Dict], metrics: Dict, processed_count: int, output_file: str):
        """Save a checkpoint of current evaluation progress."""
        if results[-1]["answer"] == "":
            print("\n\n\033[91mLost connection to the LLM API, skipping checkpoint save\033[0m\n\n")
            exit(1)

        checkpoint = {
            "model": self.model_name,
            "metrics": metrics,
            "results": results,
            "processed_count": processed_count,
            "experiment": {
                "experiment_tag": self.experiment_tag,
                "perturbation_enabled": self.perturbation_enabled,
                "perturbation_type": self.perturbation_type,
                "perturbation_step_mode": self.perturbation_step_mode,
                "save_history": self.save_history,
                "filter_repeats": self.filter_repeats,
            },
            "token_cost": {
                "prompt": TOKEN_COST["prompt"],
                "completion": TOKEN_COST["completion"]
            }
        }

        checkpoint_path = self._get_checkpoint_path(output_file)

        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        logger.info(f"Checkpoint saved: {processed_count} questions processed")

    def _load_checkpoint(self, output_file: str) -> Tuple[List[Dict], Dict, int]:
        """Load evaluation checkpoint if it exists."""
        checkpoint_path = self._get_checkpoint_path(output_file)

        if not os.path.exists(checkpoint_path):
            return [], {}, 0

        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)

            if "token_cost" in checkpoint:
                TOKEN_COST["prompt"] = checkpoint["token_cost"]["prompt"]
                TOKEN_COST["completion"] = checkpoint["token_cost"]["completion"]
                logger.info(
                    f"Restored token costs - Prompt: {TOKEN_COST['prompt']}, Completion: {TOKEN_COST['completion']}")

            logger.info(f"Loaded checkpoint: {checkpoint['processed_count']} questions already processed")
            return checkpoint["results"], checkpoint["metrics"], checkpoint["processed_count"]
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return [], {}, 0

    def run_single_model_evaluation(self, eval_data: List[Dict], output_file: str = "evaluation_results.json"):
        """Run evaluation of a single model on the given evaluation data."""
        results, metrics, processed_count = self._load_checkpoint(output_file)

        if processed_count > 0:
            eval_data = eval_data[processed_count:]
            logger.info(
                f"Resuming from checkpoint: {processed_count} questions already processed, {len(eval_data)} remaining")

        if not eval_data:
            logger.info("All questions were already processed in previous run.")
            output_path = os.path.join(RESULT_DIR, output_file)
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as f:
                    evaluation_summary = json.load(f)
                return evaluation_summary

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

                # ===== experiment metrics =====
                "reasoning_sample_count": 0,
                "perturbation_applied_count": 0,
                "warmup_early_stop_count": 0,
            }

            for k in self.eval_top_ks:
                metrics[f"top{k}_hits"] = 0
        else:
            logger.info(
                f"Restored token costs - Prompt: {TOKEN_COST['prompt']}, Completion: {TOKEN_COST['completion']}")

        total_questions = len(eval_data) + processed_count

        for i, item in enumerate(tqdm(eval_data, desc=f"Evaluating {self.model_name}")):
            question = item['question']
            gold_answer = item['answer']

            result = self.evaluate_question(
                question=question,
                gold_answer=gold_answer
            )
            results.append(result)

            metrics["total_time"] += result["time"]
            normalized_gold = normalize_answer(gold_answer)

            string_metrics = string_based_evaluation(
                result["answer"],
                gold_answer
            )
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

            if "rounds" in result:
                metrics["total_rounds"] += result["rounds"]

            if result["is_correct"]:
                metrics["answer_accuracy"] += 1

            # ===== experiment counters =====
            if result.get("has_reasoning_step", False):
                metrics["reasoning_sample_count"] += 1
            if result.get("perturbation_applied", False):
                metrics["perturbation_applied_count"] += 1
            if result.get("early_stop_from_warmup", False):
                metrics["warmup_early_stop_count"] += 1

            current_count = processed_count + i + 1
            if (current_count % self.checkpoint_interval == 0) or (i == len(eval_data) - 1):
                self._save_checkpoint(results, metrics, current_count, output_file)

        avg_metrics = {
            "avg_time": metrics["total_time"] / total_questions,
            "answer_coverage": metrics["answer_coverage"] / total_questions * 100,
            "answer_accuracy": metrics["answer_accuracy"] / total_questions * 100,
            "string_accuracy": metrics["string_accuracy"] / total_questions * 100,
            "string_precision": metrics["string_precision"] / total_questions * 100,
            "string_recall": metrics["string_recall"] / total_questions * 100,
            "avg_rounds": metrics["total_rounds"] / total_questions,
            "reasoning_sample_rate": metrics["reasoning_sample_count"] / total_questions * 100,
            "perturbation_applied_rate": metrics["perturbation_applied_count"] / total_questions * 100,
            "warmup_early_stop_rate": metrics["warmup_early_stop_count"] / total_questions * 100,
        }

        for k in self.eval_top_ks:
            avg_metrics[f"top{k}_coverage"] = metrics[f"top{k}_hits"] / total_questions * 100

        organized_metrics = {
            "performance": {
                "avg_time": avg_metrics["avg_time"],
                "avg_rounds": avg_metrics["avg_rounds"],
            },
            "string_based": {
                "accuracy": avg_metrics["string_accuracy"],
                "precision": avg_metrics["string_precision"],
                "recall": avg_metrics["string_recall"]
            },
            "llm_evaluated": {
                "answer_accuracy": avg_metrics["answer_accuracy"]
            },
            "retrieval": {
                "answer_coverage": avg_metrics["answer_coverage"]
            },
            "experiment": {
                "experiment_tag": self.experiment_tag,
                "perturbation_enabled": self.perturbation_enabled,
                "perturbation_type": self.perturbation_type,
                "perturbation_step_mode": self.perturbation_step_mode,
                "save_history": self.save_history,
                "filter_repeats": self.filter_repeats,
                "reasoning_sample_count": metrics["reasoning_sample_count"],
                "reasoning_sample_rate": avg_metrics["reasoning_sample_rate"],
                "perturbation_applied_count": metrics["perturbation_applied_count"],
                "perturbation_applied_rate": avg_metrics["perturbation_applied_rate"],
                "warmup_early_stop_count": metrics["warmup_early_stop_count"],
                "warmup_early_stop_rate": avg_metrics["warmup_early_stop_rate"],
            }
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

        organized_metrics["raw"] = metrics

        evaluation_summary = {
            "model": self.model_name,
            "metrics": organized_metrics,
            "results": results
        }

        save_results(
            results=evaluation_summary,
            output_file=output_file,
            results_dir=RESULT_DIR
        )

        logger.info(f"\nEvaluation Summary for {self.model_name}:")
        logger.info(f"Average time per question: {avg_metrics['avg_time']:.2f} seconds")
        logger.info(f"Average rounds per question: {avg_metrics['avg_rounds']:.2f}")
        logger.info(f"Average prompt tokens per question: {organized_metrics['performance']['avg_prompt_tokens']:.2f}")
        logger.info(f"Average completion tokens per question: {organized_metrics['performance']['avg_completion_tokens']:.2f}")
        logger.info(f"Average total tokens per question: {organized_metrics['performance']['avg_total_tokens']:.2f}")

        logger.info("\n1. String-based Metrics:")
        logger.info(f"  • Accuracy: {avg_metrics['string_accuracy']:.2f}%")
        logger.info(f"  • Precision: {avg_metrics['string_precision']:.2f}%")
        logger.info(f"  • Recall: {avg_metrics['string_recall']:.2f}%")

        logger.info("\n2. LLM Evaluated Metrics:")
        logger.info(f"  • Answer Accuracy: {avg_metrics['answer_accuracy']:.2f}%")

        logger.info("\n3. Retrieval Performance:")
        logger.info(f"  • Answer Coverage: {avg_metrics['answer_coverage']:.2f}%")

        for k in self.eval_top_ks:
            logger.info(f"  • Top-{k} Coverage: {avg_metrics[f'top{k}_coverage']:.2f}%")

        logger.info("\n4. Experiment Metrics:")
        logger.info(f"  • Reasoning Sample Rate: {avg_metrics['reasoning_sample_rate']:.2f}%")
        logger.info(f"  • Perturbation Applied Rate: {avg_metrics['perturbation_applied_rate']:.2f}%")
        logger.info(f"  • Warm-up Early Stop Rate: {avg_metrics['warmup_early_stop_rate']:.2f}%")

        return evaluation_summary