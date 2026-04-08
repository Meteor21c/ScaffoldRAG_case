#!/usr/bin/env python
import argparse
import json
import logging
import importlib
from typing import Dict, List, Type

from src.evaluation.evaluation import RAGEvaluator
from src.models.base_rag import BaseRAG
from src.models.logic_rag import LogicRAG


# Configure logging
logging.basicConfig(level=logging.WARNING,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# Dictionary of available RAG models
RAG_MODELS = {
    "logic-rag": LogicRAG,
}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run RAG models')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='dataset/hotpotqa.json',
                        help='Path to the dataset file')
    parser.add_argument('--corpus', type=str, default='dataset/hotpotqa_corpus.json',
                        help='Path to the corpus file')
    parser.add_argument('--limit', type=int, default=20,
                        help='Number of questions to evaluate (default: 20)')

    # RAG configuration
    parser.add_argument('--max-rounds', type=int, default=3,
                        help='Maximum number of agent rounds')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of top contexts to retrieve')
    parser.add_argument('--eval-top-ks', type=int, nargs='+', default=[5, 10],
                        help='List of k values for top-k accuracy evaluation (default: [5, 10])')

    # Single question (optional)
    parser.add_argument('--question', type=str,
                        help='Optional: Single question to answer')

    # RAG model selection
    parser.add_argument('--model', type=str, choices=list(RAG_MODELS.keys()),
                        default='logic-rag',
                        help='Which RAG model to use')

    # Evaluation options
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                        help='Output file name')

    # Checkpoint options
    parser.add_argument('--checkpoint-interval', type=int, default=5,
                        help='Number of questions to process before saving a checkpoint (default: 5)')

    # ===== Experiment options =====
    parser.add_argument(
        '--perturbation-enabled',
        action='store_true',
        help='Whether to inject perturbation into the first reasoning-step memory node.'
    )
    parser.add_argument(
        '--perturbation-type',
        type=str,
        default='none',
        choices=['none', 'answer_wrong', 'summary_wrong'],
        help='Type of perturbation to apply.'
    )
    parser.add_argument(
        '--perturbation-step-mode',
        type=str,
        default='first_reasoning_step',
        choices=['first_reasoning_step'],
        help='Which reasoning step to perturb.'
    )
    parser.add_argument(
        '--save-history',
        action='store_true',
        help='Whether to save full reasoning history in evaluation outputs.'
    )
    parser.add_argument(
        '--experiment-tag',
        type=str,
        default='baseline',
        help='Experiment tag to identify this run.'
    )
    parser.add_argument(
        '--filter-repeats',
        action='store_true',
        help='Filter repeated chunks across retrieval rounds.'
    )

    return parser.parse_args()


def load_evaluation_data(dataset_path: str, limit: int) -> List[Dict]:
    """Load and limit the evaluation dataset."""
    try:
        with open(dataset_path, 'r') as f:
            eval_data = json.load(f)

        # Limit the number of questions if needed
        if limit and limit > 0:
            eval_data = eval_data[:limit]

        return eval_data
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return []


def create_rag_model(
    model_name: str,
    corpus_path: str,
    max_rounds: int = 3,
    top_k: int = 5,
    perturbation_enabled: bool = False,
    perturbation_type: str = "none",
    perturbation_step_mode: str = "first_reasoning_step",
    save_history: bool = False,
    experiment_tag: str = "baseline",
    filter_repeats: bool = False,
) -> BaseRAG:
    """Create and configure a RAG model instance."""
    if model_name not in RAG_MODELS:
        raise ValueError(f"Unknown RAG model: {model_name}")

    model_class = RAG_MODELS[model_name]
    model = model_class(
        corpus_path=corpus_path,
        filter_repeats=filter_repeats,
        perturbation_enabled=perturbation_enabled,
        perturbation_type=perturbation_type,
        perturbation_step_mode=perturbation_step_mode,
        save_history=save_history,
        experiment_tag=experiment_tag
    )

    model.set_top_k(top_k)

    if hasattr(model, 'set_max_rounds'):
        model.set_max_rounds(max_rounds)

    return model


def run_single_question(
    model_name: str,
    question: str,
    corpus_path: str,
    max_rounds: int,
    top_k: int,
    perturbation_enabled: bool = False,
    perturbation_type: str = "none",
    perturbation_step_mode: str = "first_reasoning_step",
    save_history: bool = False,
    experiment_tag: str = "baseline",
    filter_repeats: bool = False,
):
    """Run a single question through the specified RAG model."""
    model = create_rag_model(
        model_name=model_name,
        corpus_path=corpus_path,
        max_rounds=max_rounds,
        top_k=top_k,
        perturbation_enabled=perturbation_enabled,
        perturbation_type=perturbation_type,
        perturbation_step_mode=perturbation_step_mode,
        save_history=save_history,
        experiment_tag=experiment_tag,
        filter_repeats=filter_repeats,
    )

    logger.info(f"\nQuestion: {question}")
    logger.info(f"Using {model_name} RAG model")

    answer, contexts, rounds = model.answer_question(question)
    logger.info(f"\nAnswer: {answer}")
    logger.info(f"Retrieved in {rounds} rounds")

    logger.info("\nContexts used:")
    for i, ctx in enumerate(contexts):
        logger.info(f"{i+1}. {ctx[:100]}...")

    if hasattr(model, "last_run_metadata"):
        logger.info(f"Run metadata: {json.dumps(model.last_run_metadata, ensure_ascii=False, indent=2)}")

    return answer, contexts


def main():
    """Main function to run the RAG model."""
    args = parse_arguments()

    if args.question:
        run_single_question(
            model_name=args.model,
            question=args.question,
            corpus_path=args.corpus,
            max_rounds=args.max_rounds,
            top_k=args.top_k,
            perturbation_enabled=args.perturbation_enabled,
            perturbation_type=args.perturbation_type,
            perturbation_step_mode=args.perturbation_step_mode,
            save_history=args.save_history,
            experiment_tag=args.experiment_tag,
            filter_repeats=args.filter_repeats,
        )
        return

    logger.info(f"Starting evaluation of {args.model} RAG model")
    logger.info(f"Max rounds: {args.max_rounds}, Top-k: {args.top_k}")
    logger.info(f"Evaluating top-k accuracy for k values: {args.eval_top_ks}")
    logger.info(f"Checkpoint interval: {args.checkpoint_interval} questions")
    logger.info(f"Experiment tag: {args.experiment_tag}")
    logger.info(f"Perturbation enabled: {args.perturbation_enabled}")
    logger.info(f"Perturbation type: {args.perturbation_type}")

    eval_data = load_evaluation_data(args.dataset, args.limit)
    if not eval_data:
        logger.error("No evaluation data available. Exiting.")
        return

    logger.info(f"Loaded {len(eval_data)} questions for evaluation")

    evaluator = RAGEvaluator(
        model_name=args.model,
        corpus_path=args.corpus,
        max_rounds=args.max_rounds,
        top_k=args.top_k,
        eval_top_ks=args.eval_top_ks,
        checkpoint_interval=args.checkpoint_interval,
        perturbation_enabled=args.perturbation_enabled,
        perturbation_type=args.perturbation_type,
        perturbation_step_mode=args.perturbation_step_mode,
        save_history=args.save_history,
        experiment_tag=args.experiment_tag,
        filter_repeats=args.filter_repeats,
    )

    evaluation_summary = evaluator.run_single_model_evaluation(
        eval_data=eval_data,
        output_file=args.output
    )

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()