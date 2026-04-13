#!/usr/bin/env python
import argparse
import json
import logging
from typing import Dict, List

from src.evaluation.rolling_memory_scheme_b_evaluation import RollingMemorySchemeBEvaluator
from src.models.logic_rag_rolling_memory_scheme_b import LogicRAGRollingMemoryBaseline

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run Scheme-B rolling-memory baseline')

    parser.add_argument('--dataset', type=str, default='dataset/hotpotqa.json',
                        help='Path to the dataset file')
    parser.add_argument('--corpus', type=str, default='dataset/hotpotqa_corpus.json',
                        help='Path to the corpus file')
    parser.add_argument('--limit', type=int, default=20,
                        help='Number of questions to evaluate (default: 20)')

    parser.add_argument('--max-rounds', type=int, default=3,
                        help='Maximum number of agent rounds')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of top contexts to retrieve')
    parser.add_argument('--eval-top-ks', type=int, nargs='+', default=[5, 10],
                        help='List of k values for top-k accuracy evaluation (default: [5, 10])')

    parser.add_argument('--question', type=str,
                        help='Optional: Single question to answer')

    parser.add_argument('--output', type=str, default='rolling_memory_scheme_b_results.json',
                        help='Output file name')
    parser.add_argument('--checkpoint-interval', type=int, default=5,
                        help='Number of questions to process before saving a checkpoint (default: 5)')

    parser.add_argument('--hallucination-enabled', action='store_true',
                        help='Inject the fixed summary_wrong content at the first sub-query.')
    parser.add_argument('--hallucination-type', type=str, default='none',
                        choices=['none', 'summary_wrong'],
                        help='Hallucination type for Scheme B.')
    parser.add_argument('--save-history', action='store_true',
                        help='Whether to save retrieval history and memory trace in outputs.')
    parser.add_argument('--experiment-tag', type=str, default='rolling_memory_baseline',
                        help='Experiment tag to identify this run.')
    parser.add_argument('--filter-repeats', action='store_true',
                        help='Filter repeated chunks across retrieval rounds.')

    return parser.parse_args()


def load_evaluation_data(dataset_path: str, limit: int) -> List[Dict]:
    try:
        with open(dataset_path, 'r') as f:
            eval_data = json.load(f)
        if limit and limit > 0:
            eval_data = eval_data[:limit]
        return eval_data
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return []


def create_model(
    corpus_path: str,
    max_rounds: int = 3,
    top_k: int = 5,
    hallucination_enabled: bool = False,
    hallucination_type: str = "none",
    save_history: bool = False,
    experiment_tag: str = "rolling_memory_baseline",
    filter_repeats: bool = False,
):
    model = LogicRAGRollingMemoryBaseline(
        corpus_path=corpus_path,
        filter_repeats=filter_repeats,
        hallucination_enabled=hallucination_enabled,
        hallucination_type=hallucination_type,
        save_history=save_history,
        experiment_tag=experiment_tag,
    )
    model.set_top_k(top_k)
    model.set_max_rounds(max_rounds)
    return model


def run_single_question(
    question: str,
    corpus_path: str,
    max_rounds: int,
    top_k: int,
    hallucination_enabled: bool = False,
    hallucination_type: str = "none",
    save_history: bool = False,
    experiment_tag: str = "rolling_memory_baseline",
    filter_repeats: bool = False,
):
    model = create_model(
        corpus_path=corpus_path,
        max_rounds=max_rounds,
        top_k=top_k,
        hallucination_enabled=hallucination_enabled,
        hallucination_type=hallucination_type,
        save_history=save_history,
        experiment_tag=experiment_tag,
        filter_repeats=filter_repeats,
    )

    answer, contexts, rounds = model.answer_question(question)
    logger.info(f"Answer: {answer}")
    logger.info(f"Retrieved in {rounds} rounds")

    if hasattr(model, "last_run_metadata"):
        logger.info(f"Run metadata: {json.dumps(model.last_run_metadata, ensure_ascii=False, indent=2)}")

    return answer, contexts


def main():
    args = parse_arguments()

    if args.question:
        run_single_question(
            question=args.question,
            corpus_path=args.corpus,
            max_rounds=args.max_rounds,
            top_k=args.top_k,
            hallucination_enabled=args.hallucination_enabled,
            hallucination_type=args.hallucination_type,
            save_history=args.save_history,
            experiment_tag=args.experiment_tag,
            filter_repeats=args.filter_repeats,
        )
        return

    eval_data = load_evaluation_data(args.dataset, args.limit)
    if not eval_data:
        logger.error("No evaluation data available. Exiting.")
        return

    evaluator = RollingMemorySchemeBEvaluator(
        corpus_path=args.corpus,
        max_rounds=args.max_rounds,
        top_k=args.top_k,
        eval_top_ks=args.eval_top_ks,
        checkpoint_interval=args.checkpoint_interval,
        hallucination_enabled=args.hallucination_enabled,
        hallucination_type=args.hallucination_type,
        save_history=args.save_history,
        experiment_tag=args.experiment_tag,
        filter_repeats=args.filter_repeats,
    )

    evaluator.run_evaluation(
        eval_data=eval_data,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
