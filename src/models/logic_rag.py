import json
import logging
import pdb
import time
import copy
from typing import List, Dict, Tuple, Any
from src.models.base_rag import BaseRAG
from src.utils.utils import get_response_with_retry, fix_json_response
from colorama import Fore, Style, init

# Initialize colorama
init()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class LogicRAG(BaseRAG):

    def __init__(
            self,
            corpus_path: str = None,
            cache_dir: str = "./cache",
            filter_repeats: bool = False,
            perturbation_enabled: bool = False,
            perturbation_type: str = "none",
            perturbation_step_mode: str = "first_reasoning_step",
            save_history: bool = False,
            experiment_tag: str = "baseline",
    ):
        """Initialize the LogicRAG system."""
        super().__init__(corpus_path, cache_dir)
        self.max_rounds = 3
        self.MODEL_NAME = "LogicRAG"
        self.filter_repeats = filter_repeats

        # ===== Experiment config =====
        self.perturbation_enabled = perturbation_enabled
        self.perturbation_type = perturbation_type
        self.perturbation_step_mode = perturbation_step_mode
        self.save_history = save_history
        self.experiment_tag = experiment_tag

        self.answer_wrong_template = (
            "This step incorrectly identifies the answer as a different entity or event."
        )
        self.summary_wrong_template = (
            "The evidence is incorrectly summarized around a misleading conclusion rather than the actual key fact."
        )

        # ===== Runtime logs =====
        self.last_dependency_analysis = []
        self.last_history = []
        self.last_run_metadata = {}
        self.last_perturbed_step_info = None
        self.last_retrieval_history = []

    def set_max_rounds(self, max_rounds: int):
        """Set the maximum number of retrieval rounds."""
        self.max_rounds = max_rounds

    def process_step(self, global_question: str, sub_query: str, contexts: List[str]) -> Dict[str, str]:
        """
        处理单个推理步骤：对检索内容进行总结，并尝试回答子问题。

        Args:
            global_question: 用户最原始的问题
            sub_query: 当前步骤的子查询
            contexts: 当前步骤检索到的 top-k 文档

        Returns:
            Dict containing 'summary' and 'answer'
        """
        context_text = "\n".join(contexts)

        prompt = f"""
        You are an intelligent reasoning agent. 
        Global Goal: Answer the question "{global_question}"
        Current Step: Investigate the sub-query "{sub_query}"

        Retrieved Information:
        {context_text}

        Task:
        1. Summarize the retrieved information relevant to both the Global Goal and Current Step. Be concise but preserve key entities and facts.
        2. Attempt to give a direct answer to the "Current Step" sub-query based ONLY on the retrieved information.
        3.Format your response as a JSON object with two keys:
        - "summary": "The concise summary..."
        - "answer": "The direct answer to the sub-query..."
        4.Do not output any other content.
        """

        try:
            response = get_response_with_retry(prompt)
            result = fix_json_response(response)

            if not result or "summary" not in result:
                return {
                    "summary": context_text[:500] + "...",
                    "answer": "Could not parse answer."
                }
            return result

        except Exception as e:
            logger.error(f"{Fore.RED}Error in process_step: {e}{Style.RESET_ALL}")
            return {
                "summary": context_text,
                "answer": "Error during processing."
            }

    def warm_up_analysis(self, question: str, history: List[Dict]) -> Dict:
        """
        Warm-up analysis: Analyze if the initial retrieval (Step 0) is sufficient.
        """
        history_text = self._format_history_for_llm(history)
        try:
            prompt = f"""

            Global Question: {question}

            Current Knowledge (from Initial Retrieval):
            {history_text}

Based on the Current Information provided, analyze:
1. Can the global question be answered completely ONLY with knowledge given above? (Yes/No)
2. What specific information is missing, if any?
3. What specific question should we ask to find the missing information?
4. Summarize our current understanding based on available information.
5. What are the key dependencies needed to answer this question?
6. Why is information missing? (max 20 words)

Please format your response as a JSON object with these keys:
- "can_answer": boolean
- "missing_info": string
- "subquery": string
- "current_understanding": string
- "dependencies": list of strings (key information dependencies)
- "missing_reason": string (brief explanation why info is missing, max 20 words)"""

            response = get_response_with_retry(prompt)
            response = response.strip()
            response = response.replace('```json', '').replace('```', '')

            result = fix_json_response(response)
            if result is None:
                return {
                    "can_answer": True,
                    "missing_info": "",
                    "subquery": question,
                    "current_understanding": "Failed to parse reflection response.",
                    "dependencies": ["Information relevant to the question"],
                    "missing_reason": "Parse error occurred"
                }

            required_fields = ["can_answer", "missing_info", "subquery", "current_understanding"]
            if not all(field in result for field in required_fields):
                logger.error(f"{Fore.RED}Missing required fields in response: {response}{Style.RESET_ALL}")
                raise ValueError("Missing required fields")

            if "dependencies" not in result:
                result["dependencies"] = ["Information relevant to the question"]
            if "missing_reason" not in result:
                result["missing_reason"] = "Additional context needed" if not result[
                    "can_answer"] else "No missing information"

            result["can_answer"] = bool(result["can_answer"])

            if not result["subquery"]:
                result["subquery"] = question

            return result

        except Exception as e:
            logger.error(f"{Fore.RED}Error in analyze_dependency_graph: {e}{Style.RESET_ALL}")
            return {
                "can_answer": True,
                "missing_info": "",
                "subquery": question,
                "current_understanding": f"Error during analysis: {str(e)}",
                "dependencies": ["Information relevant to the question"],
                "missing_reason": "Analysis error occurred"
            }

    def dependency_aware_rag(self, question: str, history: List[Dict], dependencies: List[str], idx: int) -> Dict:
        """
        Analyze if the question can be answered given the structured history.
        """
        history_text = self._format_history_for_llm(history)

        try:
            prompt = f"""
             We are solving the question: "{question}" by breaking it down into dependencies.

             Current Reasoning Chain (Executed Steps):
             {history_text}

             Pending Dependencies (To be solved):
             {dependencies[idx:]}

             Current dependency to be answered next: {dependencies[idx]}

             Please analyze:
             1. Based on the "Current Reasoning Chain", can the original question ("{question}") be answered completely NOW? (Yes/No)
             2. Summarize our current understanding based on the chain.

             Format response as JSON:
             - "can_answer": boolean (true or false)
             - "current_understanding": string

             Attention:Do not output any other content.
             """
            response = get_response_with_retry(prompt)
            result = fix_json_response(response)

            if result is None:
                logger.warning(
                    f"{Fore.YELLOW}dependency_aware_rag received invalid JSON. Using fallback.{Style.RESET_ALL}")
                return {
                    "can_answer": False,
                    "current_understanding": "Failed to parse dependency analysis response."
                }

            return result
        except Exception as e:
            logger.error(f"{Fore.RED}Error in dependency_aware_rag: {e}{Style.RESET_ALL}")
            return {
                "can_answer": False,
                "current_understanding": f"Error during analysis: {str(e)}",
            }

    def generate_answer(self, question: str, history: List[Dict]) -> str:
        """Generate final answer based on the reasoning chain."""
        history_text = self._format_history_for_llm(history)

        debug_message = history_text
        print(debug_message)

        try:
            prompt = f"""
            You are a strict answer generator. You must generate the final answer based on the provided reasoning process.

            Question: {question}

            Reasoning Process:
            {history_text}

            【Strict Constraints】:
            1. Give ONLY the direct answer. DO NOT explain or provide any additional context.
            2. If the answer is a name, date, or number, output JUST that entity.
            3. If the answer is a simple yes/no, just say "Yes" or "No".
            4. If the answer requires a brief phrase, make it as concise as possible.

            Concise Answer: """
            answer =get_response_with_retry(prompt)
            print(f'''  - Final Answer:{answer}''')
            return answer
        except Exception as e:
            logger.error(f"{Fore.RED}Error generating answer: {e}{Style.RESET_ALL}")
            return ""

    def _sort_dependencies(self, dependencies: List[str], query) -> List[Tuple]:
        """
        given a list of dependencies and the original query,
        sort the dependencies in a topological order.
        """
        prompt = f"""
        Given the question:
        Question: {query}

        and its decomposed dependencies:
        Dependencies: {dependencies}

        Please output the dependency pairs that dependency A relies on dependency B, if any. If no dependency pairs are found, output an empty list.

        format your response as a JSON object with these keys:
        - "dependency_pairs": list of tuples of integers (e.g., [[0, 1]])
        """
        response = get_response_with_retry(prompt)
        result = fix_json_response(response)
        dependency_pairs = result["dependency_pairs"]

        sorted_dependencies = self._topological_sort(dependencies, dependency_pairs)
        return sorted_dependencies

    @staticmethod
    def _topological_sort(dependencies: List[str], dependencies_pairs: List[Tuple[int, int]]) -> List[str]:
        """
        Use graph-based algorithm to sort the dependencies in a topological order.
        """
        graph = {dep: [] for dep in dependencies}

        for dependent_idx, dependency_idx in dependencies_pairs:
            if dependent_idx < len(dependencies) and dependency_idx < len(dependencies):
                dependent = dependencies[dependent_idx]
                dependency = dependencies[dependency_idx]
                graph[dependency].append(dependent)

        visited = set()
        stack = []

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for neighbor in graph[node]:
                dfs(neighbor)
            stack.append(node)

        for node in graph:
            if node not in visited:
                dfs(node)

        return stack[::-1]

    def _retrieve_with_filter(self, query: str, retrieved_chunks_set: set) -> list:
        """
        Retrieve top_k unique chunks not in retrieved_chunks_set.
        """
        all_results = self.retrieve(query)
        unique_results = []
        idx = self.top_k
        while len(unique_results) < self.top_k and idx <= len(self.corpus):
            all_results = self.retrieve(query) if idx == self.top_k else self._retrieve_top_n(query, idx)
            unique_results = [chunk for chunk in all_results if chunk not in retrieved_chunks_set]
            idx += self.top_k
        return unique_results[:self.top_k]

    def _retrieve_top_n(self, query: str, n: int) -> list:
        """Retrieve top-n results for a query (helper for filtering)."""
        old_top_k = self.top_k
        self.top_k = n
        results = self.retrieve(query)
        self.top_k = old_top_k
        return results

    # ===== New: perturbation helpers =====
    def _is_target_perturbation_step(self, step_type: str, reasoning_step_index: int) -> bool:
        if not self.perturbation_enabled:
            return False
        if self.perturbation_type == "none":
            return False
        if step_type != "reasoning_step":
            return False
        if self.perturbation_step_mode == "first_reasoning_step":
            return reasoning_step_index == 0
        return False

    def _maybe_perturb_step_result(self, step_result: Dict[str, str], step_type: str, reasoning_step_index: int):
        original_summary = step_result.get("summary", "")
        original_answer = step_result.get("answer", "")

        perturbation_log = {
            "perturbation_applied": False,
            "perturbation_enabled": self.perturbation_enabled,
            "perturbation_type": self.perturbation_type,
            "perturbation_step_mode": self.perturbation_step_mode,
            "target_step_type": step_type,
            "target_reasoning_step_index": reasoning_step_index,
            "original_summary": original_summary,
            "original_answer": original_answer,
            "perturbed_summary": original_summary,
            "perturbed_answer": original_answer,
        }

        if not self._is_target_perturbation_step(step_type, reasoning_step_index):
            return step_result, perturbation_log

        perturbed_result = copy.deepcopy(step_result)

        if self.perturbation_type == "answer_wrong":
            perturbed_result["answer"] = self.answer_wrong_template
            perturbation_log["perturbation_applied"] = True
            perturbation_log["perturbed_answer"] = perturbed_result["answer"]

        elif self.perturbation_type == "summary_wrong":
            perturbed_result["summary"] = self.summary_wrong_template
            perturbation_log["perturbation_applied"] = True
            perturbation_log["perturbed_summary"] = perturbed_result["summary"]

        return perturbed_result, perturbation_log

    def answer_question(self, question: str) -> Tuple[str, List[str], int]:

        # --- initialize ---
        history = []
        dependency_analysis_history = []
        last_contexts = []
        retrieval_history = []
        round_count = 0
        reasoning_step_counter = 0
        retrieved_chunks_set = set() if self.filter_repeats else None

        self.last_dependency_analysis = []
        self.last_history = []
        self.last_run_metadata = {}
        self.last_perturbed_step_info = None
        self.last_retrieval_history = []

        print(f"\n\n{Fore.CYAN}{self.MODEL_NAME} answering: {question}{Style.RESET_ALL}\n\n")

        # ===============================================
        # == Stage 1: Warm up retrieval ==
        # ===============================================
        if self.filter_repeats:
            new_contexts = self._retrieve_with_filter(question, retrieved_chunks_set)
            for chunk in new_contexts:
                retrieved_chunks_set.add(chunk)
        else:
            new_contexts = self.retrieve(question)

        last_contexts = new_contexts

        warmup_step_result = self.process_step(question, question, new_contexts)
        warmup_step_result, warmup_perturbation_log = self._maybe_perturb_step_result(
            warmup_step_result,
            step_type="initial_attempt",
            reasoning_step_index=-1
        )

        history.append({
            "step_type": "initial_attempt",
            "query": question,
            "summary": warmup_step_result.get("summary", ""),
            "answer": warmup_step_result.get("answer", ""),
            "perturbation_log": warmup_perturbation_log
        })

        analysis = self.warm_up_analysis(question, history)

        if analysis["can_answer"]:
            print(
                "Warm-up analysis indicate the question can be answered with simple fact retrieval, without any dependency analysis.")
            answer = self.generate_answer(question, history)

            self.last_dependency_analysis = []
            self.last_history = history if self.save_history else []
            self.last_retrieval_history = retrieval_history if self.save_history else []
            self.last_run_metadata = {
                "question": question,
                "num_history_steps": len(history),
                "num_reasoning_steps": reasoning_step_counter,
                "has_reasoning_step": reasoning_step_counter > 0,
                "perturbation_enabled": self.perturbation_enabled,
                "perturbation_type": self.perturbation_type,
                "perturbation_step_mode": self.perturbation_step_mode,
                "perturbation_applied": False,
                "experiment_tag": self.experiment_tag,
                "rounds": round_count,
                "early_stop_from_warmup": True,
            }
            self.last_perturbed_step_info = None
            return answer, last_contexts, round_count

        else:
            logger.info(
                "Warm-up analysis indicate the requirement of deeper reasoning-enhanced RAG. Now perform analysis with logical dependency graph.")
            logger.info(f"Dependencies: {', '.join(analysis.get('dependencies', []))}")

            sorted_dependencies = self._sort_dependencies(analysis["dependencies"], question)
            dependency_analysis_history.append({"sorted_dependencies": sorted_dependencies})
            logger.info(f"Sorted dependencies: {sorted_dependencies}\n\n")

        # ===============================================
        # == Stage 2: agentic iterative retrieval ==
        # ===============================================
        idx = 0

        while round_count < self.max_rounds and idx < len(sorted_dependencies):
            round_count += 1

            current_query = sorted_dependencies[idx]
            if self.filter_repeats:
                new_contexts = self._retrieve_with_filter(current_query, retrieved_chunks_set)
                for chunk in new_contexts:
                    retrieved_chunks_set.add(chunk)
            else:
                new_contexts = self.retrieve(current_query)
            last_contexts = new_contexts

            step_result = self.process_step(question, current_query, new_contexts)
            step_result, step_perturbation_log = self._maybe_perturb_step_result(
                step_result,
                step_type="reasoning_step",
                reasoning_step_index=reasoning_step_counter
            )

            history.append({
                "query": current_query,
                "step_type": "reasoning_step",
                "summary": step_result.get("summary", ""),
                "answer": step_result.get("answer", ""),
                "perturbation_log": step_perturbation_log
            })

            if step_perturbation_log.get("perturbation_applied", False):
                self.last_perturbed_step_info = step_perturbation_log

            reasoning_step_counter += 1

            logger.info(f"Agentic retrieval at round {round_count} - Sub-answer: {step_result['answer']}")

            analysis = self.dependency_aware_rag(question, history, sorted_dependencies, idx)

            retrieval_history.append({
                "round": round_count,
                "query": current_query,
                "contexts": new_contexts,
            })

            dependency_analysis_history.append({
                "round": round_count,
                "query": current_query,
                "analysis": analysis
            })

            if analysis["can_answer"]:
                answer = self.generate_answer(question, history)

                perturbation_applied = any(
                    step.get("perturbation_log", {}).get("perturbation_applied", False)
                    for step in history
                )

                self.last_dependency_analysis = dependency_analysis_history
                self.last_history = history if self.save_history else []
                self.last_retrieval_history = retrieval_history if self.save_history else []
                self.last_run_metadata = {
                    "question": question,
                    "num_history_steps": len(history),
                    "num_reasoning_steps": reasoning_step_counter,
                    "has_reasoning_step": reasoning_step_counter > 0,
                    "reasoning_step_indices": list(range(reasoning_step_counter)),
                    "perturbation_enabled": self.perturbation_enabled,
                    "perturbation_type": self.perturbation_type,
                    "perturbation_step_mode": self.perturbation_step_mode,
                    "perturbation_applied": perturbation_applied,
                    "experiment_tag": self.experiment_tag,
                    "rounds": round_count,
                    "early_stop_from_warmup": False,
                    "stopped_because_can_answer": True,
                }
                return answer, last_contexts, round_count
            else:
                idx += 1

        logger.info(f"Reached maximum rounds ({self.max_rounds}). Generating final answer...")
        answer = self.generate_answer(question, history)

        perturbation_applied = any(
            step.get("perturbation_log", {}).get("perturbation_applied", False)
            for step in history
        )

        self.last_dependency_analysis = dependency_analysis_history
        self.last_history = history if self.save_history else []
        self.last_retrieval_history = retrieval_history if self.save_history else []
        self.last_run_metadata = {
            "question": question,
            "num_history_steps": len(history),
            "num_reasoning_steps": reasoning_step_counter,
            "has_reasoning_step": reasoning_step_counter > 0,
            "reasoning_step_indices": list(range(reasoning_step_counter)),
            "perturbation_enabled": self.perturbation_enabled,
            "perturbation_type": self.perturbation_type,
            "perturbation_step_mode": self.perturbation_step_mode,
            "perturbation_applied": perturbation_applied,
            "experiment_tag": self.experiment_tag,
            "rounds": round_count,
            "early_stop_from_warmup": False,
            "stopped_because_can_answer": False,
        }
        return answer, last_contexts, round_count

    def _format_history_for_llm(self, history: List[Dict[str, Any]]) -> str:
        """
        将推理历史列表格式化为清晰的字符串，供LLM阅读。
        """
        formatted_text = ""
        for i, step in enumerate(history):
            formatted_text += f"Step {i + 1}:\n"
            formatted_text += f"  - Sub-Query: {step.get('query', '')}\n"
            formatted_text += f"  - Context Summary: {step.get('summary', '')}\n"
            formatted_text += f"  - Direct Answer: {step.get('answer', '')}\n\n"

        return formatted_text.strip()