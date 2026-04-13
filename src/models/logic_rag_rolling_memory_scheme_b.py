import logging
from typing import List, Dict, Tuple

from colorama import Fore, Style, init

from src.models.base_rag import BaseRAG
from src.utils.utils import get_response_with_retry, fix_json_response

# Initialize colorama
init()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogicRAGRollingMemoryBaseline(BaseRAG):
    """
    Scheme B:
    - Keep the original LogicRAG rolling-memory paradigm.
    - Use the same summary_wrong hallucination text as ScaffoldRAG_case.
    - Inject it directly at the first sub-query.
    - Skip retrieval/summarization for that first sub-query to save tokens.
    - Continue subsequent reasoning exactly with rolling memory.
    """

    def __init__(
        self,
        corpus_path: str = None,
        cache_dir: str = "./cache",
        filter_repeats: bool = False,
        hallucination_enabled: bool = False,
        hallucination_type: str = "none",
        save_history: bool = False,
        experiment_tag: str = "rolling_memory_baseline",
    ):
        super().__init__(corpus_path, cache_dir)
        self.max_rounds = 3
        self.MODEL_NAME = "LogicRAGRollingMemoryBaseline"
        self.filter_repeats = filter_repeats

        # Scheme-B experiment config
        self.hallucination_enabled = hallucination_enabled
        self.hallucination_type = hallucination_type
        self.save_history = save_history
        self.experiment_tag = experiment_tag

        # Keep EXACT same summary_wrong text as ScaffoldRAG_case
        self.summary_wrong_template = (
            "The evidence is incorrectly summarized around a misleading conclusion rather than the actual key fact."
        )

        # Runtime traces
        self.last_dependency_analysis = []
        self.last_retrieval_history = []
        self.last_memory_trace = []
        self.last_run_metadata = {}
        self.last_hallucination_info = None

    def set_max_rounds(self, max_rounds: int):
        self.max_rounds = max_rounds

    def refine_summary_with_context(
        self,
        question: str,
        new_contexts: List[str],
        current_summary: str = "",
    ) -> str:
        """
        Original LogicRAG rolling-memory summarization.
        """
        try:
            context_text = "\n".join(new_contexts)

            if not current_summary:
                prompt = f"""Please create a concise summary of the following information as it relates to answering this question:

Question: {question}

Information:
{context_text}

Your summary should:
1. Include all relevant facts that might help answer the question
2. Exclude irrelevant information
3. Be clear and concise
4. Preserve specific details, dates, numbers, and names that may be relevant

Summary:"""
            else:
                prompt = f"""Please refine the following information summary using newly retrieved information.

Question: {question}

Current summary:
{current_summary}

New information:
{context_text}

Your refined summary should:
1. Integrate new relevant facts with the existing summary
2. Remove redundancies
3. Remain concise while preserving all important information
4. Prioritize information that helps answer the question
5. Maintain specific details, dates, numbers, and names that may be relevant

Refined summary:"""

            summary = get_response_with_retry(prompt)
            return summary

        except Exception as e:
            logger.error(f"{Fore.RED}Error generating/refining summary: {e}{Style.RESET_ALL}")
            context_text = "\n".join(new_contexts)
            if current_summary:
                return f"{current_summary}\n\nNew information:\n{context_text}"
            return context_text

    def warm_up_analysis(self, question: str, info_summary: str) -> Dict:
        try:
            prompt = f"""Question: {question}

Available Information:
{info_summary}

Based on the information provided, please analyze:
1. Can the question be answered completely with this information? (Yes/No)
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
            response = response.strip().replace("```json", "").replace("```", "")
            result = fix_json_response(response)

            if result is None:
                return {
                    "can_answer": True,
                    "missing_info": "",
                    "subquery": question,
                    "current_understanding": "Failed to parse reflection response.",
                    "dependencies": ["Information relevant to the question"],
                    "missing_reason": "Parse error occurred",
                }

            required_fields = ["can_answer", "missing_info", "subquery", "current_understanding"]
            if not all(field in result for field in required_fields):
                logger.error(f"{Fore.RED}Missing required fields in response: {response}{Style.RESET_ALL}")
                raise ValueError("Missing required fields")

            if "dependencies" not in result:
                result["dependencies"] = ["Information relevant to the question"]
            if "missing_reason" not in result:
                result["missing_reason"] = (
                    "Additional context needed" if not result["can_answer"] else "No missing information"
                )

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
                "missing_reason": "Analysis error occurred",
            }

    def dependency_aware_rag(self, question: str, info_summary: str, dependencies: List[str], idx: int) -> Dict:
        try:
            prompt = f"""
            We pre-parsed the question into a list of dependencies, and the dependencies are sorted in a topological order, below is the question, the information summary, and the decomposed dependencies:

            Question: {question}

            Available Information:
            {info_summary}

            Decomposed dependencies:
            {dependencies}

            Current dependency to be answered:
            {dependencies[idx]}

            Please analyze the question and the information summary, and the decomposed dependencies, and answer the following questions:
            Please analyze:
            1. Can the question be answered completely with this information? (Yes/No)
            2. Summarize our current understanding based on available information.

            Please format your response as a JSON object with these keys:
            - "can_answer": boolean
            - "current_understanding": string
            """
            response = get_response_with_retry(prompt)
            result = fix_json_response(response)
            return result
        except Exception as e:
            logger.error(f"{Fore.RED}Error in dependency_aware_rag: {e}{Style.RESET_ALL}")
            return {
                "can_answer": True,
                "current_understanding": f"Error during analysis: {str(e)}",
            }

    def generate_answer(self, question: str, info_summary: str) -> str:
        try:
            prompt = f"""You must give ONLY the direct answer in the most concise way possible. DO NOT explain or provide any additional context.
If the answer is a simple yes/no, just say "Yes." or "No."
If the answer is a name, just give the name.
If the answer is a date, just give the date.
If the answer is a number, just give the number.
If the answer requires a brief phrase, make it as concise as possible.

Question: {question}

Information Summary:
{info_summary}

Remember: Be concise - give ONLY the essential answer, nothing more.
Ans: """
            return get_response_with_retry(prompt)
        except Exception as e:
            logger.error(f"{Fore.RED}Error generating answer: {e}{Style.RESET_ALL}")
            return ""

    def _sort_dependencies(self, dependencies: List[str], query: str) -> List[str]:
        prompt = f"""
        Given the question:
        Question: {query}

        and its decomposed dependencies:
        Dependencies: {dependencies}

        Please output the dependency pairs that dependency A relies on dependency B, if any. If no dependency pairs are found, output an empty list.

        format your response as a JSON object with these keys:
        - "dependency_pairs": list of tuples of integers
        """
        response = get_response_with_retry(prompt)
        result = fix_json_response(response)
        dependency_pairs = result["dependency_pairs"]
        sorted_dependencies = self._topological_sort(dependencies, dependency_pairs)
        return sorted_dependencies

    @staticmethod
    def _topological_sort(dependencies: List[str], dependencies_pairs: List[Tuple[int, int]]) -> List[str]:
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

    def _retrieve_with_filter(self, query: str, retrieved_chunks_set: set) -> List[str]:
        all_results = self.retrieve(query)
        unique_results = []
        idx = self.top_k

        while len(unique_results) < self.top_k and idx <= len(self.corpus):
            all_results = self.retrieve(query) if idx == self.top_k else self._retrieve_top_n(query, idx)
            unique_results = [chunk for chunk in all_results if chunk not in retrieved_chunks_set]
            idx += self.top_k
        return unique_results[:self.top_k]

    def _retrieve_top_n(self, query: str, n: int) -> List[str]:
        old_top_k = self.top_k
        self.top_k = n
        results = self.retrieve(query)
        self.top_k = old_top_k
        return results

    def _reset_runtime_logs(self):
        self.last_dependency_analysis = []
        self.last_retrieval_history = []
        self.last_memory_trace = []
        self.last_run_metadata = {}
        self.last_hallucination_info = None

    def _append_memory_trace(self, stage: str, info_summary: str, query: str = ""):
        self.last_memory_trace.append({
            "stage": stage,
            "query": query,
            "info_summary": info_summary,
        })

    def _should_inject_first_subquery_hallucination(self) -> bool:
        return self.hallucination_enabled and self.hallucination_type == "summary_wrong"

    def _inject_first_subquery_memory(self, current_summary: str) -> str:
        """
        No extra retrieval and no extra summarization call.
        Directly write the fixed hallucinated summary into rolling memory.
        """
        injected = self.summary_wrong_template
        if current_summary and current_summary.strip():
            return f"{current_summary}\n\nInjected memory update:\n{injected}"
        return injected

    def answer_question(self, question: str):
        info_summary = ""
        round_count = 0
        retrieval_history = []
        last_contexts = []
        dependency_analysis_history = []
        retrieved_chunks_set = set() if self.filter_repeats else None

        self._reset_runtime_logs()

        print(f"\n\n{Fore.CYAN}{self.MODEL_NAME} answering: {question}{Style.RESET_ALL}\n\n")

        # ===========================
        # Stage 1: warm-up retrieval
        # ===========================
        if self.filter_repeats:
            new_contexts = self._retrieve_with_filter(question, retrieved_chunks_set)
            for chunk in new_contexts:
                retrieved_chunks_set.add(chunk)
        else:
            new_contexts = self.retrieve(question)

        last_contexts = new_contexts
        info_summary = self.refine_summary_with_context(question, new_contexts, info_summary)
        self._append_memory_trace(stage="warmup", info_summary=info_summary, query=question)

        analysis = self.warm_up_analysis(question, info_summary)

        if analysis["can_answer"]:
            print("Warm-up analysis indicate the question can be answered with simple fact retrieval, without any dependency analysis.")
            answer = self.generate_answer(question, info_summary)

            self.last_dependency_analysis = []
            self.last_retrieval_history = retrieval_history if self.save_history else []
            self.last_memory_trace = self.last_memory_trace if self.save_history else []
            self.last_run_metadata = {
                "question": question,
                "memory_strategy": "rolling_memory_scheme_b",
                "hallucination_enabled": self.hallucination_enabled,
                "hallucination_type": self.hallucination_type,
                "hallucination_applied": False,
                "experiment_tag": self.experiment_tag,
                "rounds": round_count,
                "has_reasoning_step": False,
                "early_stop_from_warmup": True,
            }
            return answer, last_contexts, round_count

        logger.info("Warm-up analysis indicate the requirement of deeper reasoning-enhanced RAG. Now perform analysis with logical dependency graph.")
        logger.info(f"Dependencies: {', '.join(analysis.get('dependencies', []))}")

        sorted_dependencies = self._sort_dependencies(analysis["dependencies"], question)
        dependency_analysis_history.append({"sorted_dependencies": sorted_dependencies})
        logger.info(f"Sorted dependencies: {sorted_dependencies}\n\n")

        idx = 0

        # ==========================================================
        # Scheme B: inject summary_wrong directly at first sub-query
        # ==========================================================
        if idx < len(sorted_dependencies) and round_count < self.max_rounds and self._should_inject_first_subquery_hallucination():
            round_count += 1
            first_query = sorted_dependencies[idx]

            info_summary = self._inject_first_subquery_memory(info_summary)
            self._append_memory_trace(stage="hallucination_injection", info_summary=info_summary, query=first_query)

            hallucination_info = {
                "hallucination_applied": True,
                "hallucination_type": self.hallucination_type,
                "target_query": first_query,
                "injected_summary": self.summary_wrong_template,
                "skipped_retrieval": True,
                "stage": "first_subquery",
            }
            self.last_hallucination_info = hallucination_info

            retrieval_history.append({
                "round": round_count,
                "query": first_query,
                "contexts": [],
                "skipped_retrieval_for_hallucination": True,
                "hallucination_info": hallucination_info,
            })

            analysis = self.dependency_aware_rag(question, info_summary, sorted_dependencies, idx)
            dependency_analysis_history.append({
                "round": round_count,
                "query": first_query,
                "analysis": analysis,
                "hallucination_injected": True,
            })

            if analysis["can_answer"]:
                answer = self.generate_answer(question, info_summary)
                self.last_dependency_analysis = dependency_analysis_history
                self.last_retrieval_history = retrieval_history if self.save_history else []
                self.last_memory_trace = self.last_memory_trace if self.save_history else []
                self.last_run_metadata = {
                    "question": question,
                    "memory_strategy": "rolling_memory_scheme_b",
                    "hallucination_enabled": self.hallucination_enabled,
                    "hallucination_type": self.hallucination_type,
                    "hallucination_applied": True,
                    "experiment_tag": self.experiment_tag,
                    "rounds": round_count,
                    "has_reasoning_step": True,
                    "early_stop_from_warmup": False,
                    "stopped_because_can_answer": True,
                }
                return answer, last_contexts, round_count

            idx += 1

        # ===============================================
        # Stage 2: continue standard rolling-memory logic
        # ===============================================
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
            info_summary = self.refine_summary_with_context(question, new_contexts, info_summary)
            self._append_memory_trace(stage="rolling_update", info_summary=info_summary, query=current_query)

            logger.info(f"Agentic retrieval at round {round_count}")
            logger.info(f"current query: {current_query}")

            analysis = self.dependency_aware_rag(question, info_summary, sorted_dependencies, idx)

            retrieval_history.append({
                "round": round_count,
                "query": current_query,
                "contexts": new_contexts,
                "skipped_retrieval_for_hallucination": False,
            })

            dependency_analysis_history.append({
                "round": round_count,
                "query": current_query,
                "analysis": analysis,
                "hallucination_injected": False,
            })

            if analysis["can_answer"]:
                answer = self.generate_answer(question, info_summary)
                self.last_dependency_analysis = dependency_analysis_history
                self.last_retrieval_history = retrieval_history if self.save_history else []
                self.last_memory_trace = self.last_memory_trace if self.save_history else []
                self.last_run_metadata = {
                    "question": question,
                    "memory_strategy": "rolling_memory_scheme_b",
                    "hallucination_enabled": self.hallucination_enabled,
                    "hallucination_type": self.hallucination_type,
                    "hallucination_applied": self.last_hallucination_info is not None,
                    "experiment_tag": self.experiment_tag,
                    "rounds": round_count,
                    "has_reasoning_step": True,
                    "early_stop_from_warmup": False,
                    "stopped_because_can_answer": True,
                }
                return answer, last_contexts, round_count

            idx += 1

        logger.info(f"Reached maximum rounds ({self.max_rounds}). Generating final answer...")
        answer = self.generate_answer(question, info_summary)

        self.last_dependency_analysis = dependency_analysis_history
        self.last_retrieval_history = retrieval_history if self.save_history else []
        self.last_memory_trace = self.last_memory_trace if self.save_history else []
        self.last_run_metadata = {
            "question": question,
            "memory_strategy": "rolling_memory_scheme_b",
            "hallucination_enabled": self.hallucination_enabled,
            "hallucination_type": self.hallucination_type,
            "hallucination_applied": self.last_hallucination_info is not None,
            "experiment_tag": self.experiment_tag,
            "rounds": round_count,
            "has_reasoning_step": len(sorted_dependencies) > 0,
            "early_stop_from_warmup": False,
            "stopped_because_can_answer": False,
        }
        return answer, last_contexts, round_count
