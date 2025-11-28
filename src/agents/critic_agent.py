"""
Critic Agent

This agent evaluates the quality and accuracy of research outputs.
It provides feedback for improvement and determines if the work meets quality standards.
"""

from typing import Dict, Any, List, Optional
import logging
from src.agents.base_agent import BaseAgent


class CriticAgent(BaseAgent):
    """
    Agent responsible for quality evaluation and verification.

    The critic evaluates:
    - Relevance to the original query
    - Quality of evidence and sources
    - Completeness of coverage
    - Factual accuracy
    - Clarity and organization
    """

    def __init__(
        self,
        system_prompt: str = "",
        config: Optional[Dict[str, Any]] = None,
        model_client = None
    ):
        """
        Initialize the Critic Agent.

        Args:
            system_prompt: Custom system prompt (empty uses default)
            config: Configuration dictionary
            model_client: Optional LLM client (if using AutoGen)
        """
        super().__init__(
            name="Critic",
            role="Quality Verifier",
            system_prompt=system_prompt,
            config=config
        )
        self.model_client = model_client

    def _get_default_prompt(self) -> str:
        """Get default system prompt for the critic."""
        return """You are a Research Critic. Your job is to evaluate the quality and accuracy of research outputs.

Evaluate the research and writing on these criteria:
1. **Relevance**: Does it answer the original query?
2. **Evidence Quality**: Are sources credible and well-cited?
3. **Completeness**: Are all aspects of the query addressed?
4. **Accuracy**: Are there any factual errors or contradictions?
5. **Clarity**: Is the writing clear and well-organized?

Provide constructive but thorough feedback. End your evaluation with either "APPROVED - RESEARCH COMPLETE" if approved, or "NEEDS REVISION" if improvements are needed. You may also use "TERMINATE" to signal completion."""

    def process(
        self,
        draft: str,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a research draft and provide feedback.

        Args:
            draft: Written draft to evaluate
            query: Original research query
            context: Optional context (sources, plan, etc.)

        Returns:
            Dictionary with:
            - evaluation: Detailed evaluation and feedback
            - scores: Scores for each criterion (0-1)
            - approved: Boolean indicating if approved
            - feedback: Specific feedback for improvement
            - metadata: Additional evaluation information
        """
        self.logger.info("Evaluating research draft...")

        # Evaluate on multiple criteria
        scores = self._evaluate_criteria(draft, query, context)

        # Generate feedback
        feedback = self._generate_feedback(scores, draft, query)

        # Determine approval
        overall_score = sum(scores.values()) / len(scores) if scores else 0.0
        approved = overall_score >= 0.7  # Threshold for approval

        evaluation = self._format_evaluation(scores, feedback, approved)

        self.logger.info(f"Evaluation complete. Approved: {approved}, Score: {overall_score:.2f}")

        return {
            "evaluation": evaluation,
            "scores": scores,
            "approved": approved,
            "feedback": feedback,
            "metadata": {
                "overall_score": overall_score,
                "criteria_evaluated": list(scores.keys())
            }
        }

    def _evaluate_criteria(
        self,
        draft: str,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate the draft on multiple criteria.

        Returns scores from 0.0 to 1.0 for each criterion.
        """
        scores = {}

        # Relevance: Check if query terms appear in draft
        query_terms = set(query.lower().split())
        draft_lower = draft.lower()
        relevant_terms = sum(1 for term in query_terms if term in draft_lower)
        scores["relevance"] = min(1.0, relevant_terms / max(1, len(query_terms)))

        # Evidence Quality: Check for citations and sources
        citation_count = draft.count("[Source:") + draft.count("References")
        scores["evidence_quality"] = min(1.0, citation_count / 5.0)  # Normalize to 5 citations

        # Completeness: Check draft length and structure
        has_intro = "introduction" in draft_lower or "overview" in draft_lower
        has_findings = "finding" in draft_lower or "result" in draft_lower
        has_conclusion = "conclusion" in draft_lower or "summary" in draft_lower
        structure_score = (has_intro + has_findings + has_conclusion) / 3.0
        length_score = min(1.0, len(draft.split()) / 500.0)  # Normalize to 500 words
        scores["completeness"] = (structure_score + length_score) / 2.0

        # Accuracy: Basic check (can be enhanced with fact-checking)
        # For now, assume good if sources are cited
        scores["accuracy"] = scores["evidence_quality"] * 0.9  # Slightly lower than evidence quality

        # Clarity: Check for clear structure and formatting
        has_headings = draft.count("#") > 0
        has_paragraphs = draft.count("\n\n") > 3
        clarity_score = (has_headings + has_paragraphs) / 2.0
        scores["clarity"] = clarity_score

        return scores

    def _generate_feedback(
        self,
        scores: Dict[str, float],
        draft: str,
        query: str
    ) -> List[str]:
        """Generate specific feedback based on scores."""
        feedback = []

        if scores.get("relevance", 0) < 0.7:
            feedback.append("The response may not fully address all aspects of the query. Consider expanding coverage of key topics.")

        if scores.get("evidence_quality", 0) < 0.7:
            feedback.append("More citations and source references would strengthen the response. Ensure all claims are supported.")

        if scores.get("completeness", 0) < 0.7:
            feedback.append("The response could be more comprehensive. Consider adding more detail or covering additional aspects.")

        if scores.get("accuracy", 0) < 0.7:
            feedback.append("Verify factual claims and ensure information is accurate and up-to-date.")

        if scores.get("clarity", 0) < 0.7:
            feedback.append("Improve organization and structure. Use clear headings and well-organized paragraphs.")

        if not feedback:
            feedback.append("The response meets quality standards across all criteria.")

        return feedback

    def _format_evaluation(
        self,
        scores: Dict[str, float],
        feedback: List[str],
        approved: bool
    ) -> str:
        """Format the evaluation as a readable text."""
        evaluation = "## Quality Evaluation\n\n"

        evaluation += "### Scores\n\n"
        for criterion, score in scores.items():
            evaluation += f"- **{criterion.replace('_', ' ').title()}**: {score:.2f}/1.00\n"

        overall = sum(scores.values()) / len(scores) if scores else 0.0
        evaluation += f"\n**Overall Score**: {overall:.2f}/1.00\n\n"

        evaluation += "### Feedback\n\n"
        for i, item in enumerate(feedback, 1):
            evaluation += f"{i}. {item}\n"

        evaluation += "\n"
        if approved:
            evaluation += "**Status**: APPROVED - RESEARCH COMPLETE\n"
        else:
            evaluation += "**Status**: NEEDS REVISION\n"

        return evaluation


# For AutoGen integration
def create_critic_agent(config: Dict[str, Any], model_client) -> 'CriticAgent':
    """
    Create a Critic Agent instance for AutoGen integration.

    Args:
        config: Configuration dictionary
        model_client: AutoGen model client

    Returns:
        CriticAgent instance
    """
    agent_config = config.get("agents", {}).get("critic", {})
    system_prompt = agent_config.get("system_prompt", "").strip()

    return CriticAgent(
        system_prompt=system_prompt,
        config=config,
        model_client=model_client
    )
