"""
Planner Agent

This agent breaks down research queries into actionable research steps.
It uses LLM to analyze queries and create structured research plans.
"""

from typing import Dict, Any, List, Optional
import logging
from src.agents.base_agent import BaseAgent


class PlannerAgent(BaseAgent):
    """
    Agent responsible for planning research tasks.

    The planner analyzes research queries and breaks them down into:
    - Key concepts and topics to investigate
    - Types of sources needed (academic papers, web articles, etc.)
    - Specific search queries for the researcher
    - Outline for synthesizing findings
    """

    def __init__(
        self,
        system_prompt: str = "",
        config: Optional[Dict[str, Any]] = None,
        model_client = None
    ):
        """
        Initialize the Planner Agent.

        Args:
            system_prompt: Custom system prompt (empty uses default)
            config: Configuration dictionary
            model_client: Optional LLM client for planning (if using AutoGen)
        """
        super().__init__(
            name="Planner",
            role="Task Planner",
            system_prompt=system_prompt,
            config=config
        )
        self.model_client = model_client

    def _get_default_prompt(self) -> str:
        """Get default system prompt for the planner."""
        return """You are a Research Planner. Your job is to break down research queries into clear, actionable steps.

When given a research query, you should:
1. Identify the key concepts and topics to investigate
2. Determine what types of sources would be most valuable (academic papers, web articles, etc.)
3. Suggest specific search queries for the Researcher
4. Outline how the findings should be synthesized

Provide your plan in a structured format with numbered steps.
Be specific about what information to gather and why it's relevant.

After creating the plan, say "PLAN COMPLETE"."""

    def process(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a research query and create a plan.

        Args:
            query: Research query to plan for
            context: Optional context (e.g., topic domain, constraints)

        Returns:
            Dictionary with:
            - plan: Structured research plan
            - search_queries: List of suggested search queries
            - key_concepts: List of key concepts to investigate
            - metadata: Additional planning information
        """
        self.logger.info(f"Planning for query: {query[:100]}...")

        # If we have a model client (AutoGen integration), use it
        if self.model_client:
            # In AutoGen, the agent handles LLM calls internally
            # This method is for standalone usage
            return {
                "plan": f"Research plan for: {query}\n\n[Use AutoGen agent for full planning]",
                "search_queries": [query],
                "key_concepts": [],
                "metadata": {"method": "autogen"}
            }

        # Standalone planning logic (can be enhanced with direct LLM calls)
        # For now, create a basic plan structure
        plan = self._create_basic_plan(query, context)

        return {
            "plan": plan,
            "search_queries": self._extract_search_queries(plan),
            "key_concepts": self._extract_key_concepts(query),
            "metadata": {
                "query": query,
                "method": "standalone"
            }
        }

    def _create_basic_plan(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Create a basic research plan structure."""
        plan = f"Research Plan for: {query}\n\n"
        plan += "1. Analyze the query to identify key research areas\n"
        plan += "2. Determine required source types (academic papers, web articles)\n"
        plan += "3. Formulate specific search queries\n"
        plan += "4. Outline synthesis approach\n"
        plan += "\nPLAN COMPLETE"
        return plan

    def _extract_search_queries(self, plan: str) -> List[str]:
        """Extract suggested search queries from the plan."""
        # Simple extraction - can be enhanced
        queries = []
        lines = plan.split('\n')
        for line in lines:
            if 'search' in line.lower() or 'query' in line.lower():
                # Extract potential queries (simplified)
                queries.append(line.strip())
        return queries if queries else ["research query"]

    def _extract_key_concepts(self, query: str) -> List[str]:
        """Extract key concepts from the query."""
        # Simple keyword extraction
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'what', 'how', 'why', 'when', 'where'}
        words = query.lower().split()
        concepts = [w for w in words if w not in stop_words and len(w) > 3]
        return concepts[:5]  # Return top 5 concepts


# For AutoGen integration
def create_planner_agent(config: Dict[str, Any], model_client) -> 'PlannerAgent':
    """
    Create a Planner Agent instance for AutoGen integration.

    Args:
        config: Configuration dictionary
        model_client: AutoGen model client

    Returns:
        PlannerAgent instance
    """
    agent_config = config.get("agents", {}).get("planner", {})
    system_prompt = agent_config.get("system_prompt", "").strip()

    return PlannerAgent(
        system_prompt=system_prompt,
        config=config,
        model_client=model_client
    )
