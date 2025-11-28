"""
Researcher Agent

This agent gathers evidence from web and academic sources using search APIs.
It integrates with Tavily, Brave Search, and Semantic Scholar APIs.
"""

from typing import Dict, Any, List, Optional
import logging
import asyncio
from src.agents.base_agent import BaseAgent
from src.tools.web_search import WebSearchTool, web_search
from src.tools.paper_search import PaperSearchTool, paper_search


class ResearcherAgent(BaseAgent):
    """
    Agent responsible for gathering research evidence.

    The researcher uses web search and paper search tools to find:
    - Relevant academic papers
    - Web articles and blog posts
    - Authoritative sources
    - Recent publications
    """

    def __init__(
        self,
        system_prompt: str = "",
        config: Optional[Dict[str, Any]] = None,
        model_client = None
    ):
        """
        Initialize the Researcher Agent.

        Args:
            system_prompt: Custom system prompt (empty uses default)
            config: Configuration dictionary
            model_client: Optional LLM client (if using AutoGen)
        """
        super().__init__(
            name="Researcher",
            role="Evidence Gatherer",
            system_prompt=system_prompt,
            config=config
        )
        self.model_client = model_client

        # Initialize search tools from config
        tools_config = config.get("tools", {}) if config else {}
        web_config = tools_config.get("web_search", {})
        paper_config = tools_config.get("paper_search", {})

        self.web_search_tool = WebSearchTool(
            provider=web_config.get("provider", "tavily"),
            max_results=web_config.get("max_results", 5)
        )
        self.paper_search_tool = PaperSearchTool(
            max_results=paper_config.get("max_results", 10)
        )

        self.max_sources = config.get("agents", {}).get("researcher", {}).get("max_sources", 10) if config else 10

    def _get_default_prompt(self) -> str:
        """Get default system prompt for the researcher."""
        return """You are a Research Assistant. Your job is to gather high-quality information from academic papers and web sources.

You have access to tools for web search and paper search. When conducting research:
1. Use both web search and paper search for comprehensive coverage
2. Look for recent, high-quality sources
3. Extract key findings, quotes, and data
4. Note all source URLs and citations
5. Gather evidence that directly addresses the research query

After collecting sufficient evidence, say "RESEARCH COMPLETE"."""

    def process(
        self,
        research_plan: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Gather research evidence based on a plan.

        Args:
            research_plan: Research plan from the planner
            context: Optional context (search queries, key concepts, etc.)

        Returns:
            Dictionary with:
            - findings: List of research findings with sources
            - sources: List of source dictionaries
            - web_results: Web search results
            - paper_results: Academic paper results
            - metadata: Additional research information
        """
        self.logger.info("Starting research gathering...")

        # Extract search queries from plan or context
        search_queries = context.get("search_queries", []) if context else []
        if not search_queries:
            # Extract from plan text
            search_queries = self._extract_queries_from_plan(research_plan)

        if not search_queries:
            search_queries = ["research query"]  # Fallback

        # Gather evidence
        all_findings = []
        web_results = []
        paper_results = []

        for query in search_queries[:3]:  # Limit to 3 queries
            # Web search
            try:
                web_res = asyncio.run(self.web_search_tool.search(query))
                web_results.extend(web_res)
                all_findings.extend(self._format_web_findings(web_res))
            except Exception as e:
                self.logger.error(f"Web search error: {e}")

            # Paper search
            try:
                year_from = context.get("year_from") if context else None
                paper_res = asyncio.run(self.paper_search_tool.search(query, year_from=year_from))
                paper_results.extend(paper_res)
                all_findings.extend(self._format_paper_findings(paper_res))
            except Exception as e:
                self.logger.error(f"Paper search error: {e}")

        # Limit total sources
        all_findings = all_findings[:self.max_sources]

        self.logger.info(f"Gathered {len(all_findings)} research findings")

        return {
            "findings": all_findings,
            "sources": self._extract_sources(web_results, paper_results),
            "web_results": web_results,
            "paper_results": paper_results,
            "metadata": {
                "num_sources": len(all_findings),
                "queries_used": search_queries
            }
        }

    def _extract_queries_from_plan(self, plan: str) -> List[str]:
        """Extract search queries from the research plan."""
        queries = []
        lines = plan.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['search', 'query', 'find', 'look for']):
                # Extract potential query (simplified)
                queries.append(line.strip())
        return queries[:3]  # Limit to 3 queries

    def _format_web_findings(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format web search results as findings."""
        findings = []
        for result in results:
            findings.append({
                "type": "web",
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("snippet", ""),
                "published_date": result.get("published_date"),
            })
        return findings

    def _format_paper_findings(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format paper search results as findings."""
        findings = []
        for paper in results:
            findings.append({
                "type": "paper",
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "year": paper.get("year"),
                "abstract": paper.get("abstract", ""),
                "url": paper.get("url", ""),
                "citation_count": paper.get("citation_count", 0),
                "venue": paper.get("venue", ""),
            })
        return findings

    def _extract_sources(self, web_results: List[Dict], paper_results: List[Dict]) -> List[Dict[str, Any]]:
        """Extract source information for citations."""
        sources = []

        for web_res in web_results:
            sources.append({
                "type": "webpage",
                "title": web_res.get("title", ""),
                "url": web_res.get("url", ""),
                "year": None,  # Web sources may not have year
            })

        for paper in paper_results:
            sources.append({
                "type": "paper",
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "year": paper.get("year"),
                "venue": paper.get("venue", ""),
                "url": paper.get("url", ""),
            })

        return sources


# For AutoGen integration
def create_researcher_agent(config: Dict[str, Any], model_client) -> 'ResearcherAgent':
    """
    Create a Researcher Agent instance for AutoGen integration.

    Args:
        config: Configuration dictionary
        model_client: AutoGen model client

    Returns:
        ResearcherAgent instance
    """
    agent_config = config.get("agents", {}).get("researcher", {})
    system_prompt = agent_config.get("system_prompt", "").strip()

    return ResearcherAgent(
        system_prompt=system_prompt,
        config=config,
        model_client=model_client
    )
