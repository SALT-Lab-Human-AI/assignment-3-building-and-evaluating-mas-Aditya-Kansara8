"""
Writer Agent

This agent synthesizes research findings into coherent, well-cited responses.
It uses the citation tool to format references properly.
"""

from typing import Dict, Any, List, Optional
import logging
from src.agents.base_agent import BaseAgent
from src.tools.citation_tool import CitationTool


class WriterAgent(BaseAgent):
    """
    Agent responsible for synthesizing research into written responses.

    The writer:
    - Synthesizes findings from multiple sources
    - Creates structured, well-organized responses
    - Adds proper citations using APA format
    - Ensures the response directly answers the query
    """

    def __init__(
        self,
        system_prompt: str = "",
        config: Optional[Dict[str, Any]] = None,
        model_client = None
    ):
        """
        Initialize the Writer Agent.

        Args:
            system_prompt: Custom system prompt (empty uses default)
            config: Configuration dictionary
            model_client: Optional LLM client (if using AutoGen)
        """
        super().__init__(
            name="Writer",
            role="Report Synthesizer",
            system_prompt=system_prompt,
            config=config
        )
        self.model_client = model_client
        self.citation_tool = CitationTool(style="apa")

    def _get_default_prompt(self) -> str:
        """Get default system prompt for the writer."""
        return """You are a Research Writer. Your job is to synthesize research findings into clear, well-organized responses.

When writing:
1. Start with an overview/introduction
2. Present findings in a logical structure
3. Cite sources inline using [Source: Title/Author]
4. Synthesize information from multiple sources
5. Avoid copying text directly - paraphrase and synthesize
6. Include a references section at the end
7. Ensure the response directly answers the original query

Format your response professionally with clear headings, paragraphs, in-text citations, and a References section at the end.

After completing the draft, say "DRAFT COMPLETE"."""

    def process(
        self,
        research_findings: Dict[str, Any],
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synthesize research findings into a written response.

        Args:
            research_findings: Dictionary with findings, sources, etc. from researcher
            query: Original research query
            context: Optional context (plan, additional notes, etc.)

        Returns:
            Dictionary with:
            - draft: Written response with citations
            - citations: List of formatted citations
            - bibliography: Formatted bibliography
            - metadata: Additional writing information
        """
        self.logger.info("Synthesizing research findings into response...")

        findings = research_findings.get("findings", [])
        sources = research_findings.get("sources", [])

        # Add sources to citation tool
        for source in sources:
            self.citation_tool.add_citation(source)

        # Generate draft
        draft = self._synthesize_draft(findings, query, context)

        # Generate bibliography
        bibliography = self.citation_tool.generate_bibliography()

        # Add references section to draft
        if bibliography:
            draft += "\n\n## References\n\n"
            for i, citation in enumerate(bibliography, 1):
                draft += f"{i}. {citation}\n"

        self.logger.info("Draft synthesis complete")

        return {
            "draft": draft,
            "citations": bibliography,
            "bibliography": bibliography,
            "metadata": {
                "num_sources": len(sources),
                "num_citations": len(bibliography)
            }
        }

    def _synthesize_draft(
        self,
        findings: List[Dict[str, Any]],
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Synthesize findings into a coherent draft.

        This creates a structured response that answers the query.
        """
        draft = f"# Research Response: {query}\n\n"

        # Introduction
        draft += "## Introduction\n\n"
        draft += f"This response addresses the query: \"{query}\". "
        draft += f"Based on {len(findings)} sources, the following findings are presented.\n\n"

        # Main content - organize by type or topic
        draft += "## Findings\n\n"

        # Group findings by type
        web_findings = [f for f in findings if f.get("type") == "web"]
        paper_findings = [f for f in findings if f.get("type") == "paper"]

        if paper_findings:
            draft += "### Academic Research\n\n"
            for finding in paper_findings[:5]:  # Limit to top 5
                title = finding.get("title", "Unknown")
                authors = finding.get("authors", [])
                author_str = ", ".join([a.get("name", "") for a in authors[:2]])
                if len(authors) > 2:
                    author_str += " et al."

                draft += f"**{title}** ({finding.get('year', 'n.d.')})\n"
                draft += f"*Authors: {author_str}*\n\n"

                abstract = finding.get("abstract", "")
                if abstract:
                    draft += f"{abstract[:200]}...\n\n"

                draft += f"[Source: {title}]\n\n"

        if web_findings:
            draft += "### Web Sources\n\n"
            for finding in web_findings[:5]:  # Limit to top 5
                title = finding.get("title", "Unknown")
                draft += f"**{title}**\n\n"
                content = finding.get("content", "")
                if content:
                    draft += f"{content[:200]}...\n\n"
                draft += f"[Source: {title}]\n\n"

        # Conclusion
        draft += "## Conclusion\n\n"
        draft += "The research findings presented above address the key aspects of the query. "
        draft += "Multiple sources were consulted to provide a comprehensive response.\n\n"

        draft += "\nDRAFT COMPLETE"

        return draft


# For AutoGen integration
def create_writer_agent(config: Dict[str, Any], model_client) -> 'WriterAgent':
    """
    Create a Writer Agent instance for AutoGen integration.

    Args:
        config: Configuration dictionary
        model_client: AutoGen model client

    Returns:
        WriterAgent instance
    """
    agent_config = config.get("agents", {}).get("writer", {})
    system_prompt = agent_config.get("system_prompt", "").strip()

    return WriterAgent(
        system_prompt=system_prompt,
        config=config,
        model_client=model_client
    )
