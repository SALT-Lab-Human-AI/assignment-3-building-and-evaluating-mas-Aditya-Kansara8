"""
Writer Agent

This agent synthesizes research findings into coherent, well-cited responses.
Enhanced with:
- Thematic organization and argument structure
- Multi-perspective synthesis and gap identification
- Advanced citation management (in-text with page numbers, density tracking)
- Adaptive tone and structure templates
- Coherence checks and readability optimization
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import re
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from src.agents.base_agent import BaseAgent
from src.tools.citation_tool import CitationTool


@dataclass
class CitationInfo:
    """Information about a citation in the text."""
    source_title: str
    page_number: Optional[int] = None
    is_direct_quote: bool = False
    claim: str = ""
    section: str = ""


@dataclass
class Theme:
    """Represents a thematic grouping of findings."""
    name: str
    findings: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    perspectives: List[str] = field(default_factory=list)
    evidence_strength: str = "moderate"  # "strong", "moderate", "weak"


@dataclass
class WritingMetrics:
    """Metrics for writing quality assessment."""
    citation_density: float = 0.0  # Citations per 100 words
    citation_variety: float = 0.0  # Unique sources per section
    coherence_score: float = 0.0
    readability_score: float = 0.0
    sections_with_citations: int = 0
    total_citations: int = 0
    direct_quotes: int = 0
    paraphrases: int = 0


class WriterAgent(BaseAgent):
    """
    Agent responsible for synthesizing research into written responses.

    Enhanced features:
    - Thematic organization: groups findings by themes
    - Argument structure: builds logical arguments from evidence
    - Multi-perspective synthesis: presents contrasting views
    - Gap identification: notes areas with limited evidence
    - Advanced citations: in-text with page numbers, density tracking
    - Adaptive tone: adjusts for audience (academic vs. general)
    - Structure templates: uses templates for different query types
    - Coherence checks: ensures logical flow between sections
    - Readability optimization: adjusts complexity based on audience
    """

    # Structure templates for different query types
    TEMPLATES = {
        "comparison": {
            "sections": ["Introduction", "Overview", "Comparison Framework", "Key Differences", "Similarities", "Conclusion"],
            "description": "For comparing multiple concepts, methods, or approaches"
        },
        "analysis": {
            "sections": ["Introduction", "Background", "Analysis", "Key Findings", "Implications", "Conclusion"],
            "description": "For analyzing a topic in depth"
        },
        "overview": {
            "sections": ["Introduction", "Overview", "Key Concepts", "Current State", "Future Directions", "Conclusion"],
            "description": "For providing a comprehensive overview"
        },
        "evaluation": {
            "sections": ["Introduction", "Criteria", "Evaluation", "Strengths", "Weaknesses", "Conclusion"],
            "description": "For evaluating methods, tools, or approaches"
        },
        "default": {
            "sections": ["Introduction", "Main Findings", "Discussion", "Conclusion"],
            "description": "Default structure for general queries"
        }
    }

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
        self.citations_used: List[CitationInfo] = []
        self.writing_metrics = WritingMetrics()
        self.audience_type = "general"  # "academic" or "general"
        self.query_type = "default"

    def _get_default_prompt(self) -> str:
        """Get default system prompt for the writer."""
        return """You are an expert Research Writer specializing in synthesizing research findings into clear, well-organized, and properly cited responses.

SYNTHESIS REQUIREMENTS:
1. Thematic Organization: Group related findings by themes/topics rather than just listing sources
2. Argument Structure: Build logical arguments from evidence - present claims supported by evidence
3. Multi-Perspective Synthesis: When sources present contrasting views, synthesize them to show different perspectives
4. Gap Identification: Note areas where evidence is limited or conflicting

CITATION REQUIREMENTS:
1. In-text Citations: Use proper APA format (Author, Year, p. X) with page numbers when available
2. Citation Density: Ensure at least 2-3 citations per major claim or section
3. Citation Variety: Use multiple different sources per section (avoid over-reliance on single sources)
4. Proper Attribution: Clearly distinguish direct quotes (use quotation marks) from paraphrases
   - Direct quotes: "text" (Author, Year, p. X)
   - Paraphrases: text (Author, Year)

WRITING QUALITY:
1. Adaptive Tone: Adjust writing style based on audience:
   - Academic: Formal, precise, technical terminology appropriate
   - General: Accessible, clear explanations, minimal jargon
2. Structure: Use appropriate structure templates based on query type
3. Coherence: Ensure logical flow between sections with clear transitions
4. Readability: Adjust sentence complexity and vocabulary based on audience

OUTPUT FORMAT:
- Start with a clear introduction that addresses the query
- Organize content thematically with clear headings
- Use in-text citations throughout (Author, Year, p. X format)
- Include a References section at the end with full citations
- After completing the draft, say "DRAFT COMPLETE"

Remember: Synthesize information from multiple sources to create a coherent narrative, don't just list findings."""

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
            - metadata: Additional writing information including metrics
        """
        self.logger.info("Synthesizing research findings into response...")

        findings = research_findings.get("findings", [])
        sources = research_findings.get("sources", [])

        # Detect audience type and query type
        self.audience_type = self._detect_audience(query, context)
        self.query_type = self._detect_query_type(query)

        # Add sources to citation tool
        for source in sources:
            self.citation_tool.add_citation(source)

        # Organize findings thematically
        themes = self._organize_by_themes(findings, sources)

        # Identify gaps and conflicting perspectives
        gaps = self._identify_gaps(findings, sources)
        perspectives = self._identify_perspectives(findings, sources)

        # Generate draft with enhanced synthesis
        draft = self._synthesize_draft(
            findings=findings,
            sources=sources,
            themes=themes,
            gaps=gaps,
            perspectives=perspectives,
            query=query,
            context=context
        )

        # Perform coherence check
        coherence_issues = self._check_coherence(draft)
        if coherence_issues:
            self.logger.warning(f"Coherence issues detected: {coherence_issues}")

        # Calculate writing metrics
        self._calculate_metrics(draft, sources)

        # Generate bibliography
        bibliography = self.citation_tool.generate_bibliography()

        # Add references section to draft
        if bibliography:
            draft += "\n\n## References\n\n"
            for i, citation in enumerate(bibliography, 1):
                draft += f"{i}. {citation}\n"

        self.logger.info("Draft synthesis complete")
        self.logger.info(f"Writing metrics: {self.writing_metrics}")

        return {
            "draft": draft,
            "citations": bibliography,
            "bibliography": bibliography,
            "metadata": {
                "num_sources": len(sources),
                "num_citations": len(bibliography),
                "writing_metrics": {
                    "citation_density": self.writing_metrics.citation_density,
                    "citation_variety": self.writing_metrics.citation_variety,
                    "coherence_score": self.writing_metrics.coherence_score,
                    "readability_score": self.writing_metrics.readability_score,
                    "sections_with_citations": self.writing_metrics.sections_with_citations,
                    "total_citations": self.writing_metrics.total_citations,
                    "direct_quotes": self.writing_metrics.direct_quotes,
                    "paraphrases": self.writing_metrics.paraphrases,
                },
                "themes_identified": len(themes),
                "gaps_identified": len(gaps),
                "perspectives_identified": len(perspectives),
                "audience_type": self.audience_type,
                "query_type": self.query_type,
                "coherence_issues": coherence_issues
            }
        }

    def _detect_audience(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Detect audience type from query and context."""
        if context and context.get("audience"):
            return context.get("audience", "general")

        # Heuristic: check for academic keywords
        academic_keywords = ["research", "study", "methodology", "hypothesis", "peer-reviewed",
                           "literature review", "empirical", "theoretical framework"]
        query_lower = query.lower()

        if any(keyword in query_lower for keyword in academic_keywords):
            return "academic"

        return "general"

    def _detect_query_type(self, query: str) -> str:
        """Detect query type to select appropriate structure template."""
        query_lower = query.lower()

        comparison_keywords = ["compare", "comparison", "difference", "versus", "vs", "contrast"]
        analysis_keywords = ["analyze", "analysis", "examine", "investigate", "explore"]
        evaluation_keywords = ["evaluate", "assess", "critique", "review", "judge"]
        overview_keywords = ["overview", "summary", "introduction to", "what is", "explain"]

        if any(kw in query_lower for kw in comparison_keywords):
            return "comparison"
        elif any(kw in query_lower for kw in analysis_keywords):
            return "analysis"
        elif any(kw in query_lower for kw in evaluation_keywords):
            return "evaluation"
        elif any(kw in query_lower for kw in overview_keywords):
            return "overview"

        return "default"

    def _organize_by_themes(self, findings: List[Dict[str, Any]], sources: List[Dict[str, Any]]) -> List[Theme]:
        """
        Organize findings by themes/topics.

        Uses keyword extraction and similarity to group related findings.
        """
        themes: Dict[str, Theme] = {}

        # Extract keywords from findings to identify themes
        for finding in findings:
            # Extract potential theme keywords from title, content, or abstract
            title = finding.get("title", "").lower()
            content = finding.get("content", finding.get("abstract", "")).lower()

            # Simple keyword-based theme detection
            # In a production system, this would use NLP/ML for better clustering
            theme_keywords = self._extract_theme_keywords(title + " " + content[:500])

            # Assign to most relevant theme or create new one
            theme_name = self._assign_to_theme(theme_keywords, themes)

            if theme_name not in themes:
                themes[theme_name] = Theme(name=theme_name, findings=[], sources=[])

            themes[theme_name].findings.append(finding)
            source_title = finding.get("title", "Unknown")
            if source_title not in themes[theme_name].sources:
                themes[theme_name].sources.append(source_title)

        # Assess evidence strength for each theme
        for theme in themes.values():
            theme.evidence_strength = self._assess_evidence_strength(theme)

        return list(themes.values())

    def _extract_theme_keywords(self, text: str) -> List[str]:
        """Extract potential theme keywords from text."""
        # Simple keyword extraction - in production, use NLP libraries
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                     "of", "with", "by", "is", "are", "was", "were", "be", "been", "being"}

        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 3]

        # Return most common keywords
        word_freq = Counter(keywords)
        return [word for word, _ in word_freq.most_common(5)]

    def _assign_to_theme(self, keywords: List[str], existing_themes: Dict[str, Theme]) -> str:
        """Assign finding to an existing theme or create a new one."""
        if not existing_themes:
            # Create first theme from keywords
            return keywords[0] if keywords else "General"

        # Find theme with most keyword overlap
        best_theme = None
        best_overlap = 0

        for theme_name, theme in existing_themes.items():
            theme_keywords = self._extract_theme_keywords(
                " ".join([f.get("title", "") for f in theme.findings])
            )
            overlap = len(set(keywords) & set(theme_keywords))
            if overlap > best_overlap:
                best_overlap = overlap
                best_theme = theme_name

        # If good overlap, use existing theme; otherwise create new
        if best_overlap >= 2:
            return best_theme
        else:
            return keywords[0] if keywords else f"Theme_{len(existing_themes) + 1}"

    def _assess_evidence_strength(self, theme: Theme) -> str:
        """Assess the strength of evidence for a theme."""
        num_sources = len(theme.sources)
        num_findings = len(theme.findings)

        if num_sources >= 3 and num_findings >= 5:
            return "strong"
        elif num_sources >= 2 and num_findings >= 3:
            return "moderate"
        else:
            return "weak"

    def _identify_gaps(self, findings: List[Dict[str, Any]], sources: List[Dict[str, Any]]) -> List[str]:
        """
        Identify gaps in the research (areas with limited evidence).

        Returns list of gap descriptions.
        """
        gaps = []

        # Check for temporal gaps (old sources only)
        years = [s.get("year") for s in sources if s.get("year")]
        if years:
            recent_years = [y for y in years if y and y >= 2020]
            if len(recent_years) < len(years) * 0.3:  # Less than 30% recent
                gaps.append("Limited recent research (most sources are older than 2020)")

        # Check for source type diversity
        source_types = [s.get("type", "unknown") for s in sources]
        type_counts = Counter(source_types)
        if len(type_counts) == 1:
            gaps.append(f"Limited source diversity (only {list(type_counts.keys())[0]} sources available)")

        # Check for coverage gaps (few sources overall)
        if len(sources) < 5:
            gaps.append(f"Limited number of sources ({len(sources)} sources) - may not cover all aspects")

        return gaps

    def _identify_perspectives(self, findings: List[Dict[str, Any]], sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify contrasting perspectives or viewpoints in the findings.

        Returns list of perspective dictionaries with descriptions.
        """
        perspectives = []

        # Group findings by potential perspective indicators
        # This is a simplified version - in production, use sentiment analysis or claim extraction

        # Look for contrasting keywords
        positive_indicators = ["support", "effective", "successful", "benefit", "advantage", "improve"]
        negative_indicators = ["challenge", "limitation", "concern", "risk", "problem", "issue"]

        positive_findings = []
        negative_findings = []

        for finding in findings:
            content = (finding.get("content", "") + " " + finding.get("abstract", "")).lower()
            pos_count = sum(1 for ind in positive_indicators if ind in content)
            neg_count = sum(1 for ind in negative_indicators if ind in content)

            if pos_count > neg_count:
                positive_findings.append(finding)
            elif neg_count > pos_count:
                negative_findings.append(finding)

        if positive_findings and negative_findings:
            perspectives.append({
                "type": "contrasting_views",
                "description": "Sources present both positive and negative perspectives",
                "positive_count": len(positive_findings),
                "negative_count": len(negative_findings)
            })

        return perspectives

    def _synthesize_draft(
        self,
        findings: List[Dict[str, Any]],
        sources: List[Dict[str, Any]],
        themes: List[Theme],
        gaps: List[str],
        perspectives: List[Dict[str, Any]],
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Synthesize findings into a coherent draft with enhanced features.
        """
        # Get structure template
        template = self.TEMPLATES.get(self.query_type, self.TEMPLATES["default"])
        sections = template["sections"]

        # Determine tone
        tone_style = self._get_tone_style()

        draft = f"# {query}\n\n"

        # Introduction
        draft += f"## {sections[0]}\n\n"
        intro = self._write_introduction(query, findings, sources, tone_style)
        draft += intro + "\n\n"

        # Main content organized by themes
        for i, section_name in enumerate(sections[1:-1], 1):  # Skip intro and conclusion
            draft += f"## {section_name}\n\n"

            # Write section content based on themes and findings
            section_content = self._write_section(
                section_name=section_name,
                themes=themes,
                findings=findings,
                sources=sources,
                section_index=i,
                tone_style=tone_style
            )
            draft += section_content + "\n\n"

        # Include perspectives if identified
        if perspectives:
            draft += "## Multiple Perspectives\n\n"
            for perspective in perspectives:
                draft += self._write_perspective_section(perspective, sources, tone_style) + "\n\n"

        # Include gaps if identified
        if gaps:
            draft += "## Research Gaps and Limitations\n\n"
            for gap in gaps:
                draft += f"- {gap}\n"
            draft += "\n"

        # Conclusion
        draft += f"## {sections[-1]}\n\n"
        conclusion = self._write_conclusion(query, themes, sources, tone_style)
        draft += conclusion + "\n\n"

        draft += "\nDRAFT COMPLETE"

        return draft

    def _get_tone_style(self) -> Dict[str, Any]:
        """Get writing style parameters based on audience."""
        if self.audience_type == "academic":
            return {
                "formality": "formal",
                "jargon_level": "high",
                "sentence_complexity": "complex",
                "citation_style": "detailed"  # Include page numbers, detailed citations
            }
        else:
            return {
                "formality": "moderate",
                "jargon_level": "low",
                "sentence_complexity": "simple",
                "citation_style": "standard"  # Standard citations, fewer page numbers
            }

    def _write_introduction(
        self,
        query: str,
        findings: List[Dict[str, Any]],
        sources: List[Dict[str, Any]],
        tone_style: Dict[str, Any]
    ) -> str:
        """Write introduction section."""
        intro = f"This response addresses the query: \"{query}\". "

        if tone_style["formality"] == "formal":
            intro += f"Based on a comprehensive review of {len(sources)} sources, this analysis "
            intro += f"synthesizes findings across {len(findings)} research outputs to provide "
            intro += "a structured examination of the topic."
        else:
            intro += f"Based on {len(sources)} sources, this response brings together "
            intro += "key findings to answer your question."

        return intro

    def _write_section(
        self,
        section_name: str,
        themes: List[Theme],
        findings: List[Dict[str, Any]],
        sources: List[Dict[str, Any]],
        section_index: int,
        tone_style: Dict[str, Any]
    ) -> str:
        """Write a section with thematic organization and proper citations."""
        content = ""

        # Select relevant themes for this section
        relevant_themes = self._select_themes_for_section(section_name, themes)

        # Write content for each theme
        for theme in relevant_themes:
            content += f"### {theme.name.title()}\n\n"

            # Build argument from evidence
            argument = self._build_argument_from_evidence(theme, tone_style)
            content += argument + "\n\n"

        # If no specific themes, use general findings
        if not relevant_themes and findings:
            content += self._write_general_findings(findings[:3], sources, tone_style)

        return content

    def _select_themes_for_section(self, section_name: str, themes: List[Theme]) -> List[Theme]:
        """Select which themes are relevant for a given section."""
        # Simple heuristic: distribute themes across sections
        # In production, use semantic matching
        section_keywords = {
            "Overview": ["overview", "introduction", "background"],
            "Key Differences": ["difference", "contrast", "comparison"],
            "Similarities": ["similar", "common", "shared"],
            "Analysis": ["analysis", "examine", "investigate"],
            "Key Findings": ["finding", "result", "outcome"],
            "Implications": ["implication", "impact", "consequence"],
            "Strengths": ["strength", "advantage", "benefit"],
            "Weaknesses": ["weakness", "limitation", "challenge"]
        }

        # For now, return all themes (can be enhanced with better matching)
        return themes[:3]  # Limit to top 3 themes per section

    def _build_argument_from_evidence(self, theme: Theme, tone_style: Dict[str, Any]) -> str:
        """
        Build a logical argument from evidence in a theme.

        Structure: Claim -> Evidence -> Support -> Conclusion
        """
        argument = ""

        # Extract main claim from theme findings
        if not theme.findings:
            return argument

        # Use first finding as primary claim
        primary_finding = theme.findings[0]
        claim = primary_finding.get("title", "") or primary_finding.get("content", "")[:100]

        # Write claim
        if tone_style["formality"] == "formal":
            argument += f"Research indicates that {claim.lower()}."
        else:
            argument += f"{claim}."

        # Add evidence with citations
        cited_sources = set()
        for finding in theme.findings[:3]:  # Use top 3 findings as evidence
            source_title = finding.get("title", "Unknown")
            source = self.citation_tool.get_source_by_title(source_title)

            if source:
                # Get citation
                authors = source.get("authors", [])
                year = source.get("year", "n.d.")

                if authors:
                    author_str = self._format_author_for_citation(authors[0].get("name", ""))
                    if len(authors) > 1:
                        author_str += " et al."
                else:
                    author_str = source_title[:30]

                # Add in-text citation
                citation = f"({author_str}, {year})"
                page_num = finding.get("page")
                if tone_style["citation_style"] == "detailed" and page_num:
                    # Convert page to int if it's a string
                    try:
                        page_int = int(page_num) if isinstance(page_num, str) else page_num
                        citation = f"({author_str}, {year}, p. {page_int})"
                    except (ValueError, TypeError):
                        pass  # Keep citation without page number

                # Extract evidence text
                evidence_text = finding.get("content", finding.get("abstract", ""))[:200]
                if evidence_text:
                    argument += f" {evidence_text} {citation}."
                    cited_sources.add(source_title)

                    # Track citation - convert page to int if possible
                    page_number = None
                    if page_num:
                        try:
                            page_number = int(page_num) if isinstance(page_num, str) else page_num
                        except (ValueError, TypeError):
                            pass

                    self.citations_used.append(CitationInfo(
                        source_title=source_title,
                        page_number=page_number,
                        is_direct_quote=False,
                        claim=claim[:50],
                        section=theme.name
                    ))

        # Note evidence strength
        if theme.evidence_strength == "weak":
            argument += " However, the evidence for this is limited."
        elif theme.evidence_strength == "strong":
            argument += " Multiple sources support this finding."

        return argument

    def _format_author_for_citation(self, name: str) -> str:
        """Format author name for in-text citation (Last, F.)."""
        if not name or name == "Unknown":
            return "Unknown"

        if ',' in name:
            return name.split(',')[0]  # Already formatted

        parts = name.strip().split()
        if len(parts) == 1:
            return parts[0]

        return parts[-1]  # Return last name

    def _write_general_findings(
        self,
        findings: List[Dict[str, Any]],
        sources: List[Dict[str, Any]],
        tone_style: Dict[str, Any]
    ) -> str:
        """Write general findings when no specific themes apply."""
        content = ""

        for finding in findings:
            title = finding.get("title", "Unknown")
            source = self.citation_tool.get_source_by_title(title)

            if source:
                authors = source.get("authors", [])
                year = source.get("year", "n.d.")

                if authors:
                    author_str = self._format_author_for_citation(authors[0].get("name", ""))
                else:
                    author_str = title[:30]

                citation = f"({author_str}, {year})"

                finding_text = finding.get("content", finding.get("abstract", ""))[:300]
                if finding_text:
                    content += f"{finding_text} {citation}.\n\n"

        return content

    def _write_perspective_section(
        self,
        perspective: Dict[str, Any],
        sources: List[Dict[str, Any]],
        tone_style: Dict[str, Any]
    ) -> str:
        """Write section presenting multiple perspectives."""
        content = perspective.get("description", "Multiple perspectives are presented in the literature.")

        if perspective.get("type") == "contrasting_views":
            content += f" Some sources ({perspective.get('positive_count', 0)}) present positive views, "
            content += f"while others ({perspective.get('negative_count', 0)}) highlight challenges or limitations."

        return content

    def _write_conclusion(
        self,
        query: str,
        themes: List[Theme],
        sources: List[Dict[str, Any]],
        tone_style: Dict[str, Any]
    ) -> str:
        """Write conclusion section."""
        if tone_style["formality"] == "formal":
            conclusion = f"In conclusion, this analysis of {len(sources)} sources has synthesized "
            conclusion += f"findings across {len(themes)} key themes to address the query: \"{query}\". "
            conclusion += "The evidence presented supports a comprehensive understanding of the topic."
        else:
            conclusion = f"To summarize, based on {len(sources)} sources, we've covered the main aspects "
            conclusion += f"of \"{query}\". The findings show several important themes and perspectives."

        return conclusion

    def _check_coherence(self, draft: str) -> List[str]:
        """
        Check coherence of the draft.

        Returns list of coherence issues if any.
        """
        issues = []

        # Check for section transitions
        sections = re.findall(r'##\s+(.+?)\n', draft)
        if len(sections) < 3:
            issues.append("Insufficient sections - draft may lack structure")

        # Check for logical flow indicators
        transition_words = ["however", "furthermore", "moreover", "therefore", "consequently", "in addition"]
        transition_count = sum(1 for word in transition_words if word.lower() in draft.lower())
        if transition_count < 2:
            issues.append("Limited use of transition words - may affect flow")

        # Check for citation distribution
        citation_pattern = r'\([^)]+,\s*\d{4}'
        citations = re.findall(citation_pattern, draft)
        if len(citations) < 5:
            issues.append("Low citation density - may need more in-text citations")

        return issues

    def _calculate_metrics(self, draft: str, sources: List[Dict[str, Any]]):
        """Calculate writing quality metrics."""
        # Count words
        words = re.findall(r'\b\w+\b', draft)
        word_count = len(words)

        # Count citations
        citation_pattern = r'\([^)]+,\s*\d{4}'
        citations = re.findall(citation_pattern, draft)
        self.writing_metrics.total_citations = len(citations)

        # Citation density (citations per 100 words)
        if word_count > 0:
            self.writing_metrics.citation_density = (len(citations) / word_count) * 100

        # Citation variety (unique sources cited)
        unique_sources = set()
        for citation_info in self.citations_used:
            unique_sources.add(citation_info.source_title)
        self.writing_metrics.citation_variety = len(unique_sources)

        # Count direct quotes vs paraphrases
        direct_quotes = len(re.findall(r'"[^"]+"', draft))
        self.writing_metrics.direct_quotes = direct_quotes
        self.writing_metrics.paraphrases = len(citations) - direct_quotes

        # Count sections with citations
        sections = draft.split("##")
        sections_with_citations = sum(1 for section in sections if re.search(citation_pattern, section))
        self.writing_metrics.sections_with_citations = sections_with_citations

        # Simple coherence score (based on transitions and structure)
        transition_words = ["however", "furthermore", "moreover", "therefore", "consequently"]
        transitions = sum(1 for word in transition_words if word.lower() in draft.lower())
        self.writing_metrics.coherence_score = min(1.0, transitions / 5.0)

        # Simple readability score (based on sentence length and complexity)
        sentences = re.split(r'[.!?]+', draft)
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            # Lower is better for readability (simplified metric)
            self.writing_metrics.readability_score = max(0.0, 1.0 - (avg_sentence_length / 30.0))


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
