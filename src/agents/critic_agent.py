"""
Critic Agent

This agent evaluates the quality and accuracy of research outputs.
It provides feedback for improvement and determines if the work meets quality standards.

Enhanced features:
- Multi-dimensional scoring with expanded criteria
- Fact-checking against sources
- Consistency checks for contradictions
- Completeness analysis
- Actionable feedback with priority ranking
- Positive reinforcement
- Revision guidance
- Adaptive quality thresholds
- Domain-specific criteria (HCI)
- Citation quality assessment
- Argument strength evaluation
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import json
import re
import os
import asyncio
from dataclasses import dataclass, field
from collections import defaultdict

from src.agents.base_agent import BaseAgent

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


@dataclass
class FeedbackItem:
    """Structured feedback item with priority and type."""
    issue: str
    priority: int  # 1=critical, 2=high, 3=medium, 4=low
    category: str  # e.g., "factual_accuracy", "completeness", "citation_quality"
    suggestion: str
    location: Optional[str] = None  # Where in the draft this applies
    positive: bool = False  # True if this is positive reinforcement


@dataclass
class EvaluationResult:
    """Comprehensive evaluation result."""
    scores: Dict[str, float]
    overall_score: float
    approved: bool
    feedback_items: List[FeedbackItem]
    strengths: List[str]
    weaknesses: List[str]
    fact_check_results: List[Dict[str, Any]]
    consistency_issues: List[str]
    completeness_gaps: List[str]
    citation_quality: Dict[str, Any]
    argument_strength: float
    adaptive_threshold: float
    domain_specific_scores: Dict[str, float]


class CriticAgent(BaseAgent):
    """
    Enhanced Critic Agent for quality evaluation and verification.

    The critic evaluates:
    - Relevance to the original query
    - Quality of evidence and sources
    - Completeness of coverage
    - Factual accuracy (with fact-checking)
    - Consistency (contradiction detection)
    - Clarity and organization
    - Citation quality
    - Argument strength
    - Domain-specific criteria (HCI)
    """

    # HCI-specific quality criteria
    HCI_CRITERIA = {
        "user_centered_design": "Emphasis on user needs and perspectives",
        "usability_principles": "Application of usability heuristics and principles",
        "empirical_evidence": "Use of user studies, experiments, or empirical data",
        "accessibility": "Consideration of accessibility and inclusive design",
        "design_methodology": "Clear description of design process and methods",
        "evaluation_methods": "Appropriate evaluation methods (usability testing, etc.)",
        "theoretical_foundation": "Grounding in HCI theory and frameworks"
    }

    def __init__(
        self,
        system_prompt: str = "",
        config: Optional[Dict[str, Any]] = None,
        model_client = None
    ):
        """
        Initialize the Enhanced Critic Agent.

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
        self.config = config or {}

        # Initialize LLM client for advanced evaluation
        self._init_llm_client()

        # Domain from config (default: HCI)
        self.domain = self.config.get("system", {}).get("topic", "HCI Research").lower()
        self.is_hci_domain = "hci" in self.domain or "human-computer" in self.domain

    def _init_llm_client(self):
        """Initialize LLM client for advanced evaluation features."""
        model_config = self.config.get("models", {}).get("judge", {})
        self.provider = model_config.get("provider", "openai").lower()
        self.backup_provider = model_config.get("backup_provider", "groq").lower()

        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key and OpenAI:
                self.llm_client = OpenAI(api_key=api_key)
                self.client_type = "openai"
                return
            else:
                self.provider = self.backup_provider

        if self.provider == "groq" or self.provider == self.backup_provider:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key and Groq:
                self.llm_client = Groq(api_key=api_key)
                self.client_type = "groq"
                return

        self.llm_client = None
        self.client_type = None
        self.logger.warning("LLM client not initialized. Advanced features will use fallback methods.")

    def _get_default_prompt(self) -> str:
        """Get default system prompt for the critic."""
        return """You are an expert Research Critic specializing in quality evaluation and verification.

Your role is to evaluate research outputs comprehensively across multiple dimensions:

EVALUATION CRITERIA:
1. **Relevance**: Does it fully answer the original query?
2. **Evidence Quality**: Are sources credible, authoritative, and well-cited?
3. **Completeness**: Are all aspects of the query addressed? Any missing elements?
4. **Factual Accuracy**: Are claims accurate and verifiable against sources?
5. **Consistency**: Are there contradictions or conflicting information?
6. **Clarity**: Is the writing clear, well-organized, and accessible?
7. **Citation Quality**: Are citations properly formatted, relevant, and credible?
8. **Argument Strength**: Is the reasoning logical and well-supported?

Provide detailed, actionable feedback with:
- Specific improvement suggestions
- Priority ranking of issues (critical, high, medium, low)
- Positive reinforcement for strengths
- Concrete revision guidance

End your evaluation with either "APPROVED - RESEARCH COMPLETE" if approved, or "NEEDS REVISION" if improvements are needed."""

    def process(
        self,
        draft: str,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a research draft and provide comprehensive feedback.

        Args:
            draft: Written draft to evaluate
            query: Original research query
            context: Optional context (sources, plan, etc.)

        Returns:
            Dictionary with comprehensive evaluation results
        """
        self.logger.info("Starting comprehensive evaluation of research draft...")

        # Run comprehensive evaluation
        try:
            # Try async evaluation if LLM is available
            if self.llm_client:
                result = asyncio.run(self._evaluate_comprehensive_async(draft, query, context))
            else:
                result = self._evaluate_comprehensive_sync(draft, query, context)
        except Exception as e:
            self.logger.error(f"Error in comprehensive evaluation: {e}")
            # Fallback to basic evaluation
            result = self._evaluate_basic(draft, query, context)

        self.logger.info(f"Evaluation complete. Approved: {result.approved}, Score: {result.overall_score:.2f}")

        return {
            "evaluation": self._format_evaluation(result),
            "scores": result.scores,
            "approved": result.approved,
            "feedback": self._format_feedback_items(result.feedback_items),
            "strengths": result.strengths,
            "weaknesses": result.weaknesses,
            "fact_check_results": result.fact_check_results,
            "consistency_issues": result.consistency_issues,
            "completeness_gaps": result.completeness_gaps,
            "citation_quality": result.citation_quality,
            "argument_strength": result.argument_strength,
            "domain_specific_scores": result.domain_specific_scores,
            "metadata": {
                "overall_score": result.overall_score,
                "adaptive_threshold": result.adaptive_threshold,
                "criteria_evaluated": list(result.scores.keys()),
                "domain": self.domain
            }
        }

    async def _evaluate_comprehensive_async(
        self,
        draft: str,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> EvaluationResult:
        """Comprehensive async evaluation using LLM."""
        # Extract sources from context
        sources = self._extract_sources(context)

        # Calculate query complexity for adaptive thresholds
        query_complexity = self._assess_query_complexity(query)
        adaptive_threshold = self._calculate_adaptive_threshold(query_complexity)

        # Multi-dimensional scoring
        scores = await self._evaluate_multi_dimensional(draft, query, sources, context)

        # Fact-checking
        fact_check_results = await self._fact_check_claims(draft, sources)

        # Consistency checks
        consistency_issues = await self._check_consistency(draft, query)

        # Completeness analysis
        completeness_gaps = await self._analyze_completeness(draft, query, context)

        # Citation quality assessment
        citation_quality = await self._assess_citation_quality(draft, sources)

        # Argument strength evaluation
        argument_strength = await self._evaluate_argument_strength(draft, query)

        # Domain-specific evaluation (HCI)
        domain_specific_scores = {}
        if self.is_hci_domain:
            domain_specific_scores = await self._evaluate_hci_criteria(draft, query)

        # Calculate overall score (weighted)
        overall_score = self._calculate_weighted_score(scores, domain_specific_scores)

        # Generate actionable feedback
        feedback_items = await self._generate_actionable_feedback(
            scores, draft, query, fact_check_results, consistency_issues,
            completeness_gaps, citation_quality, argument_strength, domain_specific_scores
        )

        # Identify strengths and weaknesses
        strengths, weaknesses = self._identify_strengths_weaknesses(scores, feedback_items)

        # Determine approval
        approved = overall_score >= adaptive_threshold

        return EvaluationResult(
            scores=scores,
            overall_score=overall_score,
            approved=approved,
            feedback_items=feedback_items,
            strengths=strengths,
            weaknesses=weaknesses,
            fact_check_results=fact_check_results,
            consistency_issues=consistency_issues,
            completeness_gaps=completeness_gaps,
            citation_quality=citation_quality,
            argument_strength=argument_strength,
            adaptive_threshold=adaptive_threshold,
            domain_specific_scores=domain_specific_scores
        )

    def _evaluate_comprehensive_sync(
        self,
        draft: str,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> EvaluationResult:
        """Comprehensive sync evaluation (fallback without LLM)."""
        sources = self._extract_sources(context)
        query_complexity = self._assess_query_complexity(query)
        adaptive_threshold = self._calculate_adaptive_threshold(query_complexity)

        # Use enhanced rule-based evaluation
        scores = self._evaluate_criteria_enhanced(draft, query, context)

        # Basic fact-checking (rule-based)
        fact_check_results = self._fact_check_claims_basic(draft, sources)

        # Basic consistency checks
        consistency_issues = self._check_consistency_basic(draft)

        # Basic completeness analysis
        completeness_gaps = self._analyze_completeness_basic(draft, query, context)

        # Basic citation quality
        citation_quality = self._assess_citation_quality_basic(draft, sources)

        # Basic argument strength
        argument_strength = self._evaluate_argument_strength_basic(draft)

        # Domain-specific (basic)
        domain_specific_scores = {}
        if self.is_hci_domain:
            domain_specific_scores = self._evaluate_hci_criteria_basic(draft)

        overall_score = self._calculate_weighted_score(scores, domain_specific_scores)

        feedback_items = self._generate_feedback_enhanced(
            scores, draft, query, fact_check_results, consistency_issues,
            completeness_gaps, citation_quality, argument_strength, domain_specific_scores
        )

        strengths, weaknesses = self._identify_strengths_weaknesses(scores, feedback_items)
        approved = overall_score >= adaptive_threshold

        return EvaluationResult(
            scores=scores,
            overall_score=overall_score,
            approved=approved,
            feedback_items=feedback_items,
            strengths=strengths,
            weaknesses=weaknesses,
            fact_check_results=fact_check_results,
            consistency_issues=consistency_issues,
            completeness_gaps=completeness_gaps,
            citation_quality=citation_quality,
            argument_strength=argument_strength,
            adaptive_threshold=adaptive_threshold,
            domain_specific_scores=domain_specific_scores
        )

    def _evaluate_basic(
        self,
        draft: str,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> EvaluationResult:
        """Basic fallback evaluation."""
        scores = self._evaluate_criteria(draft, query, context)
        overall_score = sum(scores.values()) / len(scores) if scores else 0.0
        approved = overall_score >= 0.7

        feedback_items = [
            FeedbackItem(
                issue="Basic evaluation completed",
                priority=4,
                category="general",
                suggestion="Consider using LLM-based evaluation for more detailed feedback",
                positive=False
            )
        ]

        return EvaluationResult(
            scores=scores,
            overall_score=overall_score,
            approved=approved,
            feedback_items=feedback_items,
            strengths=[],
            weaknesses=[],
            fact_check_results=[],
            consistency_issues=[],
            completeness_gaps=[],
            citation_quality={},
            argument_strength=0.5,
            adaptive_threshold=0.7,
            domain_specific_scores={}
        )

    # ==================== Multi-dimensional Scoring ====================

    async def _evaluate_multi_dimensional(
        self,
        draft: str,
        query: str,
        sources: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Evaluate on multiple dimensions using LLM."""
        if not self.llm_client:
            return self._evaluate_criteria_enhanced(draft, query, context)

        prompt = f"""Evaluate this research draft across multiple dimensions. Provide scores (0.0-1.0) for each criterion.

QUERY: {query}

DRAFT:
{draft[:3000]}  # Truncate if too long

SOURCES: {len(sources)} sources provided

Evaluate and provide JSON with scores for:
- relevance: How well does it answer the query?
- evidence_quality: Quality and credibility of sources
- completeness: Are all aspects covered?
- factual_accuracy: Accuracy of claims
- consistency: Internal consistency (no contradictions)
- clarity: Writing clarity and organization
- citation_quality: Proper citation formatting and relevance
- argument_strength: Logical reasoning and support

Output JSON only:
{{
    "relevance": <float>,
    "evidence_quality": <float>,
    "completeness": <float>,
    "factual_accuracy": <float>,
    "consistency": <float>,
    "clarity": <float>,
    "citation_quality": <float>,
    "argument_strength": <float>
}}"""

        try:
            response = await self._call_llm(prompt)
            result = json.loads(self._extract_json(response))
            return {k: float(v) for k, v in result.items() if k in [
                "relevance", "evidence_quality", "completeness", "factual_accuracy",
                "consistency", "clarity", "citation_quality", "argument_strength"
            ]}
        except Exception as e:
            self.logger.error(f"Error in multi-dimensional evaluation: {e}")
            return self._evaluate_criteria_enhanced(draft, query, context)

    def _evaluate_criteria_enhanced(
        self,
        draft: str,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Enhanced rule-based evaluation."""
        scores = {}
        draft_lower = draft.lower()
        query_lower = query.lower()

        # Relevance (enhanced)
        query_terms = set(re.findall(r'\b\w+\b', query_lower))
        draft_terms = set(re.findall(r'\b\w+\b', draft_lower))
        relevant_terms = len(query_terms & draft_terms)
        scores["relevance"] = min(1.0, relevant_terms / max(1, len(query_terms)) * 1.2)

        # Evidence Quality (enhanced)
        citation_patterns = [
            r'\[Source:', r'\([A-Z][a-z]+.*?\d{4}\)', r'\[.*?\]', r'References?',
            r'doi\.org', r'https?://', r'\(.*?et al\.'
        ]
        citation_count = sum(len(re.findall(pattern, draft)) for pattern in citation_patterns)
        scores["evidence_quality"] = min(1.0, citation_count / 8.0)

        # Completeness (enhanced)
        has_intro = any(word in draft_lower for word in ["introduction", "overview", "background"])
        has_findings = any(word in draft_lower for word in ["finding", "result", "study", "research"])
        has_conclusion = any(word in draft_lower for word in ["conclusion", "summary", "discussion"])
        structure_score = (has_intro + has_findings + has_conclusion) / 3.0
        word_count = len(draft.split())
        length_score = min(1.0, word_count / 600.0)
        scores["completeness"] = (structure_score * 0.6 + length_score * 0.4)

        # Factual Accuracy (basic proxy)
        scores["factual_accuracy"] = scores["evidence_quality"] * 0.85

        # Consistency (basic check for contradictory phrases)
        contradiction_indicators = [
            ("however", "but", "although"), ("some", "others"), ("pro", "con"),
            ("support", "oppose"), ("increase", "decrease")
        ]
        contradiction_score = 1.0
        for group in contradiction_indicators:
            found = sum(1 for word in group if word in draft_lower)
            if found > 1:
                contradiction_score -= 0.1
        scores["consistency"] = max(0.0, contradiction_score)

        # Clarity
        has_headings = draft.count("#") > 0 or any(h in draft_lower for h in ["##", "###"])
        has_paragraphs = draft.count("\n\n") > 3
        has_lists = draft.count("-") > 2 or draft.count("*") > 2
        clarity_score = (has_headings + has_paragraphs + has_lists) / 3.0
        scores["clarity"] = clarity_score

        # Citation Quality (basic)
        proper_citation = bool(re.search(r'\([A-Z][a-z]+.*?\d{4}\)', draft))
        has_references = "reference" in draft_lower or "citation" in draft_lower
        scores["citation_quality"] = (proper_citation + has_references) / 2.0

        # Argument Strength (basic)
        has_because = "because" in draft_lower or "since" in draft_lower or "due to" in draft_lower
        has_evidence = "evidence" in draft_lower or "study" in draft_lower or "research" in draft_lower
        has_therefore = "therefore" in draft_lower or "thus" in draft_lower or "consequently" in draft_lower
        argument_score = (has_because + has_evidence + has_therefore) / 3.0
        scores["argument_strength"] = argument_score

        return scores

    # ==================== Fact-checking ====================

    async def _fact_check_claims(
        self,
        draft: str,
        sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fact-check claims against sources using LLM."""
        if not self.llm_client or not sources:
            return self._fact_check_claims_basic(draft, sources)

        # Extract factual claims from draft
        prompt = f"""Extract factual claims from this research draft that should be verified against sources.

DRAFT:
{draft[:2000]}

SOURCES:
{json.dumps(sources[:5], indent=2)[:1500]}

For each claim, verify if it's supported by the sources. Output JSON:
{{
    "claims": [
        {{
            "claim": "<the factual claim>",
            "verifiable": true/false,
            "supported": true/false,
            "source_match": "<which source supports it, if any>",
            "confidence": <float 0-1>
        }}
    ]
}}"""

        try:
            response = await self._call_llm(prompt)
            result = json.loads(self._extract_json(response))
            return result.get("claims", [])
        except Exception as e:
            self.logger.error(f"Error in fact-checking: {e}")
            return self._fact_check_claims_basic(draft, sources)

    def _fact_check_claims_basic(
        self,
        draft: str,
        sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Basic fact-checking (rule-based)."""
        results = []

        # Extract sentences that look like factual claims
        sentences = re.split(r'[.!?]+', draft)
        for sentence in sentences[:20]:  # Limit to first 20 sentences
            if len(sentence.strip()) > 20 and any(word in sentence.lower() for word in ["is", "are", "was", "were", "shows", "found", "indicates"]):
                results.append({
                    "claim": sentence.strip(),
                    "verifiable": True,
                    "supported": len(sources) > 0,  # Basic: assume supported if sources exist
                    "source_match": sources[0].get("title", "") if sources else "",
                    "confidence": 0.5 if sources else 0.2
                })

        return results

    # ==================== Consistency Checks ====================

    async def _check_consistency(
        self,
        draft: str,
        query: str
    ) -> List[str]:
        """Check for contradictions using LLM."""
        if not self.llm_client:
            return self._check_consistency_basic(draft)

        prompt = f"""Analyze this research draft for contradictions, inconsistencies, or conflicting statements.

DRAFT:
{draft[:2500]}

Identify any contradictions or inconsistencies. Output JSON:
{{
    "issues": [
        {{
            "issue": "<description of the contradiction>",
            "severity": "high/medium/low",
            "location": "<where in the text>"
        }}
    ]
}}"""

        try:
            response = await self._call_llm(prompt)
            result = json.loads(self._extract_json(response))
            return [item.get("issue", "") for item in result.get("issues", [])]
        except Exception as e:
            self.logger.error(f"Error in consistency check: {e}")
            return self._check_consistency_basic(draft)

    def _check_consistency_basic(self, draft: str) -> List[str]:
        """Basic consistency check."""
        issues = []
        draft_lower = draft.lower()

        # Check for contradictory pairs
        contradictions = [
            ("increase", "decrease"), ("support", "oppose"), ("prove", "disprove"),
            ("effective", "ineffective"), ("better", "worse"), ("positive", "negative")
        ]

        for word1, word2 in contradictions:
            if word1 in draft_lower and word2 in draft_lower:
                issues.append(f"Potential contradiction: '{word1}' and '{word2}' both mentioned")

        return issues

    # ==================== Completeness Analysis ====================

    async def _analyze_completeness(
        self,
        draft: str,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Analyze completeness using LLM."""
        if not self.llm_client:
            return self._analyze_completeness_basic(draft, query, context)

        research_plan = context.get("research_plan", "") if context else ""

        prompt = f"""Analyze what aspects of the query might be missing from this research draft.

QUERY: {query}

RESEARCH PLAN (if available):
{research_plan[:1000]}

DRAFT:
{draft[:2000]}

Identify missing aspects or gaps. Output JSON:
{{
    "gaps": [
        "<description of missing aspect>"
    ]
}}"""

        try:
            response = await self._call_llm(prompt)
            result = json.loads(self._extract_json(response))
            return result.get("gaps", [])
        except Exception as e:
            self.logger.error(f"Error in completeness analysis: {e}")
            return self._analyze_completeness_basic(draft, query, context)

    def _analyze_completeness_basic(
        self,
        draft: str,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Basic completeness analysis."""
        gaps = []
        draft_lower = draft.lower()
        query_lower = query.lower()

        # Extract key terms from query
        query_terms = set(re.findall(r'\b\w{4,}\b', query_lower))  # Words 4+ chars
        draft_terms = set(re.findall(r'\b\w{4,}\b', draft_lower))

        missing_terms = query_terms - draft_terms
        if missing_terms:
            gaps.append(f"Query terms not found in draft: {', '.join(list(missing_terms)[:5])}")

        # Check for expected sections
        if "compare" in query_lower or "difference" in query_lower:
            if "comparison" not in draft_lower and "difference" not in draft_lower:
                gaps.append("Comparison query but no comparison section found")

        if "how" in query_lower:
            if "method" not in draft_lower and "process" not in draft_lower:
                gaps.append("'How' query but methodology/process not clearly explained")

        return gaps

    # ==================== Citation Quality Assessment ====================

    async def _assess_citation_quality(
        self,
        draft: str,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess citation quality using LLM."""
        if not self.llm_client:
            return self._assess_citation_quality_basic(draft, sources)

        prompt = f"""Evaluate the quality of citations in this research draft.

DRAFT:
{draft[:2000]}

SOURCES:
{len(sources)} sources available

Assess:
1. Citation formatting (proper APA/MLA style)
2. Citation relevance to claims
3. Source credibility
4. Citation density (appropriate number)

Output JSON:
{{
    "formatting_score": <float 0-1>,
    "relevance_score": <float 0-1>,
    "credibility_score": <float 0-1>,
    "density_score": <float 0-1>,
    "issues": ["<list of citation issues>"]
}}"""

        try:
            response = await self._call_llm(prompt)
            result = json.loads(self._extract_json(response))
            return result
        except Exception as e:
            self.logger.error(f"Error in citation quality assessment: {e}")
            return self._assess_citation_quality_basic(draft, sources)

    def _assess_citation_quality_basic(
        self,
        draft: str,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Basic citation quality assessment."""
        # Count citations
        citation_count = len(re.findall(r'\([A-Z][a-z]+.*?\d{4}\)', draft))
        citation_count += len(re.findall(r'\[Source:', draft))

        # Check formatting
        has_proper_format = bool(re.search(r'\([A-Z][a-z]+.*?\d{4}\)', draft))

        # Density (citations per 100 words)
        word_count = len(draft.split())
        density = (citation_count / word_count * 100) if word_count > 0 else 0
        ideal_density = 2.0  # 2 citations per 100 words
        density_score = min(1.0, density / ideal_density)

        return {
            "formatting_score": 1.0 if has_proper_format else 0.5,
            "relevance_score": 0.7 if sources else 0.3,
            "credibility_score": 0.8 if sources else 0.2,
            "density_score": density_score,
            "issues": [] if citation_count >= 3 else ["Insufficient citations"]
        }

    # ==================== Argument Strength Evaluation ====================

    async def _evaluate_argument_strength(
        self,
        draft: str,
        query: str
    ) -> float:
        """Evaluate argument strength using LLM."""
        if not self.llm_client:
            return self._evaluate_argument_strength_basic(draft)

        prompt = f"""Evaluate the logical reasoning and argument strength in this research draft.

QUERY: {query}

DRAFT:
{draft[:2000]}

Assess:
- Logical flow and reasoning
- Use of evidence to support claims
- Clear cause-effect relationships
- Sound conclusions

Output JSON:
{{
    "strength_score": <float 0-1>,
    "reasoning": "<explanation>"
}}"""

        try:
            response = await self._call_llm(prompt)
            result = json.loads(self._extract_json(response))
            return float(result.get("strength_score", 0.5))
        except Exception as e:
            self.logger.error(f"Error in argument strength evaluation: {e}")
            return self._evaluate_argument_strength_basic(draft)

    def _evaluate_argument_strength_basic(self, draft: str) -> float:
        """Basic argument strength evaluation."""
        draft_lower = draft.lower()

        # Indicators of strong arguments
        has_because = "because" in draft_lower or "since" in draft_lower
        has_evidence = "evidence" in draft_lower or "study" in draft_lower
        has_therefore = "therefore" in draft_lower or "thus" in draft_lower
        has_support = "support" in draft_lower or "indicates" in draft_lower
        has_conclusion = "conclusion" in draft_lower or "summary" in draft_lower

        indicators = sum([has_because, has_evidence, has_therefore, has_support, has_conclusion])
        return min(1.0, indicators / 5.0)

    # ==================== Domain-Specific Evaluation (HCI) ====================

    async def _evaluate_hci_criteria(
        self,
        draft: str,
        query: str
    ) -> Dict[str, float]:
        """Evaluate HCI-specific criteria using LLM."""
        if not self.llm_client:
            return self._evaluate_hci_criteria_basic(draft)

        prompt = f"""Evaluate this research draft against HCI (Human-Computer Interaction) quality criteria.

DRAFT:
{draft[:2000]}

Evaluate on these HCI criteria:
1. User-centered design emphasis
2. Usability principles application
3. Empirical evidence (user studies, experiments)
4. Accessibility considerations
5. Design methodology clarity
6. Evaluation methods appropriateness
7. Theoretical foundation in HCI

Output JSON with scores (0-1) for each:
{{
    "user_centered_design": <float>,
    "usability_principles": <float>,
    "empirical_evidence": <float>,
    "accessibility": <float>,
    "design_methodology": <float>,
    "evaluation_methods": <float>,
    "theoretical_foundation": <float>
}}"""

        try:
            response = await self._call_llm(prompt)
            result = json.loads(self._extract_json(response))
            return {k: float(v) for k, v in result.items() if k in self.HCI_CRITERIA.keys()}
        except Exception as e:
            self.logger.error(f"Error in HCI criteria evaluation: {e}")
            return self._evaluate_hci_criteria_basic(draft)

    def _evaluate_hci_criteria_basic(self, draft: str) -> Dict[str, float]:
        """Basic HCI criteria evaluation."""
        draft_lower = draft.lower()
        scores = {}

        # User-centered design
        ucd_terms = ["user", "participant", "user-centered", "user experience", "ux"]
        scores["user_centered_design"] = min(1.0, sum(1 for term in ucd_terms if term in draft_lower) / 3.0)

        # Usability principles
        usability_terms = ["usability", "heuristic", "nielsen", "shneiderman", "affordance"]
        scores["usability_principles"] = min(1.0, sum(1 for term in usability_terms if term in draft_lower) / 2.0)

        # Empirical evidence
        empirical_terms = ["study", "experiment", "evaluation", "test", "participant", "user study"]
        scores["empirical_evidence"] = min(1.0, sum(1 for term in empirical_terms if term in draft_lower) / 3.0)

        # Accessibility
        accessibility_terms = ["accessibility", "inclusive", "disability", "a11y", "wcag"]
        scores["accessibility"] = min(1.0, sum(1 for term in accessibility_terms if term in draft_lower) / 2.0)

        # Design methodology
        method_terms = ["method", "methodology", "process", "design process", "approach"]
        scores["design_methodology"] = min(1.0, sum(1 for term in method_terms if term in draft_lower) / 2.0)

        # Evaluation methods
        eval_terms = ["usability testing", "evaluation", "heuristic evaluation", "user testing"]
        scores["evaluation_methods"] = min(1.0, sum(1 for term in eval_terms if term in draft_lower) / 2.0)

        # Theoretical foundation
        theory_terms = ["theory", "framework", "model", "principle", "concept"]
        scores["theoretical_foundation"] = min(1.0, sum(1 for term in theory_terms if term in draft_lower) / 2.0)

        return scores

    # ==================== Adaptive Thresholds ====================

    def _assess_query_complexity(self, query: str) -> float:
        """Assess query complexity (0.0 = simple, 1.0 = very complex)."""
        complexity_score = 0.0

        # Length factor
        word_count = len(query.split())
        complexity_score += min(0.3, word_count / 50.0)

        # Question type indicators
        complex_indicators = ["compare", "analyze", "evaluate", "synthesize", "multiple", "various"]
        simple_indicators = ["what", "who", "when", "where"]

        for indicator in complex_indicators:
            if indicator in query.lower():
                complexity_score += 0.2

        for indicator in simple_indicators:
            if indicator in query.lower():
                complexity_score -= 0.1

        # Multiple questions
        question_count = query.count("?")
        complexity_score += min(0.2, question_count * 0.1)

        return min(1.0, max(0.0, complexity_score))

    def _calculate_adaptive_threshold(self, complexity: float) -> float:
        """Calculate adaptive threshold based on query complexity."""
        # Simple queries: higher threshold (0.75)
        # Complex queries: lower threshold (0.65)
        base_threshold = 0.7
        adjustment = complexity * -0.1  # More complex = lower threshold
        return max(0.6, min(0.8, base_threshold + adjustment))

    # ==================== Feedback Generation ====================

    async def _generate_actionable_feedback(
        self,
        scores: Dict[str, float],
        draft: str,
        query: str,
        fact_check_results: List[Dict[str, Any]],
        consistency_issues: List[str],
        completeness_gaps: List[str],
        citation_quality: Dict[str, Any],
        argument_strength: float,
        domain_specific_scores: Dict[str, float]
    ) -> List[FeedbackItem]:
        """Generate actionable feedback with priority ranking."""
        feedback_items = []

        # Critical issues (priority 1)
        if scores.get("factual_accuracy", 1.0) < 0.6:
            unsupported = [c for c in fact_check_results if not c.get("supported", True)]
            if unsupported:
                feedback_items.append(FeedbackItem(
                    issue="Unsupported factual claims detected",
                    priority=1,
                    category="factual_accuracy",
                    suggestion=f"Verify {len(unsupported)} claims against sources. Add citations or remove unverified claims.",
                    positive=False
                ))

        if consistency_issues:
            feedback_items.append(FeedbackItem(
                issue=f"{len(consistency_issues)} consistency issue(s) found",
                priority=1,
                category="consistency",
                suggestion="Review and resolve contradictions. Ensure all statements align with each other.",
                location="See consistency_issues in metadata",
                positive=False
            ))

        # High priority issues (priority 2)
        if scores.get("relevance", 1.0) < 0.7:
            feedback_items.append(FeedbackItem(
                issue="Response may not fully address the query",
                priority=2,
                category="relevance",
                suggestion="Expand coverage of key topics from the original query. Ensure all aspects are addressed.",
                positive=False
            ))

        if completeness_gaps:
            feedback_items.append(FeedbackItem(
                issue=f"{len(completeness_gaps)} aspect(s) missing",
                priority=2,
                category="completeness",
                suggestion=f"Address missing aspects: {', '.join(completeness_gaps[:3])}",
                positive=False
            ))

        if citation_quality.get("density_score", 1.0) < 0.5:
            feedback_items.append(FeedbackItem(
                issue="Insufficient citations",
                priority=2,
                category="citation_quality",
                suggestion="Add more citations to support claims. Aim for 2-3 citations per 100 words.",
                positive=False
            ))

        # Medium priority issues (priority 3)
        if scores.get("clarity", 1.0) < 0.7:
            feedback_items.append(FeedbackItem(
                issue="Organization and clarity could be improved",
                priority=3,
                category="clarity",
                suggestion="Use clear headings, organize into logical sections, and improve paragraph structure.",
                positive=False
            ))

        if argument_strength < 0.6:
            feedback_items.append(FeedbackItem(
                issue="Argument strength could be stronger",
                priority=3,
                category="argument_strength",
                suggestion="Strengthen logical reasoning. Use evidence to support each claim. Add clear cause-effect relationships.",
                positive=False
            ))

        # Positive reinforcement (priority 4, positive=True)
        strong_scores = [k for k, v in scores.items() if v >= 0.8]
        if strong_scores:
            feedback_items.append(FeedbackItem(
                issue=f"Strong performance in: {', '.join(strong_scores)}",
                priority=4,
                category="strengths",
                suggestion="Maintain these strengths in future revisions.",
                positive=True
            ))

        if domain_specific_scores:
            strong_domain = [k for k, v in domain_specific_scores.items() if v >= 0.75]
            if strong_domain:
                feedback_items.append(FeedbackItem(
                    issue=f"Strong HCI criteria: {', '.join(strong_domain)}",
                    priority=4,
                    category="domain_specific",
                    suggestion="Excellent domain-specific coverage.",
                    positive=True
                ))

        # Sort by priority
        feedback_items.sort(key=lambda x: x.priority)

        return feedback_items

    def _generate_feedback_enhanced(
        self,
        scores: Dict[str, float],
        draft: str,
        query: str,
        fact_check_results: List[Dict[str, Any]],
        consistency_issues: List[str],
        completeness_gaps: List[str],
        citation_quality: Dict[str, Any],
        argument_strength: float,
        domain_specific_scores: Dict[str, float]
    ) -> List[FeedbackItem]:
        """Generate enhanced feedback (rule-based)."""
        return asyncio.run(self._generate_actionable_feedback(
            scores, draft, query, fact_check_results, consistency_issues,
            completeness_gaps, citation_quality, argument_strength, domain_specific_scores
        ))

    # ==================== Utility Methods ====================

    def _extract_sources(self, context: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract sources from context."""
        if not context:
            return []

        sources = []

        # Try different context structures
        if "sources" in context:
            sources = context["sources"]
        elif "research_findings" in context:
            sources = context["research_findings"]
        elif "metadata" in context and "sources" in context["metadata"]:
            sources = context["metadata"]["sources"]

        return sources if isinstance(sources, list) else []

    def _calculate_weighted_score(
        self,
        scores: Dict[str, float],
        domain_specific_scores: Dict[str, float]
    ) -> float:
        """Calculate weighted overall score."""
        # Weights for main criteria
        weights = {
            "relevance": 0.20,
            "evidence_quality": 0.15,
            "completeness": 0.15,
            "factual_accuracy": 0.15,
            "consistency": 0.10,
            "clarity": 0.10,
            "citation_quality": 0.10,
            "argument_strength": 0.10
        }

        weighted_sum = sum(scores.get(k, 0.0) * weights.get(k, 0.0) for k in weights.keys())
        weight_sum = sum(weights.values())

        # Add domain-specific scores if available (weighted 20% of total)
        if domain_specific_scores:
            domain_avg = sum(domain_specific_scores.values()) / len(domain_specific_scores) if domain_specific_scores else 0.0
            weighted_sum = weighted_sum * 0.8 + domain_avg * 0.2

        return weighted_sum / weight_sum if weight_sum > 0 else 0.0

    def _identify_strengths_weaknesses(
        self,
        scores: Dict[str, float],
        feedback_items: List[FeedbackItem]
    ) -> Tuple[List[str], List[str]]:
        """Identify strengths and weaknesses."""
        strengths = []
        weaknesses = []

        # Strengths: scores >= 0.8
        for criterion, score in scores.items():
            if score >= 0.8:
                strengths.append(f"{criterion.replace('_', ' ').title()} (score: {score:.2f})")
            elif score < 0.6:
                weaknesses.append(f"{criterion.replace('_', ' ').title()} (score: {score:.2f})")

        # Add positive feedback items to strengths
        for item in feedback_items:
            if item.positive:
                strengths.append(item.issue)

        return strengths, weaknesses

    def _format_evaluation(self, result: EvaluationResult) -> str:
        """Format comprehensive evaluation."""
        evaluation = "## Comprehensive Quality Evaluation\n\n"

        # Scores
        evaluation += "### Multi-Dimensional Scores\n\n"
        for criterion, score in result.scores.items():
            evaluation += f"- **{criterion.replace('_', ' ').title()}**: {score:.2f}/1.00\n"

        if result.domain_specific_scores:
            evaluation += "\n### Domain-Specific Scores (HCI)\n\n"
            for criterion, score in result.domain_specific_scores.items():
                evaluation += f"- **{criterion.replace('_', ' ').title()}**: {score:.2f}/1.00\n"

        evaluation += f"\n**Overall Score**: {result.overall_score:.2f}/1.00\n"
        evaluation += f"**Adaptive Threshold**: {result.adaptive_threshold:.2f}\n\n"

        # Strengths
        if result.strengths:
            evaluation += "### Strengths\n\n"
            for strength in result.strengths:
                evaluation += f"- ✓ {strength}\n"
            evaluation += "\n"

        # Weaknesses
        if result.weaknesses:
            evaluation += "### Areas for Improvement\n\n"
            for weakness in result.weaknesses:
                evaluation += f"- ⚠ {weakness}\n"
            evaluation += "\n"

        # Feedback
        evaluation += "### Actionable Feedback (Priority Ranked)\n\n"
        priority_labels = {1: "🔴 Critical", 2: "🟠 High", 3: "🟡 Medium", 4: "🟢 Low/Positive"}

        for item in result.feedback_items:
            priority_label = priority_labels.get(item.priority, f"Priority {item.priority}")
            icon = "✓" if item.positive else "→"
            evaluation += f"{icon} **[{priority_label}]** {item.issue}\n"
            evaluation += f"   *Suggestion*: {item.suggestion}\n"
            if item.location:
                evaluation += f"   *Location*: {item.location}\n"
            evaluation += "\n"

        # Fact-checking summary
        if result.fact_check_results:
            unsupported = [c for c in result.fact_check_results if not c.get("supported", True)]
            if unsupported:
                evaluation += f"\n### Fact-Checking\n\n"
                evaluation += f"⚠ {len(unsupported)} unsupported claim(s) detected. Review fact_check_results in metadata.\n\n"

        # Consistency issues
        if result.consistency_issues:
            evaluation += f"\n### Consistency Issues\n\n"
            for issue in result.consistency_issues[:5]:
                evaluation += f"- ⚠ {issue}\n"
            evaluation += "\n"

        # Completeness gaps
        if result.completeness_gaps:
            evaluation += f"\n### Completeness Gaps\n\n"
            for gap in result.completeness_gaps[:5]:
                evaluation += f"- ⚠ {gap}\n"
            evaluation += "\n"

        # Citation quality
        if result.citation_quality:
            evaluation += "\n### Citation Quality\n\n"
            evaluation += f"- Formatting: {result.citation_quality.get('formatting_score', 0):.2f}\n"
            evaluation += f"- Relevance: {result.citation_quality.get('relevance_score', 0):.2f}\n"
            evaluation += f"- Credibility: {result.citation_quality.get('credibility_score', 0):.2f}\n"
            evaluation += f"- Density: {result.citation_quality.get('density_score', 0):.2f}\n\n"

        # Status
        evaluation += "\n"
        if result.approved:
            evaluation += "**Status**: ✅ APPROVED - RESEARCH COMPLETE\n"
        else:
            evaluation += "**Status**: ⚠️ NEEDS REVISION\n"
            evaluation += f"\n*Threshold for approval: {result.adaptive_threshold:.2f}*\n"

        return evaluation

    def _format_feedback_items(self, feedback_items: List[FeedbackItem]) -> List[str]:
        """Format feedback items as simple list."""
        return [f"[Priority {item.priority}] {item.issue}: {item.suggestion}" for item in feedback_items]

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM API."""
        if not self.llm_client:
            raise ValueError("LLM client not initialized")

        model_config = self.config.get("models", {}).get("judge", {})
        model_name = model_config.get("name", "gpt-4o-mini" if self.client_type == "openai" else "llama-3.1-8b-instant")
        temperature = model_config.get("temperature", 0.3)
        max_tokens = model_config.get("max_tokens", 1024)

        try:
            if self.client_type == "openai":
                response = self.llm_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator. Provide responses in valid JSON format only, no markdown."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            else:  # groq
                response = self.llm_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator. Provide responses in valid JSON format only, no markdown."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}")
            raise

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text (handles markdown code blocks)."""
        text = text.strip()

        # Remove markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                text = text[start:end].strip()

        # Find JSON object
        json_start = text.find("{")
        json_end = text.rfind("}")

        if json_start != -1 and json_end != -1 and json_end > json_start:
            return text[json_start:json_end + 1]

        return text

    # Legacy methods for backward compatibility
    def _evaluate_criteria(
        self,
        draft: str,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Legacy evaluation method (backward compatibility)."""
        return self._evaluate_criteria_enhanced(draft, query, context)

    def _generate_feedback(
        self,
        scores: Dict[str, float],
        draft: str,
        query: str
    ) -> List[str]:
        """Legacy feedback generation (backward compatibility)."""
        feedback_items = self._generate_feedback_enhanced(
            scores, draft, query, [], [], [], {}, 0.5, {}
        )
        return self._format_feedback_items(feedback_items)


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
