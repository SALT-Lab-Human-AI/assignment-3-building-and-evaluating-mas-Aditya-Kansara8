"""
Planner Agent

This agent breaks down research queries into actionable research steps.
It uses LLM to analyze queries and create structured research plans with advanced
query analysis, plan quality assessment, and domain-specific planning.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from src.agents.base_agent import BaseAgent


class QueryType(Enum):
    """Types of research queries."""
    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    EXPLORATORY = "exploratory"
    CAUSAL = "causal"
    DESCRIPTIVE = "descriptive"
    EVALUATIVE = "evaluative"
    METHODOLOGICAL = "methodological"


class Scope(Enum):
    """Query scope classification."""
    BROAD = "broad"
    NARROW = "narrow"
    MODERATE = "moderate"


class Methodology(Enum):
    """Research methodology types."""
    EMPIRICAL = "empirical"
    THEORETICAL = "theoretical"
    MIXED = "mixed"
    META_ANALYSIS = "meta_analysis"
    CASE_STUDY = "case_study"


@dataclass
class TemporalConstraint:
    """Represents temporal constraints in a query."""
    type: str  # "recent", "specific_year", "range", "decade", etc.
    value: Optional[str] = None  # e.g., "2020", "last 5 years"
    start_year: Optional[int] = None
    end_year: Optional[int] = None


@dataclass
class Entity:
    """Represents an entity extracted from the query."""
    text: str
    type: str  # "concept", "method", "technology", "domain", etc.
    importance: float = 1.0  # 0.0 to 1.0


@dataclass
class PlanStep:
    """Represents a single step in a research plan."""
    id: str
    description: str
    priority: int  # 1-5, where 1 is highest priority
    confidence: float  # 0.0 to 1.0
    dependencies: List[str] = field(default_factory=list)  # IDs of steps this depends on
    estimated_time: Optional[str] = None
    source_types: List[str] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)
    domain_topics: List[str] = field(default_factory=list)
    methodology: Optional[Methodology] = None


@dataclass
class QueryAnalysis:
    """Comprehensive analysis of a research query."""
    query_type: QueryType
    scope: Scope
    entities: List[Entity]
    relationships: List[Tuple[str, str, str]]  # (entity1, relation, entity2)
    temporal_constraints: List[TemporalConstraint]
    domain: str  # e.g., "HCI", "general"
    methodology_preference: Optional[Methodology] = None
    complexity_score: float = 0.5  # 0.0 to 1.0


@dataclass
class ResearchPlan:
    """A complete research plan with quality metrics."""
    steps: List[PlanStep]
    total_confidence: float  # Average confidence across all steps
    estimated_duration: Optional[str] = None
    recommended_databases: List[str] = field(default_factory=list)
    domain_specific_notes: List[str] = field(default_factory=list)
    methodology_notes: List[str] = field(default_factory=list)


class PlannerAgent(BaseAgent):
    """
    Enhanced Agent responsible for planning research tasks.

    The planner analyzes research queries and breaks them down into:
    - Key concepts and topics to investigate
    - Types of sources needed (academic papers, web articles, etc.)
    - Specific search queries for the researcher
    - Outline for synthesizing findings

    New capabilities:
    - Query type detection and analysis
    - Entity and relationship extraction
    - Temporal constraint identification
    - Scope detection
    - Multiple plan variant generation
    - Confidence scoring and dependency mapping
    - Domain-specific planning (HCI focus)
    - Methodological considerations
    """

    # HCI domain knowledge
    HCI_SUBTOPICS = [
        "user experience (UX)", "user interface (UI)", "usability", "accessibility",
        "interaction design", "information architecture", "human-computer interaction",
        "user-centered design", "participatory design", "cognitive load", "affordances",
        "gestalt principles", "Fitts' law", "Hick's law", "visual design", "prototyping",
        "wireframing", "user testing", "A/B testing", "heuristic evaluation",
        "task analysis", "personas", "scenarios", "journey mapping", "design thinking"
    ]

    HCI_DATABASES = {
        QueryType.FACTUAL: ["ACM Digital Library", "IEEE Xplore", "Google Scholar"],
        QueryType.COMPARATIVE: ["ACM Digital Library", "Semantic Scholar", "Web of Science"],
        QueryType.EXPLORATORY: ["ACM Digital Library", "arXiv", "ResearchGate", "Google Scholar"],
        QueryType.CAUSAL: ["ACM Digital Library", "IEEE Xplore", "PubMed"],
        QueryType.DESCRIPTIVE: ["ACM Digital Library", "Google Scholar", "Semantic Scholar"],
        QueryType.EVALUATIVE: ["ACM Digital Library", "IEEE Xplore", "Semantic Scholar"],
        QueryType.METHODOLOGICAL: ["ACM Digital Library", "IEEE Xplore", "Google Scholar"]
    }

    # Temporal patterns
    TEMPORAL_PATTERNS = [
        (r'recent(ly)?', 'recent', None),
        (r'last\s+(\d+)\s+years?', 'range', None),
        (r'past\s+(\d+)\s+years?', 'range', None),
        (r'since\s+(\d{4})', 'since', None),
        (r'(\d{4})\s*[-–]\s*(\d{4})', 'range', None),
        (r'in\s+(\d{4})', 'specific_year', None),
        (r'(\d{4})s', 'decade', None),
        (r'contemporary', 'recent', None),
        (r'current', 'recent', None),
        (r'modern', 'recent', None),
    ]

    # Query type indicators
    QUERY_TYPE_INDICATORS = {
        QueryType.COMPARATIVE: ['compare', 'comparison', 'versus', 'vs', 'difference', 'similarities', 'contrast'],
        QueryType.EXPLORATORY: ['explore', 'investigate', 'examine', 'analyze', 'study', 'research'],
        QueryType.CAUSAL: ['cause', 'effect', 'impact', 'influence', 'affect', 'why', 'how does'],
        QueryType.DESCRIPTIVE: ['what is', 'describe', 'define', 'explain', 'overview'],
        QueryType.EVALUATIVE: ['evaluate', 'assess', 'critique', 'review', 'judge', 'quality'],
        QueryType.METHODOLOGICAL: ['method', 'approach', 'technique', 'framework', 'model', 'how to'],
        QueryType.FACTUAL: ['fact', 'statistics', 'data', 'numbers', 'how many', 'how much']
    }

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
        self.domain = config.get("system", {}).get("topic", "HCI Research") if config else "HCI Research"

    def _get_default_prompt(self) -> str:
        """Get default system prompt for the planner."""
        return """You are an expert Research Planner specializing in HCI (Human-Computer Interaction) research.

Your enhanced capabilities include:

1. QUERY ANALYSIS:
   - Detect query type (factual, comparative, exploratory, causal, descriptive, evaluative, methodological)
   - Extract entities (concepts, methods, technologies) and their relationships
   - Identify temporal constraints (e.g., "recent", "last 5 years", "2020-2024")
   - Determine scope (broad vs. narrow)

2. PLAN QUALITY:
   - Generate multiple plan variants and select the best one
   - Assign confidence scores (0.0-1.0) to each step
   - Map dependencies between research steps
   - Prioritize steps by importance (1-5, where 1 is highest)

3. DOMAIN-SPECIFIC PLANNING (HCI):
   - Suggest relevant HCI sub-topics (UX, UI, usability, accessibility, interaction design, etc.)
   - Recommend specific databases per query type:
     * Factual: ACM Digital Library, IEEE Xplore, Google Scholar
     * Comparative: ACM Digital Library, Semantic Scholar, Web of Science
     * Exploratory: ACM Digital Library, arXiv, ResearchGate, Google Scholar
   - Include methodological considerations (empirical vs. theoretical, case studies, meta-analyses)

When creating a plan, provide:
- Structured numbered steps with clear descriptions
- Priority levels (1-5) for each step
- Confidence scores for each step
- Dependencies between steps (e.g., "Step 3 depends on Step 1")
- Recommended databases and source types
- Domain-specific sub-topics to explore
- Methodological approach recommendations
- Temporal constraints to apply in searches

Format your response as a detailed plan with all these elements.

After creating the plan, say "PLAN COMPLETE"."""

    def process(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a research query and create an enhanced plan.

        Args:
            query: Research query to plan for
            context: Optional context (e.g., topic domain, constraints)

        Returns:
            Dictionary with:
            - plan: Structured research plan
            - search_queries: List of suggested search queries
            - key_concepts: List of key concepts to investigate
            - query_analysis: Detailed query analysis
            - research_plan: Structured plan object
            - metadata: Additional planning information
        """
        self.logger.info(f"Planning for query: {query[:100]}...")

        # Perform comprehensive query analysis
        query_analysis = self._analyze_query(query, context)
        self.logger.info(f"Query type: {query_analysis.query_type.value}, Scope: {query_analysis.scope.value}")

        # Generate multiple plan variants and select the best
        plan_variants = self._generate_plan_variants(query, query_analysis, context)
        best_plan = self._select_best_plan(plan_variants)

        self.logger.info(f"Selected plan with {len(best_plan.steps)} steps, "
                        f"avg confidence: {best_plan.total_confidence:.2f}")

        # If we have a model client (AutoGen integration), enhance with LLM
        if self.model_client:
            # In AutoGen, the agent handles LLM calls internally
            # Return structured data that can be used by AutoGen
            return {
                "plan": self._format_plan_for_autogen(best_plan, query_analysis),
                "search_queries": self._extract_all_search_queries(best_plan),
                "key_concepts": [e.text for e in query_analysis.entities],
                "query_analysis": self._serialize_query_analysis(query_analysis),
                "research_plan": self._serialize_research_plan(best_plan),
                "metadata": {
                    "method": "autogen",
                    "query_type": query_analysis.query_type.value,
                    "scope": query_analysis.scope.value,
                    "complexity": query_analysis.complexity_score
                }
            }

        # Standalone planning with enhanced capabilities
        formatted_plan = self._format_plan(best_plan, query_analysis)

        return {
            "plan": formatted_plan,
            "search_queries": self._extract_all_search_queries(best_plan),
            "key_concepts": [e.text for e in query_analysis.entities],
            "query_analysis": self._serialize_query_analysis(query_analysis),
            "research_plan": self._serialize_research_plan(best_plan),
            "metadata": {
                "query": query,
                "method": "standalone",
                "query_type": query_analysis.query_type.value,
                "scope": query_analysis.scope.value,
                "complexity": query_analysis.complexity_score
            }
        }

    def _analyze_query(self, query: str, context: Optional[Dict[str, Any]]) -> QueryAnalysis:
        """
        Perform comprehensive query analysis.

        Args:
            query: Research query
            context: Optional context

        Returns:
            QueryAnalysis object with all analysis results
        """
        query_lower = query.lower()

        # Detect query type
        query_type = self._detect_query_type(query_lower)

        # Detect scope
        scope = self._detect_scope(query, query_type)

        # Extract entities
        entities = self._extract_entities(query)

        # Extract relationships
        relationships = self._extract_relationships(query, entities)

        # Detect temporal constraints
        temporal_constraints = self._detect_temporal_constraints(query)

        # Detect domain
        domain = self._detect_domain(query, context)

        # Detect methodology preference
        methodology_preference = self._detect_methodology_preference(query_lower)

        # Calculate complexity score
        complexity_score = self._calculate_complexity(
            query, query_type, scope, len(entities), len(relationships)
        )

        return QueryAnalysis(
            query_type=query_type,
            scope=scope,
            entities=entities,
            relationships=relationships,
            temporal_constraints=temporal_constraints,
            domain=domain,
            methodology_preference=methodology_preference,
            complexity_score=complexity_score
        )

    def _detect_query_type(self, query_lower: str) -> QueryType:
        """Detect the type of research query."""
        # Count matches for each query type
        type_scores = {}
        for qtype, indicators in self.QUERY_TYPE_INDICATORS.items():
            score = sum(1 for indicator in indicators if indicator in query_lower)
            if score > 0:
                type_scores[qtype] = score

        if type_scores:
            # Return the type with highest score
            return max(type_scores.items(), key=lambda x: x[1])[0]

        # Default to exploratory if no clear match
        return QueryType.EXPLORATORY

    def _detect_scope(self, query: str, query_type: QueryType) -> Scope:
        """Detect query scope (broad vs. narrow)."""
        query_lower = query.lower()

        # Indicators of broad scope
        broad_indicators = ['overview', 'comprehensive', 'all', 'everything', 'general',
                           'broad', 'wide', 'extensive', 'survey', 'review']

        # Indicators of narrow scope
        narrow_indicators = ['specific', 'particular', 'exact', 'precise', 'focused',
                            'single', 'one', 'specific case', 'particular instance']

        broad_count = sum(1 for indicator in broad_indicators if indicator in query_lower)
        narrow_count = sum(1 for indicator in narrow_indicators if indicator in query_lower)

        # Heuristic: longer queries with multiple concepts tend to be broader
        word_count = len(query.split())
        concept_count = len([w for w in query.split() if len(w) > 5])

        if narrow_count > broad_count:
            return Scope.NARROW
        elif broad_count > narrow_count or (word_count > 15 and concept_count > 3):
            return Scope.BROAD
        else:
            return Scope.MODERATE

    def _extract_entities(self, query: str) -> List[Entity]:
        """Extract entities (concepts, methods, technologies) from query."""
        entities = []
        query_lower = query.lower()

        # Extract capitalized phrases (likely proper nouns/concepts)
        capitalized_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        capitalized_matches = re.findall(capitalized_pattern, query)
        for match in capitalized_matches:
            if match.lower() not in ['The', 'A', 'An', 'And', 'Or', 'But']:
                entities.append(Entity(
                    text=match,
                    type="concept",
                    importance=0.8
                ))

        # Extract technical terms (words with specific patterns)
        technical_patterns = [
            r'\b\w+[-_]?\w*[-_]?design\b',
            r'\b\w+[-_]?\w*[-_]?method\b',
            r'\b\w+[-_]?\w*[-_]?framework\b',
            r'\b\w+[-_]?\w*[-_]?system\b',
            r'\b\w+[-_]?\w*[-_]?interface\b',
            r'\b\w+[-_]?\w*[-_]?technology\b',
        ]

        for pattern in technical_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                if match not in [e.text.lower() for e in entities]:
                    entities.append(Entity(
                        text=match.title(),
                        type="technology" if "technology" in match else "method",
                        importance=0.7
                    ))

        # Extract important keywords (non-stop words, length > 4)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
                     'for', 'of', 'with', 'by', 'what', 'how', 'why', 'when', 'where',
                     'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were',
                     'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'}

        words = query.split()
        for word in words:
            word_clean = re.sub(r'[^\w]', '', word.lower())
            if len(word_clean) > 4 and word_clean not in stop_words:
                if word_clean not in [e.text.lower() for e in entities]:
                    entities.append(Entity(
                        text=word.title(),
                        type="concept",
                        importance=0.6
                    ))

        # Check for HCI-specific terms and boost their importance
        for entity in entities:
            if any(topic in entity.text.lower() for topic in self.HCI_SUBTOPICS):
                entity.importance = min(1.0, entity.importance + 0.2)
                entity.type = "domain"

        # Sort by importance and return top entities
        entities.sort(key=lambda x: x.importance, reverse=True)
        return entities[:10]  # Return top 10 entities

    def _extract_relationships(self, query: str, entities: List[Entity]) -> List[Tuple[str, str, str]]:
        """Extract relationships between entities."""
        relationships = []
        query_lower = query.lower()

        # Common relationship patterns
        relation_patterns = [
            (r'(\w+)\s+(affects?|influences?|impacts?)\s+(\w+)', 'affects'),
            (r'(\w+)\s+(versus|vs|compared to|compared with)\s+(\w+)', 'compared_to'),
            (r'(\w+)\s+(and|with|combined with)\s+(\w+)', 'related_to'),
            (r'(\w+)\s+(causes?|leads? to|results? in)\s+(\w+)', 'causes'),
            (r'(\w+)\s+(uses?|employs?|utilizes?)\s+(\w+)', 'uses'),
            (r'(\w+)\s+(improves?|enhances?|increases?)\s+(\w+)', 'improves'),
        ]

        entity_texts = [e.text.lower() for e in entities]

        for pattern, relation_type in relation_patterns:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                entity1 = match.group(1)
                entity2 = match.group(3)
                if entity1 in entity_texts or entity2 in entity_texts:
                    relationships.append((entity1, relation_type, entity2))

        return relationships[:5]  # Return top 5 relationships

    def _detect_temporal_constraints(self, query: str) -> List[TemporalConstraint]:
        """Detect temporal constraints in the query."""
        constraints = []
        query_lower = query.lower()

        for pattern, constraint_type, _ in self.TEMPORAL_PATTERNS:
            matches = re.finditer(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                if constraint_type == 'range':
                    # Extract year range
                    if len(match.groups()) >= 2:
                        try:
                            start = int(match.group(1))
                            end = int(match.group(2)) if match.group(2) else None
                            constraints.append(TemporalConstraint(
                                type='range',
                                start_year=start,
                                end_year=end,
                                value=f"{start}-{end}" if end else f"since {start}"
                            ))
                        except (ValueError, IndexError):
                            # Try "last N years" pattern
                            try:
                                years = int(match.group(1))
                                constraints.append(TemporalConstraint(
                                    type='range',
                                    value=f"last {years} years"
                                ))
                            except (ValueError, IndexError):
                                pass
                    else:
                        # "last N years" pattern
                        try:
                            years = int(match.group(1))
                            constraints.append(TemporalConstraint(
                                type='range',
                                value=f"last {years} years"
                            ))
                        except (ValueError, IndexError):
                            pass
                elif constraint_type == 'specific_year':
                    try:
                        year = int(match.group(1))
                        constraints.append(TemporalConstraint(
                            type='specific_year',
                            value=str(year)
                        ))
                    except (ValueError, IndexError):
                        pass
                elif constraint_type == 'since':
                    try:
                        year = int(match.group(1))
                        constraints.append(TemporalConstraint(
                            type='since',
                            value=str(year),
                            start_year=year
                        ))
                    except (ValueError, IndexError):
                        pass
                elif constraint_type == 'decade':
                    try:
                        decade = match.group(1)
                        constraints.append(TemporalConstraint(
                            type='decade',
                            value=f"{decade}s"
                        ))
                    except (ValueError, IndexError):
                        pass
                else:
                    # Recent, contemporary, current, modern
                    constraints.append(TemporalConstraint(
                        type='recent',
                        value='recent'
                    ))

        # Default to recent if no constraints found (common in research)
        if not constraints:
            constraints.append(TemporalConstraint(
                type='recent',
                value='recent (last 5 years)'
            ))

        return constraints

    def _detect_domain(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Detect the research domain."""
        query_lower = query.lower()

        # Check context first
        if context and 'domain' in context:
            return context['domain']

        # Check config
        if self.config and 'system' in self.config:
            topic = self.config['system'].get('topic', '')
            if 'HCI' in topic or 'Human-Computer' in topic:
                return "HCI"

        # Detect from query
        hci_indicators = ['hci', 'human-computer', 'user interface', 'user experience',
                         'ux', 'ui', 'usability', 'interaction design', 'ux design',
                         'user-centered', 'interface design', 'interaction', 'affordance']

        if any(indicator in query_lower for indicator in hci_indicators):
            return "HCI"

        return "general"

    def _detect_methodology_preference(self, query_lower: str) -> Optional[Methodology]:
        """Detect preferred research methodology from query."""
        empirical_indicators = ['empirical', 'experiment', 'study', 'survey', 'evaluation',
                               'testing', 'trial', 'observation', 'data', 'results']
        theoretical_indicators = ['theory', 'theoretical', 'framework', 'model', 'concept',
                                 'principle', 'design pattern', 'paradigm']
        meta_indicators = ['meta-analysis', 'meta analysis', 'systematic review', 'review']
        case_study_indicators = ['case study', 'case', 'example', 'instance']

        if any(indicator in query_lower for indicator in meta_indicators):
            return Methodology.META_ANALYSIS
        elif any(indicator in query_lower for indicator in case_study_indicators):
            return Methodology.CASE_STUDY
        elif any(indicator in query_lower for indicator in empirical_indicators):
            if any(indicator in query_lower for indicator in theoretical_indicators):
                return Methodology.MIXED
            return Methodology.EMPIRICAL
        elif any(indicator in query_lower for indicator in theoretical_indicators):
            return Methodology.THEORETICAL

        return None

    def _calculate_complexity(self, query: str, query_type: QueryType, scope: Scope,
                             entity_count: int, relationship_count: int) -> float:
        """Calculate query complexity score (0.0 to 1.0)."""
        complexity = 0.0

        # Query type complexity
        type_complexity = {
            QueryType.FACTUAL: 0.2,
            QueryType.DESCRIPTIVE: 0.3,
            QueryType.EXPLORATORY: 0.5,
            QueryType.COMPARATIVE: 0.6,
            QueryType.EVALUATIVE: 0.7,
            QueryType.CAUSAL: 0.8,
            QueryType.METHODOLOGICAL: 0.7
        }
        complexity += type_complexity.get(query_type, 0.5) * 0.3

        # Scope complexity
        scope_complexity = {
            Scope.NARROW: 0.3,
            Scope.MODERATE: 0.5,
            Scope.BROAD: 0.8
        }
        complexity += scope_complexity.get(scope, 0.5) * 0.2

        # Entity count complexity (normalized)
        entity_complexity = min(1.0, entity_count / 5.0)
        complexity += entity_complexity * 0.3

        # Relationship complexity
        relationship_complexity = min(1.0, relationship_count / 3.0)
        complexity += relationship_complexity * 0.2

        return min(1.0, complexity)

    def _generate_plan_variants(self, query: str, analysis: QueryAnalysis,
                               context: Optional[Dict[str, Any]]) -> List[ResearchPlan]:
        """Generate multiple plan variants and return them."""
        variants = []

        # Variant 1: Comprehensive approach (more steps, thorough)
        variant1 = self._create_comprehensive_plan(query, analysis, context)
        variants.append(variant1)

        # Variant 2: Focused approach (fewer steps, high priority)
        variant2 = self._create_focused_plan(query, analysis, context)
        variants.append(variant2)

        # Variant 3: Domain-optimized approach (HCI-specific)
        if analysis.domain == "HCI":
            variant3 = self._create_domain_optimized_plan(query, analysis, context)
            variants.append(variant3)

        return variants

    def _create_comprehensive_plan(self, query: str, analysis: QueryAnalysis,
                                  context: Optional[Dict[str, Any]]) -> ResearchPlan:
        """Create a comprehensive research plan with many steps."""
        steps = []
        step_id = 1

        # Step 1: Background research
        steps.append(PlanStep(
            id=f"step_{step_id}",
            description=f"Conduct background research on {', '.join([e.text for e in analysis.entities[:3]])}",
            priority=1,
            confidence=0.9,
            dependencies=[],
            source_types=["academic papers", "survey articles"],
            search_queries=[f"{e.text} {analysis.domain}" for e in analysis.entities[:2]],
            domain_topics=analysis.domain.split() if analysis.domain != "general" else [],
            methodology=Methodology.THEORETICAL
        ))
        step_id += 1

        # Step 2: Domain-specific exploration
        if analysis.domain == "HCI":
            hci_topics = [topic for topic in self.HCI_SUBTOPICS
                         if any(e.text.lower() in topic for e in analysis.entities[:3])]
            if not hci_topics:
                hci_topics = self.HCI_SUBTOPICS[:3]

            steps.append(PlanStep(
                id=f"step_{step_id}",
                description=f"Explore HCI sub-topics: {', '.join(hci_topics[:3])}",
                priority=2,
                confidence=0.85,
                dependencies=[f"step_{step_id-1}"],
                source_types=["academic papers"],
                search_queries=[f"{topic} {query}" for topic in hci_topics[:2]],
                domain_topics=hci_topics[:3],
                methodology=Methodology.EMPIRICAL
            ))
            step_id += 1

        # Step 3: Temporal-constrained search
        if analysis.temporal_constraints:
            temp_constraint = analysis.temporal_constraints[0]
            steps.append(PlanStep(
                id=f"step_{step_id}",
                description=f"Search for {temp_constraint.value} publications",
                priority=2,
                confidence=0.8,
                dependencies=[],
                source_types=["academic papers"],
                search_queries=[f"{query} {temp_constraint.value}"],
                domain_topics=[],
                methodology=None
            ))
            step_id += 1

        # Step 4: Comparative analysis (if applicable)
        if analysis.query_type == QueryType.COMPARATIVE and analysis.relationships:
            steps.append(PlanStep(
                id=f"step_{step_id}",
                description=f"Compare {analysis.relationships[0][0]} and {analysis.relationships[0][2]}",
                priority=1,
                confidence=0.85,
                dependencies=[f"step_{step_id-2}"],
                source_types=["academic papers", "comparative studies"],
                search_queries=[f"comparison {analysis.relationships[0][0]} {analysis.relationships[0][2]}"],
                domain_topics=[],
                methodology=Methodology.EMPIRICAL
            ))
            step_id += 1

        # Step 5: Methodology-specific search
        if analysis.methodology_preference:
            steps.append(PlanStep(
                id=f"step_{step_id}",
                description=f"Find {analysis.methodology_preference.value} studies",
                priority=3,
                confidence=0.75,
                dependencies=[f"step_{step_id-2}"],
                source_types=["academic papers"],
                search_queries=[f"{query} {analysis.methodology_preference.value}"],
                domain_topics=[],
                methodology=analysis.methodology_preference
            ))
            step_id += 1

        # Step 6: Synthesis preparation
        steps.append(PlanStep(
            id=f"step_{step_id}",
            description="Organize findings and prepare synthesis outline",
            priority=1,
            confidence=0.9,
            dependencies=[f"step_{step_id-1}"],
            source_types=[],
            search_queries=[],
            domain_topics=[],
            methodology=None
        ))

        # Calculate average confidence
        avg_confidence = sum(step.confidence for step in steps) / len(steps) if steps else 0.0

        # Get recommended databases
        recommended_databases = self.HCI_DATABASES.get(
            analysis.query_type,
            ["ACM Digital Library", "Google Scholar", "Semantic Scholar"]
        )

        return ResearchPlan(
            steps=steps,
            total_confidence=avg_confidence,
            recommended_databases=recommended_databases,
            domain_specific_notes=self._generate_domain_notes(analysis),
            methodology_notes=self._generate_methodology_notes(analysis)
        )

    def _create_focused_plan(self, query: str, analysis: QueryAnalysis,
                            context: Optional[Dict[str, Any]]) -> ResearchPlan:
        """Create a focused research plan with fewer, high-priority steps."""
        steps = []
        step_id = 1

        # Step 1: Core concept research
        top_entities = [e.text for e in sorted(analysis.entities, key=lambda x: x.importance, reverse=True)[:2]]
        steps.append(PlanStep(
            id=f"step_{step_id}",
            description=f"Research core concepts: {', '.join(top_entities)}",
            priority=1,
            confidence=0.95,
            dependencies=[],
            source_types=["academic papers"],
            search_queries=[f"{entity} {analysis.domain}" for entity in top_entities],
            domain_topics=top_entities,
            methodology=analysis.methodology_preference or Methodology.EMPIRICAL
        ))
        step_id += 1

        # Step 2: Direct answer search
        steps.append(PlanStep(
            id=f"step_{step_id}",
            description=f"Search for direct answers to: {query[:100]}",
            priority=1,
            confidence=0.9,
            dependencies=[f"step_{step_id-1}"],
            source_types=["academic papers", "web articles"],
            search_queries=[query],
            domain_topics=[],
            methodology=None
        ))
        step_id += 1

        # Step 3: Synthesis
        steps.append(PlanStep(
            id=f"step_{step_id}",
            description="Synthesize findings into coherent response",
            priority=1,
            confidence=0.95,
            dependencies=[f"step_{step_id-1}"],
            source_types=[],
            search_queries=[],
            domain_topics=[],
            methodology=None
        ))

        avg_confidence = sum(step.confidence for step in steps) / len(steps) if steps else 0.0
        recommended_databases = self.HCI_DATABASES.get(
            analysis.query_type,
            ["ACM Digital Library", "Google Scholar"]
        )

        return ResearchPlan(
            steps=steps,
            total_confidence=avg_confidence,
            recommended_databases=recommended_databases,
            domain_specific_notes=self._generate_domain_notes(analysis),
            methodology_notes=self._generate_methodology_notes(analysis)
        )

    def _create_domain_optimized_plan(self, query: str, analysis: QueryAnalysis,
                                     context: Optional[Dict[str, Any]]) -> ResearchPlan:
        """Create an HCI domain-optimized research plan."""
        steps = []
        step_id = 1

        # Step 1: HCI foundational concepts
        relevant_hci_topics = [topic for topic in self.HCI_SUBTOPICS
                              if any(e.text.lower() in topic or topic in e.text.lower()
                                     for e in analysis.entities)]
        if not relevant_hci_topics:
            relevant_hci_topics = ["user experience", "usability", "interaction design"]

        steps.append(PlanStep(
            id=f"step_{step_id}",
            description=f"Research HCI foundations: {', '.join(relevant_hci_topics[:3])}",
            priority=1,
            confidence=0.9,
            dependencies=[],
            source_types=["academic papers"],
            search_queries=[f"{topic} HCI" for topic in relevant_hci_topics[:2]],
            domain_topics=relevant_hci_topics[:3],
            methodology=Methodology.THEORETICAL
        ))
        step_id += 1

        # Step 2: Recent HCI research
        steps.append(PlanStep(
            id=f"step_{step_id}",
            description="Find recent HCI research (last 5 years)",
            priority=2,
            confidence=0.85,
            dependencies=[f"step_{step_id-1}"],
            source_types=["academic papers"],
            search_queries=[f"{query} HCI recent"],
            domain_topics=["recent research"],
            methodology=Methodology.EMPIRICAL
        ))
        step_id += 1

        # Step 3: ACM Digital Library specific search
        steps.append(PlanStep(
            id=f"step_{step_id}",
            description="Search ACM Digital Library for authoritative sources",
            priority=2,
            confidence=0.9,
            dependencies=[f"step_{step_id-1}"],
            source_types=["academic papers"],
            search_queries=[f"{query} site:dl.acm.org"],
            domain_topics=[],
            methodology=None
        ))
        step_id += 1

        # Step 4: Synthesis with HCI perspective
        steps.append(PlanStep(
            id=f"step_{step_id}",
            description="Synthesize findings with HCI design principles",
            priority=1,
            confidence=0.9,
            dependencies=[f"step_{step_id-1}"],
            source_types=[],
            search_queries=[],
            domain_topics=["HCI principles"],
            methodology=None
        ))

        avg_confidence = sum(step.confidence for step in steps) / len(steps) if steps else 0.0

        return ResearchPlan(
            steps=steps,
            total_confidence=avg_confidence,
            recommended_databases=["ACM Digital Library", "IEEE Xplore", "Google Scholar"],
            domain_specific_notes=[
                "Focus on CHI, UIST, and other top HCI venues",
                "Consider usability heuristics and design principles",
                "Include both theoretical frameworks and empirical studies"
            ],
            methodology_notes=self._generate_methodology_notes(analysis)
        )

    def _select_best_plan(self, variants: List[ResearchPlan]) -> ResearchPlan:
        """Select the best plan from multiple variants."""
        if not variants:
            # Return a default plan
            return ResearchPlan(steps=[], total_confidence=0.0)

        # Score each variant based on:
        # - Average confidence (40%)
        # - Number of steps (30% - prefer moderate number)
        # - Coverage (30% - based on step diversity)

        scored_variants = []
        for variant in variants:
            confidence_score = variant.total_confidence * 0.4

            # Step count score (prefer 3-6 steps)
            step_count = len(variant.steps)
            if 3 <= step_count <= 6:
                step_score = 1.0
            elif step_count < 3:
                step_score = step_count / 3.0
            else:
                step_score = max(0.5, 1.0 - (step_count - 6) * 0.1)
            step_score *= 0.3

            # Coverage score (diversity of source types and methodologies)
            source_types = set()
            methodologies = set()
            for step in variant.steps:
                source_types.update(step.source_types)
                if step.methodology:
                    methodologies.add(step.methodology)
            coverage_score = (len(source_types) + len(methodologies)) / 5.0  # Normalize
            coverage_score = min(1.0, coverage_score) * 0.3

            total_score = confidence_score + step_score + coverage_score
            scored_variants.append((total_score, variant))

        # Return the variant with highest score
        scored_variants.sort(key=lambda x: x[0], reverse=True)
        return scored_variants[0][1]

    def _generate_domain_notes(self, analysis: QueryAnalysis) -> List[str]:
        """Generate domain-specific planning notes."""
        notes = []

        if analysis.domain == "HCI":
            notes.append("Prioritize ACM Digital Library and IEEE Xplore for HCI research")
            notes.append("Consider CHI, UIST, DIS, and other top HCI conferences")
            notes.append("Include both theoretical frameworks and empirical usability studies")

            # Add notes based on query type
            if analysis.query_type == QueryType.COMPARATIVE:
                notes.append("Look for comparative studies and A/B testing results")
            elif analysis.query_type == QueryType.EVALUATIVE:
                notes.append("Focus on evaluation methods: heuristic evaluation, user testing, surveys")
            elif analysis.query_type == QueryType.METHODOLOGICAL:
                notes.append("Include design methodologies: user-centered design, participatory design")
        else:
            notes.append("Use general academic databases: Google Scholar, Semantic Scholar")

        return notes

    def _generate_methodology_notes(self, analysis: QueryAnalysis) -> List[str]:
        """Generate methodology-specific planning notes."""
        notes = []

        if analysis.methodology_preference:
            if analysis.methodology_preference == Methodology.EMPIRICAL:
                notes.append("Prioritize empirical studies: experiments, surveys, user studies")
                notes.append("Look for quantitative data and statistical analysis")
            elif analysis.methodology_preference == Methodology.THEORETICAL:
                notes.append("Focus on theoretical frameworks and conceptual models")
                notes.append("Include foundational papers and design principles")
            elif analysis.methodology_preference == Methodology.META_ANALYSIS:
                notes.append("Search for systematic reviews and meta-analyses")
                notes.append("Prioritize comprehensive survey papers")
            elif analysis.methodology_preference == Methodology.CASE_STUDY:
                notes.append("Look for case studies and real-world implementations")
                notes.append("Include industry reports and practical examples")

        # Add temporal constraint notes
        if analysis.temporal_constraints:
            for constraint in analysis.temporal_constraints:
                notes.append(f"Apply temporal filter: {constraint.value}")

        return notes

    def _format_plan(self, plan: ResearchPlan, analysis: QueryAnalysis) -> str:
        """Format a research plan as a readable string."""
        output = f"Research Plan\n"
        output += f"{'='*60}\n\n"

        output += f"Query Analysis:\n"
        output += f"  Type: {analysis.query_type.value}\n"
        output += f"  Scope: {analysis.scope.value}\n"
        output += f"  Domain: {analysis.domain}\n"
        output += f"  Complexity: {analysis.complexity_score:.2f}\n"
        if analysis.temporal_constraints:
            output += f"  Temporal Constraints: {', '.join([c.value for c in analysis.temporal_constraints])}\n"
        output += f"\n"

        output += f"Plan Steps (Confidence: {plan.total_confidence:.2f}):\n"
        output += f"{'-'*60}\n"

        for step in plan.steps:
            output += f"\nStep {step.id} (Priority: {step.priority}, Confidence: {step.confidence:.2f}):\n"
            output += f"  {step.description}\n"
            if step.dependencies:
                output += f"  Depends on: {', '.join(step.dependencies)}\n"
            if step.source_types:
                output += f"  Source types: {', '.join(step.source_types)}\n"
            if step.search_queries:
                output += f"  Search queries: {', '.join(step.search_queries)}\n"
            if step.domain_topics:
                output += f"  Domain topics: {', '.join(step.domain_topics)}\n"
            if step.methodology:
                output += f"  Methodology: {step.methodology.value}\n"

        output += f"\n{'='*60}\n"
        output += f"Recommended Databases: {', '.join(plan.recommended_databases)}\n"

        if plan.domain_specific_notes:
            output += f"\nDomain-Specific Notes:\n"
            for note in plan.domain_specific_notes:
                output += f"  - {note}\n"

        if plan.methodology_notes:
            output += f"\nMethodology Notes:\n"
            for note in plan.methodology_notes:
                output += f"  - {note}\n"

        output += f"\nPLAN COMPLETE"
        return output

    def _format_plan_for_autogen(self, plan: ResearchPlan, analysis: QueryAnalysis) -> str:
        """Format plan for AutoGen agent consumption."""
        # Similar to _format_plan but optimized for LLM parsing
        output = f"Research Plan for Query Analysis:\n"
        output += f"- Query Type: {analysis.query_type.value}\n"
        output += f"- Scope: {analysis.scope.value}\n"
        output += f"- Domain: {analysis.domain}\n"
        output += f"- Key Entities: {', '.join([e.text for e in analysis.entities[:5]])}\n"
        if analysis.temporal_constraints:
            output += f"- Temporal Constraints: {', '.join([c.value for c in analysis.temporal_constraints])}\n"
        output += f"\n"

        output += f"Research Steps:\n"
        for i, step in enumerate(plan.steps, 1):
            output += f"\n{i}. {step.description} (Priority: {step.priority}, Confidence: {step.confidence:.2f})\n"
            if step.dependencies:
                output += f"   Depends on steps: {', '.join(step.dependencies)}\n"
            if step.search_queries:
                output += f"   Search queries: {', '.join(step.search_queries)}\n"
            if step.source_types:
                output += f"   Source types: {', '.join(step.source_types)}\n"

        output += f"\nRecommended Databases: {', '.join(plan.recommended_databases)}\n"
        output += f"\nPLAN COMPLETE"
        return output

    def _extract_all_search_queries(self, plan: ResearchPlan) -> List[str]:
        """Extract all search queries from plan steps."""
        queries = []
        for step in plan.steps:
            queries.extend(step.search_queries)
        return queries if queries else ["research query"]

    def _serialize_query_analysis(self, analysis: QueryAnalysis) -> Dict[str, Any]:
        """Serialize QueryAnalysis to dictionary."""
        return {
            "query_type": analysis.query_type.value,
            "scope": analysis.scope.value,
            "entities": [{"text": e.text, "type": e.type, "importance": e.importance} for e in analysis.entities],
            "relationships": [{"entity1": r[0], "relation": r[1], "entity2": r[2]} for r in analysis.relationships],
            "temporal_constraints": [
                {
                    "type": c.type,
                    "value": c.value,
                    "start_year": c.start_year,
                    "end_year": c.end_year
                } for c in analysis.temporal_constraints
            ],
            "domain": analysis.domain,
            "methodology_preference": analysis.methodology_preference.value if analysis.methodology_preference else None,
            "complexity_score": analysis.complexity_score
        }

    def _serialize_research_plan(self, plan: ResearchPlan) -> Dict[str, Any]:
        """Serialize ResearchPlan to dictionary."""
        return {
            "steps": [
                {
                    "id": step.id,
                    "description": step.description,
                    "priority": step.priority,
                    "confidence": step.confidence,
                    "dependencies": step.dependencies,
                    "estimated_time": step.estimated_time,
                    "source_types": step.source_types,
                    "search_queries": step.search_queries,
                    "domain_topics": step.domain_topics,
                    "methodology": step.methodology.value if step.methodology else None
                } for step in plan.steps
            ],
            "total_confidence": plan.total_confidence,
            "estimated_duration": plan.estimated_duration,
            "recommended_databases": plan.recommended_databases,
            "domain_specific_notes": plan.domain_specific_notes,
            "methodology_notes": plan.methodology_notes
        }

    def _create_basic_plan(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Create a basic research plan structure (fallback)."""
        plan = f"Research Plan for: {query}\n\n"
        plan += "1. Analyze the query to identify key research areas\n"
        plan += "2. Determine required source types (academic papers, web articles)\n"
        plan += "3. Formulate specific search queries\n"
        plan += "4. Outline synthesis approach\n"
        plan += "\nPLAN COMPLETE"
        return plan

    def _extract_search_queries(self, plan: str) -> List[str]:
        """Extract suggested search queries from the plan (legacy method)."""
        queries = []
        lines = plan.split('\n')
        for line in lines:
            if 'search' in line.lower() or 'query' in line.lower():
                queries.append(line.strip())
        return queries if queries else ["research query"]

    def _extract_key_concepts(self, query: str) -> List[str]:
        """Extract key concepts from the query (legacy method)."""
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'what', 'how', 'why', 'when', 'where'}
        words = query.lower().split()
        concepts = [w for w in words if w not in stop_words and len(w) > 3]
        return concepts[:5]


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
