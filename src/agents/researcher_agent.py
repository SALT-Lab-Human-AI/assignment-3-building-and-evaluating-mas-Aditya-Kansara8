"""
Researcher Agent

This agent gathers evidence from web and academic sources using search APIs.
It integrates with Tavily, Brave Search, and Semantic Scholar APIs.

Enhanced with:
- Multi-query generation and iterative refinement
- Source diversity balancing
- Recency weighting
- Relevance scoring with embeddings
- Quality metrics (citations, venue, author reputation)
- Deduplication
- Coverage analysis
- Evidence extraction (claims, statistics, quotes)
- Conflict detection
- Source reliability scoring
- Methodology extraction
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import asyncio
from datetime import datetime
from collections import defaultdict
import re
from dataclasses import dataclass, field
from src.agents.base_agent import BaseAgent
from src.tools.web_search import WebSearchTool, web_search
from src.tools.paper_search import PaperSearchTool, paper_search


@dataclass
class Evidence:
    """Structured evidence extracted from sources."""
    claim: str
    source_id: str
    evidence_type: str  # "claim", "statistic", "quote", "methodology"
    value: Optional[str] = None  # For statistics
    context: str = ""
    reliability_score: float = 0.0


@dataclass
class Source:
    """Enhanced source representation with quality metrics."""
    id: str
    type: str  # "web", "paper"
    title: str
    url: str
    content: str
    relevance_score: float = 0.0
    quality_score: float = 0.0
    reliability_score: float = 0.0
    year: Optional[int] = None
    citation_count: int = 0
    venue: str = ""
    authors: List[str] = field(default_factory=list)
    published_date: Optional[str] = None
    evidence: List[Evidence] = field(default_factory=list)
    methodology: Optional[str] = None


class EmbeddingModel:
    """Lightweight embedding model for relevance scoring and deduplication."""

    def __init__(self):
        self.model = None
        self.logger = logging.getLogger("researcher.embeddings")
        self._initialize_model()

    def _initialize_model(self):
        """Initialize embedding model (sentence-transformers or fallback)."""
        try:
            from sentence_transformers import SentenceTransformer
            # Use a lightweight, fast model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("Initialized sentence-transformers embedding model")
        except ImportError:
            self.logger.warning(
                "sentence-transformers not available. "
                "Install with: pip install sentence-transformers. "
                "Falling back to keyword-based similarity."
            )
            self.model = None
        except Exception as e:
            self.logger.warning(f"Could not initialize embedding model: {e}. Using keyword fallback.")
            self.model = None

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts to embeddings."""
        if self.model is None:
            # Fallback: return simple keyword-based vectors
            return self._keyword_vectors(texts)
        try:
            return self.model.encode(texts, convert_to_numpy=True).tolist()
        except Exception as e:
            self.logger.error(f"Embedding encoding error: {e}")
            return self._keyword_vectors(texts)

    def _keyword_vectors(self, texts: List[str]) -> List[List[float]]:
        """Simple keyword-based vectorization fallback."""
        # Extract unique words and create simple TF vectors
        all_words = set()
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.update(words)

        word_list = sorted(list(all_words))
        vectors = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            vector = [words.count(word) for word in word_list]
            # Normalize
            norm = sum(x*x for x in vector) ** 0.5
            if norm > 0:
                vector = [x/norm for x in vector]
            vectors.append(vector)
        return vectors

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        embeddings = self.encode([text1, text2])
        if len(embeddings) != 2:
            return 0.0

        vec1, vec2 = embeddings[0], embeddings[1]
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)


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

        # Enhanced configuration
        researcher_config = config.get("agents", {}).get("researcher", {}) if config else {}
        self.max_sources = researcher_config.get("max_sources", 10)
        self.max_iterations = researcher_config.get("max_iterations", 2)
        self.min_relevance_score = researcher_config.get("min_relevance_score", 0.3)
        self.deduplication_threshold = researcher_config.get("deduplication_threshold", 0.85)

        # Initialize embedding model for relevance and deduplication
        self.embedding_model = EmbeddingModel()

        # Venue reputation scores (top-tier venues get higher scores)
        self.venue_reputation = {
            "Nature": 1.0, "Science": 1.0, "Cell": 1.0,
            "ACM CHI": 0.95, "ACM UIST": 0.95, "IEEE VIS": 0.95,
            "ACM SIGGRAPH": 0.95, "NeurIPS": 0.95, "ICML": 0.95,
            "ICLR": 0.95, "AAAI": 0.9, "IJCAI": 0.9,
            "ACM": 0.85, "IEEE": 0.85, "Springer": 0.8,
        }

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
        Gather research evidence based on a plan with enhanced search strategy.

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
            - evidence: Extracted evidence (claims, statistics, quotes)
            - conflicts: Conflicting evidence identified
        """
        self.logger.info("Starting enhanced research gathering...")

        # Extract initial search queries
        initial_queries = context.get("search_queries", []) if context else []
        if not initial_queries:
            initial_queries = self._extract_queries_from_plan(research_plan)
        if not initial_queries:
            initial_queries = [research_plan]  # Use plan as query

        # A. Multi-query generation: expand original query
        expanded_queries = self._generate_multi_queries(initial_queries, research_plan)
        self.logger.info(f"Generated {len(expanded_queries)} expanded queries from {len(initial_queries)} initial queries")

        # Gather sources with iterative refinement
        all_sources = []
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            self.logger.info(f"Research iteration {iteration}/{self.max_iterations}")

            # Search with current queries
            new_sources = self._gather_sources(expanded_queries, context)
            all_sources.extend(new_sources)

            # B. Iterative refinement: adjust queries based on results
            if iteration < self.max_iterations and new_sources:
                expanded_queries = self._refine_queries(
                    expanded_queries,
                    new_sources,
                    research_plan
                )
                self.logger.info(f"Refined to {len(expanded_queries)} queries for next iteration")
            else:
                break

        # Convert to Source objects
        sources = self._create_source_objects(all_sources)

        # Score relevance using embeddings
        sources = self._score_relevance(sources, research_plan)

        # Score quality metrics
        sources = self._score_quality(sources)

        # Apply recency weighting
        sources = self._apply_recency_weighting(sources, context)

        # Deduplication
        sources = self._deduplicate_sources(sources)

        # Source diversity balancing
        sources = self._balance_source_diversity(sources)

        # Coverage analysis
        coverage_analysis = self._analyze_coverage(sources, research_plan)

        # Rank and filter
        sources = self._rank_and_filter(sources)

        # Extract evidence
        sources = self._extract_evidence(sources, research_plan)

        # Extract methodology from papers
        sources = self._extract_methodology(sources)

        # Identify conflicts
        conflicts = self._identify_conflicts(sources)

        # Calculate source reliability
        sources = self._calculate_reliability(sources)

        # Format for return
        findings = self._format_findings(sources)

        self.logger.info(f"Gathered {len(sources)} high-quality sources with {len([e for s in sources for e in s.evidence])} evidence items")

        return {
            "findings": findings,
            "sources": [self._source_to_dict(s) for s in sources],
            "web_results": [s for s in sources if s.type == "web"],
            "paper_results": [s for s in sources if s.type == "paper"],
            "evidence": [self._evidence_to_dict(e) for s in sources for e in s.evidence],
            "conflicts": conflicts,
            "coverage_analysis": coverage_analysis,
            "metadata": {
                "num_sources": len(sources),
                "queries_used": expanded_queries,
                "iterations": iteration,
                "evidence_count": sum(len(s.evidence) for s in sources),
                "conflict_count": len(conflicts)
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
        return queries[:3] if queries else [plan[:200]]  # Limit to 3 queries or use plan

    def _generate_multi_queries(self, initial_queries: List[str], research_plan: str) -> List[str]:
        """
        A. Multi-query generation: expand the original query into related queries.

        Generates variations:
        - Synonyms and related terms
        - Different aspects/perspectives
        - Broader and narrower scopes
        """
        expanded = list(initial_queries)  # Start with original queries

        for query in initial_queries:
            # Generate variations
            words = query.lower().split()

            # Add "what is", "how does", "why", "when" variations
            question_words = ["what", "how", "why", "when", "where", "who"]
            if not any(q in query.lower() for q in question_words):
                expanded.extend([
                    f"what is {query}",
                    f"how does {query} work",
                    f"why {query}",
                ])

            # Add related terms (simple keyword expansion)
            # In a production system, this could use word embeddings or a thesaurus
            if len(words) > 1:
                # Try different combinations
                if len(words) >= 3:
                    expanded.append(" ".join(words[:2]))  # Shorter version
                    expanded.append(" ".join(words[-2:]))  # Last two words

            # Add "research", "study", "analysis" variants
            if "research" not in query.lower():
                expanded.append(f"{query} research")
                expanded.append(f"{query} study")

        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in expanded:
            q_lower = q.lower().strip()
            if q_lower and q_lower not in seen:
                seen.add(q_lower)
                unique_queries.append(q)

        return unique_queries[:10]  # Limit to 10 queries

    def _refine_queries(
        self,
        current_queries: List[str],
        results: List[Dict[str, Any]],
        research_plan: str
    ) -> List[str]:
        """
        B. Iterative refinement: adjust queries based on initial results.

        Analyzes results to identify:
        - Missing aspects
        - Underrepresented topics
        - New related terms found in results
        """
        if not results:
            return current_queries

        # Extract keywords from successful results
        result_texts = []
        for r in results[:5]:  # Analyze top 5 results
            text = f"{r.get('title', '')} {r.get('snippet', '')} {r.get('abstract', '')}"
            result_texts.append(text)

        # Find common terms in results (potential expansion terms)
        all_words = defaultdict(int)
        for text in result_texts:
            words = re.findall(r'\b\w{4,}\b', text.lower())  # Words with 4+ chars
            for word in words:
                if word not in ['that', 'this', 'with', 'from', 'have', 'been', 'were', 'will']:
                    all_words[word] += 1

        # Get top terms not in current queries
        current_query_words = set()
        for q in current_queries:
            current_query_words.update(q.lower().split())

        new_terms = [word for word, count in sorted(all_words.items(), key=lambda x: -x[1])[:5]
                    if word not in current_query_words]

        # Create refined queries
        refined = list(current_queries)
        if new_terms:
            # Add queries with new terms
            for term in new_terms[:3]:
                refined.append(f"{research_plan[:50]} {term}")

        return refined[:10]  # Limit to 10 queries

    def _gather_sources(
        self,
        queries: List[str],
        context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Gather sources from web and paper search."""
        all_results = []

        # Balance between web and paper searches
        web_queries = queries[:len(queries)//2 + 1]  # Slightly more web queries
        paper_queries = queries[:len(queries)//2 + 1]

        # Web search
        for query in web_queries[:5]:  # Limit to 5 queries per iteration
            try:
                web_res = asyncio.run(self.web_search_tool.search(query))
                for res in web_res:
                    res['query_used'] = query
                    res['source_type'] = 'web'
                all_results.extend(web_res)
            except Exception as e:
                self.logger.error(f"Web search error: {e}")

        # Paper search
        year_from = context.get("year_from") if context else None
        for query in paper_queries[:5]:  # Limit to 5 queries per iteration
            try:
                paper_res = asyncio.run(
                    self.paper_search_tool.search(query, year_from=year_from)
                )
                for res in paper_res:
                    res['query_used'] = query
                    res['source_type'] = 'paper'
                all_results.extend(paper_res)
            except Exception as e:
                self.logger.error(f"Paper search error: {e}")

        return all_results

    def _create_source_objects(self, results: List[Dict[str, Any]]) -> List[Source]:
        """Convert raw search results to Source objects."""
        sources = []
        for i, result in enumerate(results):
            source_id = f"source_{i}_{result.get('source_type', 'unknown')}"

            if result.get('source_type') == 'web':
                source = Source(
                    id=source_id,
                    type="web",
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    content=result.get("snippet", ""),
                    published_date=result.get("published_date"),
                )
            else:  # paper
                source = Source(
                    id=source_id,
                    type="paper",
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    content=result.get("abstract", ""),
                    year=result.get("year"),
                    citation_count=result.get("citation_count", 0),
                    venue=result.get("venue", ""),
                    authors=[a.get("name", "") if isinstance(a, dict) else str(a)
                            for a in result.get("authors", [])],
                )

            sources.append(source)

        return sources

    def _score_relevance(self, sources: List[Source], research_plan: str) -> List[Source]:
        """
        B. Relevance scoring using embeddings or keyword matching.
        """
        if not sources:
            return sources

        # Create text representations for similarity
        source_texts = [f"{s.title} {s.content}" for s in sources]

        # Compute similarity to research plan
        for i, source in enumerate(sources):
            similarity = self.embedding_model.similarity(
                source_texts[i],
                research_plan
            )
            source.relevance_score = similarity

        return sources

    def _score_quality(self, sources: List[Source]) -> List[Source]:
        """
        B. Quality metrics: citation count, venue reputation, author reputation.
        """
        for source in sources:
            quality_score = 0.0

            if source.type == "paper":
                # Citation count (normalized, assuming max ~10000)
                citation_score = min(source.citation_count / 100.0, 1.0) * 0.4
                quality_score += citation_score

                # Venue reputation
                venue_score = 0.0
                for venue_key, score in self.venue_reputation.items():
                    if venue_key.lower() in source.venue.lower():
                        venue_score = score * 0.4
                        break
                quality_score += venue_score

                # Author count (more authors might indicate collaboration)
                author_score = min(len(source.authors) / 5.0, 1.0) * 0.2
                quality_score += author_score

            else:  # web source
                # For web sources, use domain reputation (simplified)
                domain = source.url.split('/')[2] if '/' in source.url else ""
                reputable_domains = ['edu', 'gov', 'org', 'ac.uk', 'ac.jp']
                if any(d in domain for d in reputable_domains):
                    quality_score = 0.6
                else:
                    quality_score = 0.4

            source.quality_score = quality_score

        return sources

    def _apply_recency_weighting(
        self,
        sources: List[Source],
        context: Optional[Dict[str, Any]]
    ) -> List[Source]:
        """
        A. Recency weighting: prioritize recent sources when relevant.
        """
        current_year = datetime.now().year

        for source in sources:
            recency_weight = 1.0

            if source.year:
                age = current_year - source.year
                if age <= 1:
                    recency_weight = 1.2
                elif age <= 3:
                    recency_weight = 1.1
                elif age <= 5:
                    recency_weight = 1.0
                elif age <= 10:
                    recency_weight = 0.9
                else:
                    recency_weight = 0.8
            elif source.published_date:
                # Try to parse date (simplified)
                try:
                    # Assume format like "2024-01-15" or "2 days ago"
                    if "ago" in source.published_date.lower():
                        recency_weight = 1.15  # Very recent
                    else:
                        recency_weight = 1.0
                except:
                    recency_weight = 1.0

            # Apply recency weight to relevance score
            source.relevance_score *= recency_weight

        return sources

    def _deduplicate_sources(self, sources: List[Source]) -> List[Source]:
        """
        B. Deduplication: remove near-duplicate sources.
        """
        if len(sources) <= 1:
            return sources

        unique_sources = []
        seen_urls = set()

        for source in sources:
            # Check URL exact match
            if source.url in seen_urls:
                continue
            seen_urls.add(source.url)

            # Check content similarity with existing sources
            is_duplicate = False
            source_text = f"{source.title} {source.content[:200]}"

            for existing in unique_sources:
                existing_text = f"{existing.title} {existing.content[:200]}"
                similarity = self.embedding_model.similarity(source_text, existing_text)

                if similarity > self.deduplication_threshold:
                    is_duplicate = True
                    # Keep the one with higher quality score
                    if source.quality_score > existing.quality_score:
                        unique_sources.remove(existing)
                        unique_sources.append(source)
                    break

            if not is_duplicate:
                unique_sources.append(source)

        return unique_sources

    def _balance_source_diversity(self, sources: List[Source]) -> List[Source]:
        """
        A. Source diversity: balance academic papers, web articles, and other sources.
        """
        web_sources = [s for s in sources if s.type == "web"]
        paper_sources = [s for s in sources if s.type == "paper"]

        # Target: 40% papers, 60% web (adjustable)
        target_paper_ratio = 0.4
        target_count = min(len(sources), self.max_sources)
        target_papers = int(target_count * target_paper_ratio)
        target_web = target_count - target_papers

        # Sort by combined score
        web_sources.sort(key=lambda s: s.relevance_score + s.quality_score, reverse=True)
        paper_sources.sort(key=lambda s: s.relevance_score + s.quality_score, reverse=True)

        # Select balanced mix
        balanced = []
        balanced.extend(paper_sources[:target_papers])
        balanced.extend(web_sources[:target_web])

        # Fill remaining slots with best overall
        remaining = [s for s in sources if s not in balanced]
        remaining.sort(key=lambda s: s.relevance_score + s.quality_score, reverse=True)
        balanced.extend(remaining[:target_count - len(balanced)])

        return balanced[:self.max_sources]

    def _analyze_coverage(
        self,
        sources: List[Source],
        research_plan: str
    ) -> Dict[str, Any]:
        """
        B. Coverage analysis: ensure diverse perspectives.
        """
        # Extract key topics from research plan
        plan_words = set(re.findall(r'\b\w{4,}\b', research_plan.lower()))

        # Check which topics are covered
        covered_topics = set()
        for source in sources:
            source_text = f"{source.title} {source.content}".lower()
            source_words = set(re.findall(r'\b\w{4,}\b', source_text))
            covered_topics.update(plan_words.intersection(source_words))

        coverage_ratio = len(covered_topics) / len(plan_words) if plan_words else 0.0

        # Check source type diversity
        type_distribution = defaultdict(int)
        for source in sources:
            type_distribution[source.type] += 1

        return {
            "topic_coverage": coverage_ratio,
            "topics_covered": len(covered_topics),
            "total_topics": len(plan_words),
            "type_distribution": dict(type_distribution),
            "diversity_score": len(set(type_distribution.values())) / max(len(sources), 1)
        }

    def _rank_and_filter(self, sources: List[Source]) -> List[Source]:
        """Rank sources by combined score and filter by minimum relevance."""
        # Combined score: relevance (60%) + quality (40%)
        for source in sources:
            source.relevance_score = source.relevance_score  # Already weighted

        # Sort by combined score
        sources.sort(
            key=lambda s: s.relevance_score * 0.6 + s.quality_score * 0.4,
            reverse=True
        )

        # Filter by minimum relevance
        filtered = [s for s in sources if s.relevance_score >= self.min_relevance_score]

        return filtered[:self.max_sources]

    def _extract_evidence(
        self,
        sources: List[Source],
        research_plan: str
    ) -> List[Source]:
        """
        C. Evidence extraction: extract key claims, statistics, and quotes.
        """
        for source in sources:
            evidence_list = []
            text = source.content

            # Extract statistics (numbers with units or percentages)
            stat_patterns = [
                r'\d+%',  # Percentages
                r'\d+\.\d+%',  # Decimal percentages
                r'\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|thousand)',  # Large numbers
                r'\d+\.\d+\s*(?:times|fold|×)',  # Multipliers
            ]

            for pattern in stat_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    context_start = max(0, match.start() - 50)
                    context_end = min(len(text), match.end() + 50)
                    context = text[context_start:context_end]

                    evidence = Evidence(
                        claim=context,
                        source_id=source.id,
                        evidence_type="statistic",
                        value=match.group(),
                        context=context,
                    )
                    evidence_list.append(evidence)

            # Extract quotes (text in quotes)
            quote_pattern = r'"([^"]{20,200})"'
            quotes = re.findall(quote_pattern, text)
            for quote in quotes[:3]:  # Limit to 3 quotes per source
                evidence = Evidence(
                    claim=quote,
                    source_id=source.id,
                    evidence_type="quote",
                    context=text[:200],
                )
                evidence_list.append(evidence)

            # Extract key claims (sentences with important keywords)
            sentences = re.split(r'[.!?]+', text)
            important_keywords = re.findall(r'\b\w{5,}\b', research_plan.lower())

            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in important_keywords[:5]):
                    if len(sentence.strip()) > 20:  # Meaningful length
                        evidence = Evidence(
                            claim=sentence.strip(),
                            source_id=source.id,
                            evidence_type="claim",
                            context=sentence[:200],
                        )
                        evidence_list.append(evidence)
                        if len(evidence_list) >= 5:  # Limit evidence per source
                            break

            source.evidence = evidence_list[:5]  # Limit to 5 evidence items per source

        return sources

    def _extract_methodology(self, sources: List[Source]) -> List[Source]:
        """
        C. Extract methodology information from papers.
        """
        methodology_keywords = [
            'methodology', 'method', 'approach', 'experiment', 'study design',
            'participants', 'sample size', 'data collection', 'analysis',
            'procedure', 'protocol', 'framework', 'model', 'algorithm'
        ]

        for source in sources:
            if source.type == "paper":
                text = f"{source.title} {source.content}".lower()

                # Find sentences containing methodology keywords
                sentences = re.split(r'[.!?]+', source.content)
                methodology_sentences = []

                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    if any(keyword in sentence_lower for keyword in methodology_keywords):
                        methodology_sentences.append(sentence.strip())

                if methodology_sentences:
                    source.methodology = " ".join(methodology_sentences[:3])  # First 3 sentences

        return sources

    def _identify_conflicts(self, sources: List[Source]) -> List[Dict[str, Any]]:
        """
        C. Identify conflicting evidence.
        """
        conflicts = []

        # Group evidence by topic
        evidence_by_topic = defaultdict(list)
        for source in sources:
            for evidence in source.evidence:
                # Simple topic extraction (first few words of claim)
                topic = " ".join(evidence.claim.split()[:3]).lower()
                evidence_by_topic[topic].append((source, evidence))

        # Check for conflicting claims
        for topic, evidence_list in evidence_by_topic.items():
            if len(evidence_list) < 2:
                continue

            # Compare evidence items for conflicts
            for i, (source1, ev1) in enumerate(evidence_list):
                for source2, ev2 in evidence_list[i+1:]:
                    # Check for negation or contradiction
                    similarity = self.embedding_model.similarity(ev1.claim, ev2.claim)

                    # Low similarity might indicate conflict
                    if similarity < 0.3:
                        # Check for explicit contradictions
                        negation_words = ['not', 'no', 'never', 'none', 'cannot', 'disagree', 'contradict']
                        if any(neg in ev1.claim.lower() for neg in negation_words) or \
                           any(neg in ev2.claim.lower() for neg in negation_words):
                            conflicts.append({
                                "topic": topic,
                                "evidence1": {
                                    "claim": ev1.claim,
                                    "source": source1.title,
                                    "source_id": source1.id
                                },
                                "evidence2": {
                                    "claim": ev2.claim,
                                    "source": source2.title,
                                    "source_id": source2.id
                                },
                                "conflict_type": "contradiction"
                            })

        return conflicts

    def _calculate_reliability(self, sources: List[Source]) -> List[Source]:
        """
        C. Track source reliability scores.
        """
        for source in sources:
            reliability = 0.0

            if source.type == "paper":
                # Papers are generally more reliable
                base_reliability = 0.7

                # Boost for high citations
                if source.citation_count > 50:
                    base_reliability += 0.15
                elif source.citation_count > 10:
                    base_reliability += 0.1

                # Boost for reputable venue
                for venue_key, score in self.venue_reputation.items():
                    if venue_key.lower() in source.venue.lower():
                        base_reliability += 0.1
                        break

                # Boost for recent papers (more current)
                if source.year and source.year >= datetime.now().year - 3:
                    base_reliability += 0.05

                reliability = min(base_reliability, 1.0)

            else:  # web source
                # Web sources are less reliable by default
                base_reliability = 0.5

                # Boost for .edu, .gov domains
                domain = source.url.split('/')[2] if '/' in source.url else ""
                if '.edu' in domain or '.gov' in domain:
                    base_reliability += 0.2
                elif '.org' in domain:
                    base_reliability += 0.1

                reliability = min(base_reliability, 1.0)

            source.reliability_score = reliability

            # Update evidence reliability based on source
            for evidence in source.evidence:
                evidence.reliability_score = reliability

        return sources

    def _format_findings(self, sources: List[Source]) -> List[Dict[str, Any]]:
        """Format sources as findings for backward compatibility."""
        findings = []
        for source in sources:
            finding = {
                "type": source.type,
                "title": source.title,
                "url": source.url,
                "content": source.content,
                "relevance_score": source.relevance_score,
                "quality_score": source.quality_score,
                "reliability_score": source.reliability_score,
                "published_date": source.published_date,
            }

            if source.type == "paper":
                finding.update({
                    "authors": source.authors,
                    "year": source.year,
                    "citation_count": source.citation_count,
                    "venue": source.venue,
                    "methodology": source.methodology,
                })

            findings.append(finding)

        return findings

    def _source_to_dict(self, source: Source) -> Dict[str, Any]:
        """Convert Source object to dictionary."""
        return {
            "id": source.id,
            "type": source.type,
            "title": source.title,
            "url": source.url,
            "content": source.content,
            "relevance_score": source.relevance_score,
            "quality_score": source.quality_score,
            "reliability_score": source.reliability_score,
            "year": source.year,
            "citation_count": source.citation_count,
            "venue": source.venue,
            "authors": source.authors,
            "published_date": source.published_date,
            "methodology": source.methodology,
            "evidence_count": len(source.evidence),
        }

    def _evidence_to_dict(self, evidence: Evidence) -> Dict[str, Any]:
        """Convert Evidence object to dictionary."""
        return {
            "claim": evidence.claim,
            "source_id": evidence.source_id,
            "evidence_type": evidence.evidence_type,
            "value": evidence.value,
            "context": evidence.context,
            "reliability_score": evidence.reliability_score,
        }


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
