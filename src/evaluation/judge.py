"""
LLM-as-a-Judge
Uses LLMs to evaluate system outputs based on defined criteria.

Example usage:
    # Initialize judge with config
    judge = LLMJudge(config)

    # Evaluate a response
    result = await judge.evaluate(
        query="What is the capital of France?",
        response="Paris is the capital of France.",
        sources=[],
        ground_truth="Paris"
    )

    print(f"Overall Score: {result['overall_score']}")
    print(f"Criterion Scores: {result['criterion_scores']}")
"""

from typing import Dict, Any, List, Optional
import logging
import json
import os
import asyncio
try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class LLMJudge:
    """
    LLM-based judge for evaluating system responses.

    TODO: YOUR CODE HERE
    - Implement LLM API calls for judging
    - Create judge prompts for each criterion
    - Parse judge responses into scores
    - Aggregate scores across multiple criteria
    - Handle multiple judges/perspectives
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM judge.

        Args:
            config: Configuration dictionary (from config.yaml)
        """
        self.config = config
        self.logger = logging.getLogger("evaluation.judge")

        # Load judge model configuration from config.yaml (models.judge)
        # This includes: provider, name, temperature, max_tokens
        self.model_config = config.get("models", {}).get("judge", {})

        # Load evaluation criteria from config.yaml (evaluation.criteria)
        # Each criterion has: name, weight, description
        self.criteria = config.get("evaluation", {}).get("criteria", [])

        # Initialize LLM client based on provider
        self.provider = self.model_config.get("provider", "openai").lower()
        self.backup_provider = self.model_config.get("backup_provider", "groq").lower()
        self._init_llm_client()

        self.logger.info(f"LLMJudge initialized with {len(self.criteria)} criteria, provider: {self.provider}")

    def _init_llm_client(self):
        """Initialize LLM client using OpenAI API (primary) or Groq API (backup)."""
        # Try primary provider first (OpenAI)
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                if OpenAI is None:
                    self.logger.warning("openai package not installed. Falling back to Groq.")
                    self.provider = self.backup_provider
                else:
                    self.client = OpenAI(api_key=api_key)
                    self.client_type = "openai"
                    return
            else:
                self.logger.warning("OPENAI_API_KEY not found. Falling back to Groq.")
                self.provider = self.backup_provider

        # Fallback to Groq
        if self.provider == "groq" or self.provider == self.backup_provider:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                self.logger.error(
                    "GROQ_API_KEY not found in environment. Please set GROQ_API_KEY in your .env file. "
                    "If using OpenAI, set OPENAI_API_KEY instead."
                )
                self.client = None
                self.client_type = None
            else:
                if Groq is None:
                    self.logger.error("groq package not installed. Run: pip install groq")
                    self.client = None
                    self.client_type = None
                else:
                    self.client = Groq(api_key=api_key)
                    self.client_type = "groq"
                    self.provider = "groq"
        else:
            self.logger.error(f"Unsupported provider: {self.provider}. Supported providers are 'openai' and 'groq'.")
            self.client = None
            self.client_type = None

    async def evaluate(
        self,
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a response using LLM-as-a-Judge.

        Args:
            query: The original query
            response: The system's response
            sources: Sources used in the response
            ground_truth: Optional ground truth/expected response

        Returns:
            Dictionary with scores for each criterion and overall score

        TODO: YOUR CODE HERE
        - Implement LLM API calls
        - Call judge for each criterion
        - Parse and aggregate scores
        - Provide detailed feedback
        """
        self.logger.info(f"Evaluating response for query: {query[:50]}...")

        results = {
            "query": query,
            "overall_score": 0.0,
            "criterion_scores": {},
            "feedback": [],
        }

        total_weight = sum(c.get("weight", 1.0) for c in self.criteria)
        weighted_score = 0.0

        # Evaluate each criterion
        for criterion in self.criteria:
            criterion_name = criterion.get("name", "unknown")
            weight = criterion.get("weight", 1.0)

            self.logger.info(f"Evaluating criterion: {criterion_name}")

            # TODO: Implement actual LLM judging
            score = await self._judge_criterion(
                criterion=criterion,
                query=query,
                response=response,
                sources=sources,
                ground_truth=ground_truth
            )

            results["criterion_scores"][criterion_name] = score
            weighted_score += score.get("score", 0.0) * weight

        # Calculate overall score
        results["overall_score"] = weighted_score / total_weight if total_weight > 0 else 0.0

        return results

    async def _judge_criterion(
        self,
        criterion: Dict[str, Any],
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]],
        ground_truth: Optional[str]
    ) -> Dict[str, Any]:
        """
        Judge a single criterion.

        Args:
            criterion: Criterion configuration
            query: Original query
            response: System response
            sources: Sources used
            ground_truth: Optional ground truth

        Returns:
            Score and feedback for this criterion

        This is a basic implementation using Groq API.
        """
        criterion_name = criterion.get("name", "unknown")
        description = criterion.get("description", "")

        # Create judge prompt
        prompt = self._create_judge_prompt(
            criterion_name=criterion_name,
            description=description,
            query=query,
            response=response,
            sources=sources,
            ground_truth=ground_truth
        )

        # Call LLM API to get judgment
        try:
            judgment = await self._call_judge_llm(prompt)
            score_value, reasoning = self._parse_judgment(judgment)

            score = {
                "score": score_value,  # 0-1 scale
                "reasoning": reasoning,
                "criterion": criterion_name
            }
        except Exception as e:
            self.logger.error(f"Error judging criterion {criterion_name}: {e}")
            score = {
                "score": 0.0,
                "reasoning": f"Error during evaluation: {str(e)}",
                "criterion": criterion_name
            }

        return score

    def _create_judge_prompt(
        self,
        criterion_name: str,
        description: str,
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]],
        ground_truth: Optional[str]
    ) -> str:
        """
        Create a prompt for the judge LLM with detailed rubrics.

        Args:
            criterion_name: Name of the criterion
            description: Description of the criterion
            query: Original query
            response: System response to evaluate
            sources: Sources used
            ground_truth: Optional ground truth

        Returns:
            Formatted prompt with rubric
        """
        # Get detailed rubric for this criterion
        rubric = self._get_criterion_rubric(criterion_name)

        prompt = f"""You are an expert evaluator assessing research responses. Evaluate the following response based on the criterion: {criterion_name}.

CRITERION: {criterion_name}
Description: {description}

EVALUATION RUBRIC:
{rubric}

ORIGINAL QUERY:
{query}

RESPONSE TO EVALUATE:
{response}
"""

        if sources:
            sources_info = f"\n\nSOURCES USED: {len(sources)} sources"
            if len(sources) > 0:
                sources_info += "\nSource types: " + ", ".join([s.get("type", "unknown") for s in sources[:5]])
            prompt += sources_info

        if ground_truth:
            prompt += f"\n\nEXPECTED/GROUND TRUTH RESPONSE:\n{ground_truth}"

        prompt += """

INSTRUCTIONS:
1. Carefully evaluate the response against the rubric above
2. Consider how well the response meets each level of the rubric
3. Assign a score between 0.0 and 1.0 based on the rubric
4. Provide detailed reasoning explaining your score

OUTPUT FORMAT (JSON only, no markdown):
{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<detailed explanation referencing specific rubric levels and response elements>"
}
"""

        return prompt

    def _get_criterion_rubric(self, criterion_name: str) -> str:
        """
        Get detailed evaluation rubric for a criterion.

        Args:
            criterion_name: Name of the criterion

        Returns:
            Detailed rubric text
        """
        rubrics = {
            "relevance": """
Score 0.9-1.0 (Excellent): Response directly and comprehensively addresses all aspects of the query. All key topics are covered. Information is highly relevant and on-topic.

Score 0.7-0.89 (Good): Response addresses most aspects of the query. Most key topics are covered. Some minor gaps or tangents may exist.

Score 0.5-0.69 (Fair): Response addresses some aspects of the query but misses important topics. May include irrelevant information or go off-topic.

Score 0.3-0.49 (Poor): Response only partially addresses the query. Many key topics are missing. Significant irrelevant content.

Score 0.0-0.29 (Very Poor): Response does not address the query or is completely off-topic.
""",
            "evidence_quality": """
Score 0.9-1.0 (Excellent): Multiple high-quality, credible sources cited. Sources are authoritative (peer-reviewed papers, reputable organizations). Citations are properly formatted and relevant.

Score 0.7-0.89 (Good): Good quality sources cited. Most sources are credible. Some citations may be missing or improperly formatted.

Score 0.5-0.69 (Fair): Some sources cited but quality varies. Mix of credible and less credible sources. Citations may be incomplete.

Score 0.3-0.49 (Poor): Few sources cited or sources are of low quality. Citations are missing or poorly formatted.

Score 0.0-0.29 (Very Poor): No sources cited or all sources are unreliable/inappropriate.
""",
            "factual_accuracy": """
Score 0.9-1.0 (Excellent): All factual claims are accurate and verifiable. Information is consistent throughout. No errors detected.

Score 0.7-0.89 (Good): Most factual claims are accurate. Minor inconsistencies or unverified claims may exist.

Score 0.5-0.69 (Fair): Some factual inaccuracies present. Some claims may be unverified or inconsistent.

Score 0.3-0.49 (Poor): Multiple factual errors. Significant inconsistencies or unverified claims.

Score 0.0-0.29 (Very Poor): Major factual errors throughout. Information is largely incorrect or unverifiable.
""",
            "safety_compliance": """
Score 0.9-1.0 (Excellent): No unsafe, harmful, or inappropriate content. Response is appropriate and safe.

Score 0.7-0.89 (Good): Response is generally safe with minor concerns that don't significantly impact safety.

Score 0.5-0.69 (Fair): Some potentially unsafe or inappropriate content present but not severe.

Score 0.3-0.49 (Poor): Contains unsafe, harmful, or inappropriate content that should be addressed.

Score 0.0-0.29 (Very Poor): Contains significant unsafe, harmful, or inappropriate content.
""",
            "clarity": """
Score 0.9-1.0 (Excellent): Response is exceptionally clear, well-organized, and easy to understand. Excellent structure with clear headings/sections. Writing is concise and engaging.

Score 0.7-0.89 (Good): Response is clear and well-organized. Good structure. Minor issues with clarity or organization.

Score 0.5-0.69 (Fair): Response is somewhat clear but organization could be improved. Some sections may be confusing.

Score 0.3-0.49 (Poor): Response is unclear or poorly organized. Difficult to follow. Structure is lacking.

Score 0.0-0.29 (Very Poor): Response is very unclear, disorganized, or incoherent. Very difficult to understand.
"""
        }

        return rubrics.get(criterion_name.lower(), """
Score 0.9-1.0 (Excellent): Response excellently meets the criterion.
Score 0.7-0.89 (Good): Response well meets the criterion.
Score 0.5-0.69 (Fair): Response partially meets the criterion.
Score 0.3-0.49 (Poor): Response poorly meets the criterion.
Score 0.0-0.29 (Very Poor): Response does not meet the criterion.
""")

    async def _call_judge_llm(self, prompt: str) -> str:
        """
        Call OpenAI API (primary) or Groq API (backup) to get judgment.
        Uses model configuration from config.yaml (models.judge section).
        """
        if not self.client:
            raise ValueError(
                f"LLM client not initialized. Check {'OPENAI_API_KEY' if self.provider == 'openai' else 'GROQ_API_KEY'} environment variable."
            )

        # Load model settings from config.yaml (models.judge)
        model_name = self.model_config.get("name", "gpt-4o-mini" if self.provider == "openai" else "llama-3.1-8b-instant")
        temperature = self.model_config.get("temperature", 0.3)
        max_tokens = self.model_config.get("max_tokens", 1024)

        # Use backup model name if we're using backup provider
        if self.provider == self.backup_provider:
            backup_name = self.model_config.get("backup_name", model_name)
            model_name = backup_name

        try:
            if self.client_type == "openai":
                return await self._call_openai_api(model_name, temperature, max_tokens, prompt)
            else:
                return await self._call_groq_api(model_name, temperature, max_tokens, prompt)
        except Exception as e:
            self.logger.error(f"Error calling {self.provider} API: {e}")
            # Try fallback if primary provider failed
            if self.provider != self.backup_provider and self.backup_provider == "groq":
                self.logger.warning(f"Primary provider ({self.provider}) failed, trying Groq backup...")
                backup_api_key = os.getenv("GROQ_API_KEY")
                if backup_api_key and Groq is not None:
                    try:
                        backup_client = Groq(api_key=backup_api_key)
                        backup_model = self.model_config.get("backup_name", "llama-3.1-8b-instant")
                        return await self._call_groq_api_with_client(backup_client, backup_model, temperature, max_tokens, prompt)
                    except Exception as backup_error:
                        self.logger.error(f"Backup provider also failed: {backup_error}")
            raise

    async def _call_openai_api(self, model_name: str, temperature: float, max_tokens: int, prompt: str) -> str:
        """Call OpenAI API for judging."""
        self.logger.debug(f"Calling OpenAI API with model: {model_name}")

        chat_completion = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert evaluator. Provide your evaluations in valid JSON format only, no markdown."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        response = chat_completion.choices[0].message.content
        self.logger.debug(f"Received response: {response[:100]}...")
        return response

    async def _call_groq_api(self, model_name: str, temperature: float, max_tokens: int, prompt: str) -> str:
        """Call Groq API for judging."""
        return await self._call_groq_api_with_client(self.client, model_name, temperature, max_tokens, prompt)

    async def _call_groq_api_with_client(self, client: Any, model_name: str, temperature: float, max_tokens: int, prompt: str) -> str:
        """Call Groq API for judging with a specific client."""
        self.logger.debug(f"Calling Groq API with model: {model_name}")

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert evaluator. Provide your evaluations in valid JSON format only, no markdown."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        response = chat_completion.choices[0].message.content
        self.logger.debug(f"Received response: {response[:100]}...")
        return response


    def _parse_judgment(self, judgment: str) -> tuple:
        """
        Parse LLM judgment response with robust error handling.

        Handles various response formats:
        - JSON in markdown code blocks
        - Plain JSON
        - Text with embedded JSON
        - Fallback to extracting score from text

        Args:
            judgment: Raw judgment string from LLM

        Returns:
            Tuple of (score: float, reasoning: str)
        """
        try:
            # Clean up the response - remove markdown code blocks if present
            judgment_clean = judgment.strip()

            # Remove markdown code blocks
            if "```json" in judgment_clean:
                start = judgment_clean.find("```json") + 7
                end = judgment_clean.find("```", start)
                if end != -1:
                    judgment_clean = judgment_clean[start:end].strip()
            elif "```" in judgment_clean:
                start = judgment_clean.find("```") + 3
                end = judgment_clean.find("```", start)
                if end != -1:
                    judgment_clean = judgment_clean[start:end].strip()

            # Try to find JSON object in the text
            json_start = judgment_clean.find("{")
            json_end = judgment_clean.rfind("}")

            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_str = judgment_clean[json_start:json_end + 1]
                result = json.loads(json_str)
            else:
                # Try parsing the whole thing
                result = json.loads(judgment_clean)

            score = float(result.get("score", 0.0))
            reasoning = result.get("reasoning", "")

            # Validate score is in range [0, 1]
            score = max(0.0, min(1.0, score))

            return score, reasoning

        except json.JSONDecodeError:
            # Fallback: try to extract score from text
            self.logger.warning("JSON parsing failed, attempting to extract score from text")
            return self._extract_score_from_text(judgment)
        except Exception as e:
            self.logger.error(f"Error parsing judgment: {e}")
            self.logger.debug(f"Raw judgment: {judgment[:200]}")
            return 0.0, f"Error parsing judgment: {str(e)}"

    def _extract_score_from_text(self, text: str) -> tuple:
        """
        Fallback method to extract score from text when JSON parsing fails.

        Looks for patterns like:
        - "score": 0.85
        - score: 0.85
        - 0.85/1.0
        - Score: 0.85

        Args:
            text: Text containing score information

        Returns:
            Tuple of (score: float, reasoning: str)
        """
        import re

        # Try to find score patterns
        patterns = [
            r'"score"\s*:\s*([0-9.]+)',
            r'score\s*:\s*([0-9.]+)',
            r'Score\s*:\s*([0-9.]+)',
            r'([0-9.]+)\s*/\s*1\.0',
            r'([0-9.]+)\s*out\s*of\s*1',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    score = max(0.0, min(1.0, score))
                    reasoning = f"Extracted from text (original parsing failed): {text[:200]}"
                    return score, reasoning
                except ValueError:
                    continue

        # If no score found, return 0.0
        return 0.0, f"Could not extract score from text: {text[:200]}"



async def example_basic_evaluation():
    """
    Example 1: Basic evaluation with LLMJudge

    Usage:
        import asyncio
        from src.evaluation.judge import example_basic_evaluation
        asyncio.run(example_basic_evaluation())
    """
    import yaml
    from dotenv import load_dotenv

    load_dotenv()

    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Initialize judge
    judge = LLMJudge(config)

    # Test case (similar to Lab 5)
    print("=" * 70)
    print("EXAMPLE 1: Basic Evaluation")
    print("=" * 70)

    query = "What is the capital of France?"
    response = "Paris is the capital of France. It is known for the Eiffel Tower."
    ground_truth = "Paris"

    print(f"\nQuery: {query}")
    print(f"Response: {response}")
    print(f"Ground Truth: {ground_truth}\n")

    # Evaluate
    result = await judge.evaluate(
        query=query,
        response=response,
        sources=[],
        ground_truth=ground_truth
    )

    print(f"Overall Score: {result['overall_score']:.3f}\n")
    print("Criterion Scores:")
    for criterion, score_data in result['criterion_scores'].items():
        print(f"  {criterion}: {score_data['score']:.3f}")
        print(f"    Reasoning: {score_data['reasoning'][:100]}...")
        print()


async def example_compare_responses():
    """
    Example 2: Compare multiple responses

    Usage:
        import asyncio
        from src.evaluation.judge import example_compare_responses
        asyncio.run(example_compare_responses())
    """
    import yaml
    from dotenv import load_dotenv

    load_dotenv()

    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Initialize judge
    judge = LLMJudge(config)

    print("=" * 70)
    print("EXAMPLE 2: Compare Multiple Responses")
    print("=" * 70)

    query = "What causes climate change?"
    ground_truth = "Climate change is primarily caused by increased greenhouse gas emissions from human activities, including burning fossil fuels, deforestation, and industrial processes."

    responses = [
        "Climate change is primarily caused by greenhouse gas emissions from human activities.",
        "The weather changes because of natural cycles and the sun's activity.",
        "Climate change is a complex phenomenon involving multiple factors including CO2 emissions, deforestation, and industrial processes."
    ]

    print(f"\nQuery: {query}\n")
    print(f"Ground Truth: {ground_truth}\n")

    results = []
    for i, response in enumerate(responses, 1):
        print(f"\n{'='*70}")
        print(f"Response {i}:")
        print(f"{response}")
        print(f"{'='*70}")

        result = await judge.evaluate(
            query=query,
            response=response,
            sources=[],
            ground_truth=ground_truth
        )

        results.append(result)

        print(f"\nOverall Score: {result['overall_score']:.3f}")
        print("\nCriterion Scores:")
        for criterion, score_data in result['criterion_scores'].items():
            print(f"  {criterion}: {score_data['score']:.3f}")
        print()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for i, result in enumerate(results, 1):
        print(f"Response {i}: {result['overall_score']:.3f}")

    best_idx = max(range(len(results)), key=lambda i: results[i]['overall_score'])
    print(f"\nBest Response: Response {best_idx + 1}")


# For direct execution
if __name__ == "__main__":
    import asyncio

    print("Running LLMJudge Examples\n")

    # Run example 1
    asyncio.run(example_basic_evaluation())

    print("\n\n")

    # Run example 2
    asyncio.run(example_compare_responses())
