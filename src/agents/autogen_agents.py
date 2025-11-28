"""
AutoGen Agent Implementations

This module provides concrete AutoGen-based implementations of the research agents.
Each agent is implemented as an AutoGen AssistantAgent with specific tools and behaviors.

Based on the AutoGen literature review example:
https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/examples/literature-review.html
"""

import os
from typing import Dict, Any, List, Optional
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, FunctionalTermination
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily
# Import our research tools
from src.tools.web_search import web_search
from src.tools.paper_search import paper_search


def create_multi_string_termination(termination_strings: List[str]) -> FunctionalTermination:
    """
    Create a termination condition that checks for multiple termination strings.

    This allows the critic to use either "TERMINATE", "APPROVED - RESEARCH COMPLETE",
    or other custom termination signals.

    Args:
        termination_strings: List of strings that signal termination

    Returns:
        FunctionalTermination configured to check for the termination strings
    """
    def check_termination(messages: List[Any]) -> bool:
        """
        Check if any termination string appears in the recent messages.

        Args:
            messages: List of message objects from the conversation

        Returns:
            True if any termination string is found, False otherwise
        """
        # Only check the last message from Critic to avoid premature termination
        if not messages:
            return False

        # Check the last few messages for termination signals
        # Only look at messages from Critic
        for message in reversed(messages[-5:]):  # Check last 5 messages
            # Get message content as string
            content = ""
            source = ""

            if hasattr(message, 'content'):
                content = str(message.content)
            elif hasattr(message, 'text'):
                content = str(message.text)
            else:
                content = str(message)

            # Get source
            if hasattr(message, 'source'):
                source = str(message.source)
            elif hasattr(message, 'name'):
                source = str(message.name)

            # Only check termination from Critic
            if "Critic" not in source:
                continue

            # Check if any termination string appears in the content
            content_upper = content.upper()
            for term_string in termination_strings:
                if term_string.upper() in content_upper:
                    return True

        return False

    return FunctionalTermination(check_termination)


def create_model_client(config: Dict[str, Any]) -> OpenAIChatCompletionClient:
    """
    Create model client for AutoGen agents using OpenAI API (primary) or Groq API (backup).

    Args:
        config: Configuration dictionary from config.yaml

    Returns:
        OpenAIChatCompletionClient configured for OpenAI or Groq
    """
    model_config = config.get("models", {}).get("default", {})
    provider = model_config.get("provider", "openai").lower()
    backup_provider = model_config.get("backup_provider", "groq").lower()

    import logging
    logger = logging.getLogger(__name__)

    # Try primary provider first (OpenAI)
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            model_name = model_config.get("name", "gpt-4o")
            logger.info(f"Using OpenAI as primary provider with model: {model_name}")
            logger.info(f"Function calling enabled: True")

            return OpenAIChatCompletionClient(
                model=model_name,
                api_key=api_key,
                # OpenAI uses default base_url, no need to specify
                model_capabilities={
                    "json_output": True,
                    "vision": False,
                    "function_calling": True,
                }
            )
        else:
            logger.warning("OPENAI_API_KEY not found. Falling back to Groq.")
            provider = backup_provider

    # Fallback to Groq if OpenAI is not available or if Groq is explicitly requested
    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment. Please set GROQ_API_KEY in your .env file. "
                "If using OpenAI, set OPENAI_API_KEY instead."
            )

        model_name = model_config.get("name", "llama-3.1-70b-versatile")
        if provider == backup_provider:
            # Use backup model name if we're using backup provider
            backup_name = model_config.get("backup_name", model_name)
            model_name = backup_name

        logger.info(f"Using Groq provider with model: {model_name}")
        logger.info(f"Function calling enabled: True")

        return OpenAIChatCompletionClient(
            model=model_name,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            model_capabilities={
                "json_output": False,
                "vision": False,
                "function_calling": True,
            }
        )

    raise ValueError(f"Unsupported provider: {provider}. Supported providers are 'openai' and 'groq'.")


def create_planner_agent(config: Dict[str, Any], model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    """
    Create a Planner Agent using AutoGen.

    The planner breaks down research queries into actionable steps.
    It doesn't use tools, but provides strategic direction.

    Args:
        config: Configuration dictionary
        model_client: Model client for the agent

    Returns:
        AutoGen AssistantAgent configured as a planner
    """
    agent_config = config.get("agents", {}).get("planner", {})

    # Load system prompt from config or use default
    default_system_message = """You are a Research Planner. Your job is to break down research queries into clear, actionable steps.

When given a research query, you should:
1. Identify the key concepts and topics to investigate
2. Determine what types of sources would be most valuable (academic papers, web articles, etc.)
3. Suggest specific search queries for the Researcher
4. Outline how the findings should be synthesized

Provide your plan in a structured format with numbered steps.
Be specific about what information to gather and why it's relevant.

After creating the plan, say "PLAN COMPLETE"."""

    # Use custom prompt from config if available, otherwise use default
    custom_prompt = agent_config.get("system_prompt", "").strip()
    if custom_prompt:
        system_message = custom_prompt
    else:
        system_message = default_system_message

    planner = AssistantAgent(
        name="Planner",
        model_client=model_client,
        description="Breaks down research queries into actionable steps",
        system_message=system_message,
    )

    return planner


def create_researcher_agent(config: Dict[str, Any], model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    """
    Create a Researcher Agent using AutoGen.

    The researcher has access to web search and paper search tools.
    It gathers evidence based on the planner's guidance.

    Args:
        config: Configuration dictionary
        model_client: Model client for the agent

    Returns:
        AutoGen AssistantAgent configured as a researcher with tool access
    """
    agent_config = config.get("agents", {}).get("researcher", {})

    # Load system prompt from config or use default
    default_system_message = """You are a Research Assistant. Your job is to gather high-quality information from academic papers and web sources.

CRITICAL INSTRUCTIONS FOR TOOL USAGE:
You have access to two tools: web_search and paper_search. These tools are available through the function calling API.

IMPORTANT - HOW TO USE TOOLS:
- DO NOT write function calls in text format like <function=web_search>...</function> or <function=paper_search>...</function>
- DO NOT write function calls as text in your response
- The tools are automatically available - when you need to search, the system will automatically call them for you
- Simply state what you want to search for in natural language, and the tools will be invoked automatically
- For example, say "I need to search for information about AI transparency" rather than writing a function call

You have access to tools for web search and paper search. When conducting research:
1. Use both web search and paper_search for comprehensive coverage
2. Look for recent, high-quality sources
3. Extract key findings, quotes, and data
4. Note all source URLs and citations
5. Gather evidence that directly addresses the research query

After collecting sufficient evidence, say "RESEARCH COMPLETE"."""

    # Use custom prompt from config if available, otherwise use default
    custom_prompt = agent_config.get("system_prompt", "").strip()
    if custom_prompt:
        system_message = custom_prompt
    else:
        system_message = default_system_message

    # Wrap tools in FunctionTool with more detailed descriptions
    web_search_tool = FunctionTool(
        web_search,
        description="""Search the web for articles, blog posts, and general information.

Parameters:
- query (str, required): The search query string
- provider (str, optional): Search provider, either "tavily" or "brave". Default is "tavily"
- max_results (int, optional): Maximum number of results to return. Default is 5

Returns: A formatted string with search results including titles, URLs, and snippets.

Example usage: web_search(query="AI transparency", provider="tavily", max_results=10)"""
    )

    paper_search_tool = FunctionTool(
        paper_search,
        description="""Search academic papers on Semantic Scholar.

Parameters:
- query (str, required): The search query string
- max_results (int, optional): Maximum number of results to return. Default is 10
- year_from (int, optional): Filter papers published from this year onwards

Returns: A formatted string with paper results including authors, abstracts, citation counts, and URLs.

Example usage: paper_search(query="explainable AI", max_results=10, year_from=2020)"""
    )

    # Create the researcher with tool access
    researcher = AssistantAgent(
        name="Researcher",
        model_client=model_client,
        tools=[web_search_tool, paper_search_tool],
        description="Gathers evidence from web and academic sources using search tools",
        system_message=system_message,
    )

    # Log tool registration for debugging
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Researcher agent created with {len(researcher.tools) if hasattr(researcher, 'tools') else 0} tools")
    if hasattr(researcher, 'tools'):
        for tool in researcher.tools:
            tool_name = tool.name if hasattr(tool, 'name') else type(tool).__name__
            logger.info(f"  - Tool: {tool_name}")

    return researcher


def create_writer_agent(config: Dict[str, Any], model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    """
    Create a Writer Agent using AutoGen.

    The writer synthesizes research findings into coherent responses with proper citations.

    Args:
        config: Configuration dictionary
        model_client: Model client for the agent

    Returns:
        AutoGen AssistantAgent configured as a writer
    """
    agent_config = config.get("agents", {}).get("writer", {})

    # Load system prompt from config or use default
    default_system_message = """You are a Research Writer. Your job is to synthesize research findings into clear, well-organized responses.

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

    # Use custom prompt from config if available, otherwise use default
    custom_prompt = agent_config.get("system_prompt", "").strip()
    if custom_prompt:
        system_message = custom_prompt
    else:
        system_message = default_system_message

    writer = AssistantAgent(
        name="Writer",
        model_client=model_client,
        description="Synthesizes research findings into coherent, well-cited responses",
        system_message=system_message,
    )

    return writer


def create_critic_agent(config: Dict[str, Any], model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    """
    Create a Critic Agent using AutoGen.

    The critic evaluates the quality of the research and writing,
    providing feedback for improvement.

    IMPORTANT: The Critic does NOT have access to search tools.
    Only the Researcher has access to web_search and paper_search tools.

    Args:
        config: Configuration dictionary
        model_client: Model client for the agent

    Returns:
        AutoGen AssistantAgent configured as a critic
    """
    agent_config = config.get("agents", {}).get("critic", {})

    # Load system prompt from config or use default
    default_system_message = """You are a Research Critic. Your job is to evaluate the quality and accuracy of research outputs.

IMPORTANT: You do NOT have access to search tools. Only evaluate based on the content provided by the Writer.

Evaluate the research and writing on these criteria:
1. **Relevance**: Does it answer the original query?
2. **Evidence Quality**: Are sources credible and well-cited?
3. **Completeness**: Are all aspects of the query addressed?
4. **Accuracy**: Are there any factual errors or contradictions?
5. **Clarity**: Is the writing clear and well-organized?

Provide constructive but thorough feedback. End your evaluation with either "APPROVED - RESEARCH COMPLETE" if approved, or "NEEDS REVISION" if improvements are needed. You may also use "TERMINATE" to signal completion."""

    # Use custom prompt from config if available, otherwise use default
    custom_prompt = agent_config.get("system_prompt", "").strip()
    if custom_prompt:
        system_message = custom_prompt
    else:
        system_message = default_system_message

    critic = AssistantAgent(
        name="Critic",
        model_client=model_client,
        description="Evaluates research quality and provides feedback",
        system_message=system_message,
        tools=[],  # Explicitly no tools for Critic
    )

    return critic


def create_research_team(config: Dict[str, Any]) -> RoundRobinGroupChat:
    """
    Create the research team as a RoundRobinGroupChat.

    Args:
        config: Configuration dictionary

    Returns:
        RoundRobinGroupChat with all agents configured
    """
    # Create model client (shared by all agents)
    model_client = create_model_client(config)

    # Create all agents
    planner = create_planner_agent(config, model_client)
    researcher = create_researcher_agent(config, model_client)
    writer = create_writer_agent(config, model_client)
    critic = create_critic_agent(config, model_client)

    # Create termination condition - accept multiple termination signals
    # The critic can use "TERMINATE", "APPROVED - RESEARCH COMPLETE", or "NEEDS REVISION" (for revision loops)
    # Note: "NEEDS REVISION" is not a termination signal, it triggers revision loops
    termination = create_multi_string_termination(["TERMINATE", "APPROVED - RESEARCH COMPLETE"])

    # Create team with round-robin ordering
    team = RoundRobinGroupChat(
        participants=[planner, researcher, writer, critic],
        termination_condition=termination,
    )

    return team
