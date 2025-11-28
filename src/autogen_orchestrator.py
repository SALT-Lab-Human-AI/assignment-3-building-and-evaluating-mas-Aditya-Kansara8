"""
AutoGen-Based Orchestrator

This orchestrator uses AutoGen's RoundRobinGroupChat to coordinate multiple agents
in a research workflow.

Workflow:
1. Planner: Breaks down the query into research steps
2. Researcher: Gathers evidence using web and paper search tools
3. Writer: Synthesizes findings into a coherent response
4. Critic: Evaluates quality and provides feedback
5. Revision Loop: If critic says "NEEDS REVISION", go back to Writer

This orchestrator implements the full workflow with revision loops and comprehensive error handling.
"""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional
from enum import Enum
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage

from src.agents.autogen_agents import create_research_team


class WorkflowStage(Enum):
    """Enumeration of workflow stages."""
    INITIALIZED = "initialized"
    PLANNING = "planning"
    RESEARCHING = "researching"
    WRITING = "writing"
    CRITIQUING = "critiquing"
    REVISING = "revising"
    COMPLETED = "completed"
    ERROR = "error"


class AutoGenOrchestrator:
    """
    Orchestrates multi-agent research using AutoGen's RoundRobinGroupChat.

    This orchestrator manages a team of specialized agents that work together
    to answer research queries. It uses AutoGen's built-in conversation
    management and tool execution capabilities.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AutoGen orchestrator.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.logger = logging.getLogger("autogen_orchestrator")

        # Get configuration values
        system_config = config.get("system", {})
        self.max_iterations = system_config.get("max_iterations", 10)
        self.timeout_seconds = system_config.get("timeout_seconds", 300)

        # Don't create team here - create it fresh for each query to avoid "already running" errors
        self.logger.info("AutoGen orchestrator initialized (team will be created per query)")

        # Workflow state tracking
        self.workflow_trace: List[Dict[str, Any]] = []
        self.current_stage = WorkflowStage.INITIALIZED
        self.revision_count = 0
        self.max_revisions = 3  # Maximum number of revision cycles

    def process_query(self, query: str, max_rounds: int = 20) -> Dict[str, Any]:
        """
        Process a research query through the multi-agent system.

        Implements the full workflow: plan → research → write → critique → revise (if needed)

        Args:
            query: The research question to answer
            max_rounds: Maximum number of conversation rounds per iteration

        Returns:
            Dictionary containing:
            - query: Original query
            - response: Final synthesized response
            - conversation_history: Full conversation between agents
            - workflow_stages: List of workflow stages traversed
            - metadata: Additional information about the process
        """
        self.logger.info(f"Processing query: {query}")
        self.current_stage = WorkflowStage.INITIALIZED
        self.revision_count = 0
        self.workflow_trace = []

        start_time = time.time()

        try:
            # Run the async query processing with timeout
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if loop.is_running():
                # If we're already in an async context, create a new loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self._process_query_with_revisions(query, max_rounds, start_time)
                    )
                    result = future.result(timeout=self.timeout_seconds)
            else:
                result = loop.run_until_complete(
                    asyncio.wait_for(
                        self._process_query_with_revisions(query, max_rounds, start_time),
                        timeout=self.timeout_seconds
                    )
                )

            self.current_stage = WorkflowStage.COMPLETED
            self.logger.info(f"Query processing complete in {time.time() - start_time:.2f}s")

            # Validate result
            if not result.get("conversation_history"):
                self.logger.warning("No conversation history in result!")
                result["conversation_history"] = []

            if not result.get("response") or result.get("response") == query:
                self.logger.warning("Response is empty or same as query - workflow may not have executed")

            return result

        except asyncio.TimeoutError:
            self.logger.error(f"Query processing timed out after {self.timeout_seconds}s")
            self.current_stage = WorkflowStage.ERROR
            return {
                "query": query,
                "error": "timeout",
                "response": f"Query processing timed out after {self.timeout_seconds} seconds. Please try a simpler query or increase the timeout.",
                "conversation_history": self.workflow_trace,
                "workflow_stages": [stage.value for stage in [WorkflowStage.INITIALIZED, WorkflowStage.ERROR]],
                "metadata": {
                    "error": True,
                    "timeout": True,
                    "revision_count": self.revision_count
                }
            }
        except Exception as e:
            self.logger.error(f"Error processing query: {e}", exc_info=True)
            self.current_stage = WorkflowStage.ERROR
            return {
                "query": query,
                "error": str(e),
                "response": f"An error occurred while processing your query: {str(e)}",
                "conversation_history": self.workflow_trace,
                "workflow_stages": [stage.value for stage in [WorkflowStage.INITIALIZED, WorkflowStage.ERROR]],
                "metadata": {
                    "error": True,
                    "error_type": type(e).__name__,
                    "revision_count": self.revision_count
                }
            }

    def _get_or_create_team(self):
        """
        Get or create a fresh team instance.
        Creates a new team for each query to avoid "already running" errors.

        Returns:
            RoundRobinGroupChat team instance
        """
        try:
            # Always create a fresh team to avoid state issues
            self.logger.info("Creating fresh research team for this query...")
            team = create_research_team(self.config)
            self.logger.info("Research team created successfully")
            return team
        except Exception as e:
            self.logger.error(f"Failed to create research team: {e}", exc_info=True)
            raise

    async def _process_query_with_revisions(
        self,
        query: str,
        max_rounds: int,
        start_time: float
    ) -> Dict[str, Any]:
        """
        Process query with revision loop support.

        Implements: plan → research → write → critique → (revise if needed)

        Args:
            query: The research question to answer
            max_rounds: Maximum number of conversation rounds per iteration
            start_time: Start time for timeout tracking

        Returns:
            Dictionary containing results
        """
        all_messages = []
        iteration = 0
        task_message = ""  # Initialize to track the task message

        while iteration < self.max_iterations:
            iteration += 1
            self.logger.info(f"Starting iteration {iteration}/{self.max_iterations}")

            # Create a fresh team for each iteration to avoid "already running" errors
            team = self._get_or_create_team()

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.timeout_seconds:
                self.logger.warning(f"Timeout approaching: {elapsed:.2f}s / {self.timeout_seconds}s")

            # Create task message (different for first iteration vs revisions)
            if iteration == 1:
                task_message = f"""Research Query: {query}

Please work together to answer this query comprehensively:

1. Planner: Create a research plan and say "PLAN COMPLETE"

2. Researcher: Gather evidence from web and academic sources.
   IMPORTANT: Use the web_search() and paper_search() tools by calling them through the tool API,
   NOT by writing them in text format. The tools are already registered and will be called automatically.
   After gathering evidence, say "RESEARCH COMPLETE"

3. Writer: Synthesize findings into a well-cited response and say "DRAFT COMPLETE"

4. Critic: Evaluate the quality and provide feedback. Say "APPROVED - RESEARCH COMPLETE" if approved,
   or "NEEDS REVISION" if improvements are needed."""
            else:
                # Revision iteration
                self.current_stage = WorkflowStage.REVISING
                self.revision_count += 1
                self.logger.info(f"Starting revision {self.revision_count}")

                if self.revision_count > self.max_revisions:
                    self.logger.warning(f"Maximum revisions ({self.max_revisions}) reached")
                    break

                # Get previous critique for context
                previous_critique = self._get_last_critique(all_messages)
                task_message = f"""Revision Request #{self.revision_count}

The previous draft was evaluated and needs revision. Please revise based on this feedback:

{previous_critique}

Writer: Please revise the draft addressing the feedback above and say "DRAFT COMPLETE"
Critic: Re-evaluate the revised draft. Say "APPROVED - RESEARCH COMPLETE" if approved, or "NEEDS REVISION" if more work is needed."""

            try:
                self.logger.info(f"Running team with task message (length: {len(task_message)})")

                # Run the team for this iteration
                # Use the fresh team instance created for this query
                result = await team.run(task=task_message)

                self.logger.info(f"Team.run() completed. Checking result...")

                # Extract messages from this iteration
                iteration_messages = []

                # Check what we got back
                if not hasattr(result, 'messages'):
                    self.logger.error("Result does not have 'messages' attribute")
                    self.logger.error(f"Result type: {type(result)}, Result: {result}")
                    raise ValueError("team.run() result does not contain messages")

                messages_list = result.messages if hasattr(result, 'messages') else []
                self.logger.info(f"Found {len(messages_list)} messages in result")

                # Log agent participation
                agent_participation = {}

                for i, message in enumerate(messages_list):
                    try:
                        # Try different ways to get source and content
                        source = None
                        content = None

                        if hasattr(message, 'source'):
                            source = str(message.source)
                        elif hasattr(message, 'name'):
                            source = str(message.name)
                        else:
                            # Try to get from agent attribute
                            if hasattr(message, 'agent'):
                                source = str(message.agent) if hasattr(message.agent, 'name') else str(message.agent)
                            else:
                                source = "Unknown"

                        if hasattr(message, 'content'):
                            content = str(message.content)
                        elif hasattr(message, 'text'):
                            content = str(message.text)
                        else:
                            content = str(message)

                        msg_dict = {
                            "source": source,
                            "content": content,
                            "iteration": iteration,
                            "timestamp": time.time(),
                        }
                        iteration_messages.append(msg_dict)
                        all_messages.append(msg_dict)

                        # Track which agents participated
                        if source not in agent_participation:
                            agent_participation[source] = 0
                        agent_participation[source] += 1

                        self.logger.info(f"Message {i+1}: source={source}, content_length={len(content)}")

                        # Track workflow stage based on agent
                        if "Planner" in source or source == "Planner":
                            self.current_stage = WorkflowStage.PLANNING
                        elif "Researcher" in source or source == "Researcher":
                            self.current_stage = WorkflowStage.RESEARCHING
                        elif "Writer" in source or source == "Writer":
                            self.current_stage = WorkflowStage.WRITING
                        elif "Critic" in source or source == "Critic":
                            self.current_stage = WorkflowStage.CRITIQUING

                    except Exception as msg_error:
                        self.logger.error(f"Error extracting message {i}: {msg_error}", exc_info=True)
                        # Continue with next message
                        continue

                # Log agent participation summary
                if agent_participation:
                    self.logger.info(f"Agent participation: {agent_participation}")

                if not iteration_messages:
                    self.logger.warning("No messages extracted from team.run() result!")
                    # If no messages, we might need to check the result structure differently
                    self.logger.info(f"Result object: {result}")
                    self.logger.info(f"Result attributes: {dir(result)}")

                    # Try to create a fallback message
                    if hasattr(result, 'content'):
                        iteration_messages.append({
                            "source": "System",
                            "content": str(result.content),
                            "iteration": iteration,
                            "timestamp": time.time(),
                        })
                    else:
                        # Last resort: use the task message as a message
                        self.logger.warning("Using task message as fallback - this indicates the team didn't execute")
                        iteration_messages.append({
                            "source": "System",
                            "content": task_message,
                            "iteration": iteration,
                            "timestamp": time.time(),
                        })
                        all_messages.extend(iteration_messages)
                        # Break out - something is wrong
                        break

                # Check if we should continue (revision needed) or stop (approved)
                should_continue = self._check_if_revision_needed(iteration_messages)

                # Validate that we have messages from expected agents (for first iteration)
                if iteration == 1:
                    agent_sources = [msg.get("source") for msg in iteration_messages]
                    expected_agents = ["Planner", "Researcher", "Writer", "Critic"]
                    missing_agents = [agent for agent in expected_agents if agent not in agent_sources]

                    if missing_agents:
                        self.logger.warning(f"Missing agents in first iteration: {missing_agents}")
                        self.logger.warning("Workflow may not have executed properly. Continuing anyway...")

                if not should_continue:
                    self.logger.info("Critic approved the response. Workflow complete.")
                    break

            except Exception as e:
                error_str = str(e)
                self.logger.error(f"Error in iteration {iteration}: {e}", exc_info=True)

                # Check if it's a tool calling error
                if "tool_use_failed" in error_str or "BadRequestError" in error_str or "failed_generation" in error_str:
                    self.logger.error("=" * 80)
                    self.logger.error("TOOL CALLING ERROR DETECTED")
                    self.logger.error("=" * 80)
                    self.logger.error("The model is generating function calls in text format instead of using the function calling API.")
                    self.logger.error("")
                    self.logger.error("This is a known compatibility issue with some models.")
                    self.logger.error("")
                    self.logger.error("SOLUTIONS:")
                    self.logger.error("1. If using OpenAI, try switching to a different model in config.yaml:")
                    self.logger.error("   - gpt-4o (recommended)")
                    self.logger.error("   - gpt-4-turbo")
                    self.logger.error("   - gpt-3.5-turbo")
                    self.logger.error("")
                    self.logger.error("2. If using Groq, try switching to a different model in config.yaml:")
                    self.logger.error("   - llama-3.1-70b-versatile (recommended)")
                    self.logger.error("   - llama-3.1-8b-instant")
                    self.logger.error("   - mixtral-8x7b-32768")
                    self.logger.error("")
                    self.logger.error("3. Update config.yaml models.default.name to one of the above models")
                    self.logger.error("")
                    self.logger.error("3. The current model may not properly support function calling")
                    self.logger.error("=" * 80)

                    # If first iteration, this is critical
                    if iteration == 1:
                        self.logger.error("First iteration failed due to tool error. Workflow cannot proceed.")
                        # You might want to break here or try a different approach

                # If first iteration fails, we should restart the full workflow
                # Don't continue to revision if we never completed the initial workflow
                if iteration == 1:
                    self.logger.warning("First iteration failed. This indicates the workflow didn't execute properly.")
                    self.logger.warning("The error may be due to tool calling issues or agent configuration problems.")

                    # If we have no messages from agents, something is seriously wrong
                    if not any(msg.get("source") in ["Planner", "Researcher", "Writer", "Critic"] for msg in all_messages):
                        self.logger.error("No agent messages found. Workflow cannot continue.")
                        # Break out and return error
                        break

                # Continue to next iteration if possible
                if iteration >= self.max_iterations:
                    raise
                continue

        # Extract final response
        final_response = self._extract_final_response(all_messages)

        # If we still don't have a good response, try to get it from the last non-system message
        if not final_response or final_response == task_message:
            self.logger.warning("Final response extraction failed, trying alternative method")
            for msg in reversed(all_messages):
                if msg.get("source") not in ["System", "user", "User"]:
                    final_response = msg.get("content", "")
                    if final_response and final_response != task_message:
                        break

        return self._extract_results(query, all_messages, final_response, iteration)

    def _check_if_revision_needed(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Check if revision is needed based on critic's feedback.

        Args:
            messages: Messages from the current iteration

        Returns:
            True if revision is needed, False if approved
        """
        for msg in reversed(messages):
            if msg.get("source") == "Critic":
                content = msg.get("content", "").upper()
                if "NEEDS REVISION" in content:
                    return True
                elif "APPROVED" in content or "TERMINATE" in content:
                    return False
        # Default: if no clear signal, assume approved
        return False

    def _get_last_critique(self, messages: List[Dict[str, Any]]) -> str:
        """Extract the last critique from messages."""
        for msg in reversed(messages):
            if msg.get("source") == "Critic":
                return msg.get("content", "No specific feedback provided.")
        return "No previous critique found."

    def _extract_final_response(self, messages: List[Dict[str, Any]]) -> str:
        """Extract the final response from messages - should be Writer's approved draft, NOT Critic's evaluation."""
        # Strategy: Find the last Writer draft that was approved
        # Look for the pattern: Writer says "DRAFT COMPLETE" -> Critic says "APPROVED"

        # First, find the last "APPROVED" message from Critic to know which draft was approved
        last_approved_index = -1
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("source") == "Critic":
                content = msg.get("content", "").upper()
                if "APPROVED" in content or "TERMINATE" in content:
                    last_approved_index = i
                    break

        # Now find the Writer draft that comes before this approval
        # This should be the final approved draft
        if last_approved_index > 0:
            # Look backwards from the approval to find the Writer's draft
            for i in range(last_approved_index - 1, -1, -1):
                msg = messages[i]
                if msg.get("source") == "Writer":
                    content = msg.get("content", "")
                    # This is the draft that was approved
                    if content and len(content) > 50:  # Ensure it's substantial
                        return content

        # Fallback: If no approval found, get the last Writer message with "DRAFT COMPLETE"
        for msg in reversed(messages):
            if msg.get("source") == "Writer":
                content = msg.get("content", "")
                if "DRAFT COMPLETE" in content.upper():
                    # Return the full draft content
                    return content

        # Fallback: Get the last substantial Writer message
        for msg in reversed(messages):
            if msg.get("source") == "Writer":
                content = msg.get("content", "")
                # Skip if it's just a termination signal or very short
                if content and len(content) > 50:
                    return content

        # Last resort: get last non-system, non-critic message (to avoid approval messages)
        for msg in reversed(messages):
            source = msg.get("source", "")
            if source not in ["System", "user", "User", "Critic"]:
                content = msg.get("content", "")
                if content and len(content) > 50:
                    return content

        # Final fallback: use last message (but log a warning)
        if messages:
            self.logger.warning("Could not find Writer draft, using last message as fallback")
            return messages[-1].get("content", "")
        return ""

    def _extract_results(
        self,
        query: str,
        messages: List[Dict[str, Any]],
        final_response: str = "",
        iterations: int = 1
    ) -> Dict[str, Any]:
        """
        Extract structured results from the conversation history.

        Args:
            query: Original query
            messages: List of conversation messages
            final_response: Final response from the team
            iterations: Number of iterations completed

        Returns:
            Structured result dictionary
        """
        # Extract components from conversation
        research_findings = []
        plan = ""
        critiques = []
        drafts = []

        for msg in messages:
            source = msg.get("source", "")
            content = msg.get("content", "")
            iteration = msg.get("iteration", 1)

            if source == "Planner" and not plan:
                plan = content

            elif source == "Researcher":
                research_findings.append({
                    "content": content,
                    "iteration": iteration
                })

            elif source == "Writer":
                drafts.append({
                    "content": content,
                    "iteration": iteration
                })

            elif source == "Critic":
                critiques.append({
                    "content": content,
                    "iteration": iteration
                })

        # Get latest critique
        latest_critique = critiques[-1]["content"] if critiques else ""

        # Count sources mentioned in research
        num_sources = 0
        for finding in research_findings:
            content = finding.get("content", "")
            # Rough count of sources based on numbered results
            num_sources += content.count("\n1.") + content.count("\n2.") + content.count("\n3.")

        # Clean up final response - remove termination signals
        if final_response:
            # Remove various termination signals
            termination_signals = [
                "TERMINATE",
                "APPROVED - RESEARCH COMPLETE",
                "APPROVED-RESEARCH COMPLETE",
                "NEEDS REVISION",
                "DRAFT COMPLETE"
            ]
            for signal in termination_signals:
                final_response = final_response.replace(signal, "").strip()

        # Track workflow stages
        stages_traversed = []
        current_stage = None
        for msg in messages:
            source = msg.get("source", "")
            if source == "Planner" and current_stage != WorkflowStage.PLANNING:
                stages_traversed.append(WorkflowStage.PLANNING.value)
                current_stage = WorkflowStage.PLANNING
            elif source == "Researcher" and current_stage != WorkflowStage.RESEARCHING:
                stages_traversed.append(WorkflowStage.RESEARCHING.value)
                current_stage = WorkflowStage.RESEARCHING
            elif source == "Writer" and current_stage != WorkflowStage.WRITING:
                stages_traversed.append(WorkflowStage.WRITING.value)
                current_stage = WorkflowStage.WRITING
            elif source == "Critic" and current_stage != WorkflowStage.CRITIQUING:
                stages_traversed.append(WorkflowStage.CRITIQUING.value)
                current_stage = WorkflowStage.CRITIQUING

        if self.revision_count > 0:
            stages_traversed.append(WorkflowStage.REVISING.value)

        return {
            "query": query,
            "response": final_response,
            "conversation_history": messages,
            "workflow_stages": stages_traversed,
            "metadata": {
                "num_messages": len(messages),
                "num_sources": max(num_sources, 1),  # At least 1
                "iterations": iterations,
                "revision_count": self.revision_count,
                "plan": plan,
                "research_findings": [f["content"] for f in research_findings],
                "critiques": [c["content"] for c in critiques],
                "latest_critique": latest_critique,
                "drafts": [d["content"] for d in drafts],
                "agents_involved": list(set([msg.get("source", "") for msg in messages])),
                "current_stage": self.current_stage.value,
            }
        }

    def get_agent_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all agents.

        Returns:
            Dictionary mapping agent names to their descriptions
        """
        return {
            "Planner": "Breaks down research queries into actionable steps",
            "Researcher": "Gathers evidence from web and academic sources",
            "Writer": "Synthesizes findings into coherent responses",
            "Critic": "Evaluates quality and provides feedback",
        }

    def visualize_workflow(self) -> str:
        """
        Generate a text visualization of the workflow.

        Returns:
            String representation of the workflow
        """
        workflow = """
AutoGen Research Workflow (with Revision Loop):

1. User Query
   ↓
2. Planner
   - Analyzes query
   - Creates research plan
   - Identifies key topics
   - Says "PLAN COMPLETE"
   ↓
3. Researcher (with tools)
   - Uses web_search() tool
   - Uses paper_search() tool
   - Gathers evidence
   - Collects citations
   - Says "RESEARCH COMPLETE"
   ↓
4. Writer
   - Synthesizes findings
   - Creates structured response
   - Adds citations
   - Says "DRAFT COMPLETE"
   ↓
5. Critic
   - Evaluates quality
   - Checks completeness
   - Provides feedback
   - Decision:
     * "APPROVED - RESEARCH COMPLETE" → Final Response
     * "NEEDS REVISION" → Revision Loop
   ↓
6. Revision Loop (if needed)
   - Writer revises based on feedback
   - Critic re-evaluates
   - Repeat until approved or max revisions reached
   ↓
7. Final Response
        """
        return workflow

    def get_workflow_status(self) -> Dict[str, Any]:
        """
        Get current workflow status.

        Returns:
            Dictionary with current workflow state
        """
        return {
            "current_stage": self.current_stage.value,
            "revision_count": self.revision_count,
            "max_revisions": self.max_revisions,
            "max_iterations": self.max_iterations,
            "timeout_seconds": self.timeout_seconds,
            "trace_length": len(self.workflow_trace)
        }


def demonstrate_usage():
    """
    Demonstrate how to use the AutoGen orchestrator.

    This function shows a simple example of using the orchestrator.
    """
    import yaml
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create orchestrator
    orchestrator = AutoGenOrchestrator(config)

    # Print workflow visualization
    print(orchestrator.visualize_workflow())

    # Example query
    query = "What are the latest trends in human-computer interaction research?"

    print(f"\nProcessing query: {query}\n")
    print("=" * 70)

    # Process query
    result = orchestrator.process_query(query)

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nQuery: {result['query']}")
    print(f"\nResponse:\n{result['response']}")
    print(f"\nMetadata:")
    print(f"  - Messages exchanged: {result['metadata']['num_messages']}")
    print(f"  - Sources gathered: {result['metadata']['num_sources']}")
    print(f"  - Agents involved: {', '.join(result['metadata']['agents_involved'])}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    demonstrate_usage()
