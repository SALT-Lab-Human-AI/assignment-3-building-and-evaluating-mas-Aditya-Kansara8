"""
Streamlit Web Interface
Web UI for the multi-agent research system.

Run with: streamlit run src/ui/streamlit_app.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import asyncio
import yaml
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

from src.autogen_orchestrator import AutoGenOrchestrator

# Load environment variables
load_dotenv()


def load_config():
    """Load configuration file."""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'history' not in st.session_state:
        st.session_state.history = []

    if 'orchestrator' not in st.session_state:
        config = load_config()
        # Initialize AutoGen orchestrator
        try:
            st.session_state.orchestrator = AutoGenOrchestrator(config)
        except Exception as e:
            st.error(f"Failed to initialize orchestrator: {e}")
            st.session_state.orchestrator = None

    if 'show_traces' not in st.session_state:
        st.session_state.show_traces = False

    if 'show_safety_log' not in st.session_state:
        st.session_state.show_safety_log = False


async def process_query(query: str) -> Dict[str, Any]:
    """
    Process a query through the orchestrator.

    Args:
        query: Research query to process

    Returns:
        Result dictionary with response, citations, and metadata
    """
    orchestrator = st.session_state.orchestrator

    if orchestrator is None:
        return {
            "query": query,
            "error": "Orchestrator not initialized",
            "response": "Error: System not properly initialized. Please check your configuration.",
            "citations": [],
            "metadata": {}
        }

    try:
        # Process query through AutoGen orchestrator
        result = orchestrator.process_query(query)

        # Check for errors
        if "error" in result:
            # Even on error, include workflow_stages and conversation_history if available
            error_result = result.copy()
            if "workflow_stages" not in error_result:
                error_result["workflow_stages"] = []
            if "conversation_history" not in error_result:
                error_result["conversation_history"] = []
            return error_result

        # Extract citations and sources from conversation history
        citations, sources = extract_citations_and_sources(result)

        # Extract agent traces for display
        agent_traces = extract_agent_traces(result)

        # Extract safety events
        safety_events = extract_safety_events(result)

        # Format metadata
        metadata = result.get("metadata", {})
        metadata["agent_traces"] = agent_traces
        metadata["citations"] = citations
        metadata["sources"] = sources
        metadata["safety_events"] = safety_events
        metadata["critique_score"] = calculate_quality_score(result)

        # Build return dictionary with all necessary fields
        return_dict = {
            "query": query,
            "response": result.get("response", ""),
            "citations": citations,
            "workflow_stages": result.get("workflow_stages", []),  # Include workflow stages
            "conversation_history": result.get("conversation_history", []),  # Include conversation history
            "metadata": metadata
        }

        # Debug: Log what we're returning (only in development)
        if st.session_state.get("debug_mode", False):
            st.write(f"Debug - Workflow stages: {return_dict.get('workflow_stages')}")
            st.write(f"Debug - Conversation history length: {len(return_dict.get('conversation_history', []))}")

        return return_dict

    except Exception as e:
        return {
            "query": query,
            "error": str(e),
            "response": f"An error occurred: {str(e)}",
            "citations": [],
            "workflow_stages": [],  # Include empty workflow stages
            "conversation_history": [],  # Include empty conversation history
            "metadata": {"error": True}
        }


def extract_citations_and_sources(result: Dict[str, Any]) -> tuple:
    """
    Extract citations and sources from research result.

    Returns:
        Tuple of (citations: list, sources: list)
    """
    citations = []
    sources = []
    seen_urls = set()

    import re

    # Look through conversation history
    for msg in result.get("conversation_history", []):
        content = msg.get("content", "")
        source_agent = msg.get("source", "")

        # Find URLs in content
        urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', content)

        for url in urls:
            if url not in seen_urls:
                citations.append(url)
                seen_urls.add(url)

        # Extract source information from Researcher messages
        if source_agent == "Researcher":
            # Look for paper titles and URLs
            paper_pattern = r'(\d+)\.\s*([^\n]+)\n.*?URL:\s*(https?://[^\s]+)'
            papers = re.findall(paper_pattern, content, re.MULTILINE)

            for paper_num, title, url in papers:
                if url not in seen_urls:
                    sources.append({
                        "type": "paper",
                        "title": title.strip(),
                        "url": url,
                        "agent": source_agent
                    })
                    seen_urls.add(url)

            # Look for web source titles and URLs
            web_pattern = r'(\d+)\.\s*([^\n]+)\n\s*URL:\s*(https?://[^\s]+)'
            web_sources = re.findall(web_pattern, content, re.MULTILINE)

            for web_num, title, url in web_sources:
                if url not in seen_urls:
                    sources.append({
                        "type": "web",
                        "title": title.strip(),
                        "url": url,
                        "agent": source_agent
                    })
                    seen_urls.add(url)

    # Also check metadata for sources
    metadata = result.get("metadata", {})
    research_findings = metadata.get("research_findings", [])
    for finding in research_findings:
        if isinstance(finding, dict):
            if finding.get("url") and finding.get("url") not in seen_urls:
                sources.append({
                    "type": finding.get("type", "unknown"),
                    "title": finding.get("title", "Unknown"),
                    "url": finding.get("url"),
                    "agent": "Researcher"
                })
                seen_urls.add(finding.get("url"))

    return citations[:10], sources[:15]  # Limit results


def extract_safety_events(result: Dict[str, Any]) -> list:
    """Extract safety events from result."""
    safety_events = []

    # Check metadata for safety events
    metadata = result.get("metadata", {})
    if metadata.get("safety_events"):
        safety_events.extend(metadata.get("safety_events", []))

    # Check conversation history for safety-related messages
    for msg in result.get("conversation_history", []):
        content = msg.get("content", "").lower()
        if "safety" in content or "blocked" in content or "violation" in content:
            safety_events.append({
                "type": "detected",
                "reason": "Safety-related content detected in conversation",
                "agent": msg.get("source", "Unknown")
            })

    return safety_events


def extract_agent_traces(result: Dict[str, Any]) -> Dict[str, list]:
    """Extract agent execution traces from conversation history."""
    traces = {}

    for msg in result.get("conversation_history", []):
        agent = msg.get("source", "Unknown")
        content = msg.get("content", "")[:200]  # First 200 chars

        if agent not in traces:
            traces[agent] = []

        traces[agent].append({
            "action_type": "message",
            "details": content
        })

    return traces


def calculate_quality_score(result: Dict[str, Any]) -> float:
    """Calculate a quality score based on various factors."""
    score = 5.0  # Base score

    metadata = result.get("metadata", {})

    # Add points for sources
    num_sources = metadata.get("num_sources", 0)
    score += min(num_sources * 0.5, 2.0)

    # Add points for critique
    if metadata.get("critique"):
        score += 1.0

    # Add points for conversation length (indicates thorough discussion)
    num_messages = metadata.get("num_messages", 0)
    score += min(num_messages * 0.1, 2.0)

    return min(score, 10.0)  # Cap at 10


def display_response(result: Dict[str, Any]):
    """
    Display query response with full details.
    """
    # Check for errors
    if "error" in result:
        st.error(f"❌ Error: {result['error']}")
        if result.get("metadata", {}).get("error_type"):
            st.info(f"Error Type: {result['metadata']['error_type']}")
        return

    # Display response first - show complete final draft (Writer's draft, NOT Critic's evaluation)
    st.markdown("### 📝 Response")
    response = result.get("response", "")

    # Verify that the response is actually from Writer, not Critic
    # If it looks like a critique (contains evaluation keywords), find the Writer's draft instead
    response_lower = response.lower() if response else ""
    is_critique = any(keyword in response_lower for keyword in [
        "thoroughly evaluated", "based on the following criteria", "relevance:",
        "evidence quality:", "completeness:", "accuracy:", "clarity:",
        "the response is", "the synthesis", "evaluated based"
    ])

    # If response looks like a critique, or is empty, find the Writer's draft
    if is_critique or not response or len(response) < 50:
        conversation_history = result.get("conversation_history", [])

        # Strategy: Find the Writer draft that was approved (comes before "APPROVED" from Critic)
        # First find the last "APPROVED" message
        last_approved_index = -1
        for i in range(len(conversation_history) - 1, -1, -1):
            msg = conversation_history[i]
            if msg.get("source") == "Critic":
                content = msg.get("content", "").upper()
                if "APPROVED" in content or "TERMINATE" in content:
                    last_approved_index = i
                    break

        # Find the Writer draft before the approval
        if last_approved_index > 0:
            for i in range(last_approved_index - 1, -1, -1):
                msg = conversation_history[i]
                if msg.get("source") == "Writer":
                    content = msg.get("content", "")
                    if content and len(content) > 50:
                        response = content
                        break

        # Fallback: Look for Writer message with "DRAFT COMPLETE"
        if not response or is_critique:
            for msg in reversed(conversation_history):
                if msg.get("source") == "Writer":
                    content = msg.get("content", "")
                    if "DRAFT COMPLETE" in content.upper() and len(content) > 50:
                        response = content
                        break

        # Last fallback: Get any substantial Writer message
        if not response or is_critique:
            for msg in reversed(conversation_history):
                if msg.get("source") == "Writer":
                    content = msg.get("content", "")
                    if content and len(content) > 50:
                        response = content
                        break

    # Display the complete response - use markdown to show full content without truncation
    if response:
        # Clean up any termination signals for display
        display_response = response
        termination_signals = ["DRAFT COMPLETE", "APPROVED - RESEARCH COMPLETE", "APPROVED-RESEARCH COMPLETE", "TERMINATE"]
        for signal in termination_signals:
            display_response = display_response.replace(signal, "").strip()

        # Display in a container with proper formatting to ensure full visibility
        st.markdown(display_response)
    else:
        st.warning("⚠️ No response available. The workflow may not have completed successfully.")

    st.divider()

    # Display workflow stages below the response (terminal-like format)
    workflow_stages = result.get("workflow_stages", [])
    st.markdown("### 🔄 Workflow Stages")
    st.markdown("----------------------------------------------------------------------")
    if workflow_stages:
        stage_icons = {
            "planning": "📋",
            "researching": "🔍",
            "writing": "✍️",
            "critiquing": "✅",
            "revising": "🔄"
        }
        for stage in workflow_stages:
            icon = stage_icons.get(stage, "•")
            st.markdown(f"  {icon} {stage.replace('_', ' ').title()}")
    else:
        st.markdown("  ⚠️ No workflow stages recorded")
    st.markdown("----------------------------------------------------------------------")

    st.divider()

    # Display agent context and exchanges
    display_agent_context_and_exchanges(result)

    # Display citations and sources
    citations, sources = extract_citations_and_sources(result)
    if citations or sources:
        with st.expander("📚 Citations & Sources", expanded=True):
            if sources:
                st.markdown("#### Sources Found")
                for i, source in enumerate(sources[:15], 1):
                    source_type = source.get("type", "unknown")
                    title = source.get("title", "Unknown")
                    url = source.get("url", "")

                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**[{i}]** {source_type.upper()}: {title}")
                    with col2:
                        if url:
                            st.markdown(f"[🔗 Link]({url})")

            if citations:
                st.markdown("#### URLs Referenced")
                for i, citation in enumerate(citations, 1):
                    st.markdown(f"**[{i}]** [{citation}]({citation})")

    # Display metadata metrics
    metadata = result.get("metadata", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sources", metadata.get("num_sources", 0))
    with col2:
        st.metric("Messages", metadata.get("num_messages", 0))
    with col3:
        revisions = metadata.get("revision_count", 0)
        st.metric("Revisions", revisions)
    with col4:
        score = metadata.get("critique_score", 0)
        st.metric("Quality", f"{score:.2f}")

    # Display agents involved
    agents_involved = metadata.get("agents_involved", [])
    if agents_involved:
        st.markdown(f"**Agents Involved:** {', '.join(agents_involved)}")

    # Safety events
    safety_events = extract_safety_events(result)
    if safety_events:
        with st.expander("🛡️ Safety Events", expanded=True):
            for event in safety_events:
                event_type = event.get("type", "unknown")
                reason = event.get("reason", "Safety event detected")

                if event_type == "blocked":
                    st.error(f"⚠️ **BLOCKED:** {reason}")
                elif event_type == "sanitized":
                    st.warning(f"🔧 **SANITIZED:** {reason}")
                else:
                    st.info(f"ℹ️ **{event_type.upper()}:** {reason}")

    # Agent traces
    if st.session_state.show_traces:
        display_agent_traces_detailed(result)


def display_agent_context_and_exchanges(result: Dict[str, Any]):
    """
    Display all agent context and information exchanges below the final response.
    """
    conversation_history = result.get("conversation_history", [])

    st.markdown("### 🤖 Agent Context & Exchanges")

    if not conversation_history:
        st.info("ℹ️ No conversation history available. The agents may not have exchanged messages yet.")
        return

    # Agent icons mapping
    agent_icons = {
        "Planner": "📋",
        "Researcher": "🔍",
        "Writer": "✍️",
        "Critic": "✅",
        "System": "⚙️",
        "user": "👤",
        "User": "👤"
    }

    # Group messages by iteration for better organization
    messages_by_iteration = {}
    for msg in conversation_history:
        iteration = msg.get("iteration", 1)
        if iteration not in messages_by_iteration:
            messages_by_iteration[iteration] = []
        messages_by_iteration[iteration].append(msg)

    # Display messages organized by iteration
    for iteration in sorted(messages_by_iteration.keys()):
        iteration_messages = messages_by_iteration[iteration]

        # Show iteration header if there are multiple iterations
        if len(messages_by_iteration) > 1:
            st.markdown(f"#### Iteration {iteration}")

        # Display each message
        for i, msg in enumerate(iteration_messages, 1):
            source = msg.get("source", "Unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", None)

            icon = agent_icons.get(source, "•")

            # Create a container for each message
            with st.container():
                # Message header with agent name and icon
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.markdown(f"**{icon} {source}**")
                with col2:
                    if timestamp:
                        try:
                            dt = datetime.fromtimestamp(timestamp)
                            st.caption(f"Time: {dt.strftime('%H:%M:%S')}")
                        except (ValueError, TypeError, OSError):
                            pass  # Invalid timestamp, skip displaying time

                # Message content in an expandable/collapsible format
                with st.expander(f"Message {i} from {source}", expanded=False):
                    # Show full content
                    st.markdown(content)

                    # Show handoff signals if present
                    handoff_signals = []
                    content_upper = content.upper()
                    if "PLAN COMPLETE" in content_upper:
                        handoff_signals.append("✓ PLAN COMPLETE")
                    if "RESEARCH COMPLETE" in content_upper:
                        handoff_signals.append("✓ RESEARCH COMPLETE")
                    if "DRAFT COMPLETE" in content_upper:
                        handoff_signals.append("✓ DRAFT COMPLETE")
                    if "APPROVED" in content_upper or "TERMINATE" in content_upper:
                        handoff_signals.append("✓ APPROVED/TERMINATE")
                    if "NEEDS REVISION" in content_upper:
                        handoff_signals.append("⚠ NEEDS REVISION")

                    if handoff_signals:
                        st.markdown(f"**Handoff Signals:** {', '.join(handoff_signals)}")

                # Show a preview of the message content
                preview = content[:300] + "..." if len(content) > 300 else content
                st.text_area(
                    "",
                    value=preview,
                    height=80,
                    key=f"exchange_{iteration}_{i}_{source}",
                    disabled=True,
                    label_visibility="collapsed"
                )

                st.divider()


def display_agent_traces_detailed(result: Dict[str, Any]):
    """
    Display detailed agent execution traces.
    """
    conversation_history = result.get("conversation_history", [])
    if not conversation_history:
        return

    with st.expander("🔍 Agent Execution Traces", expanded=False):
        # Group messages by agent
        agent_messages = {}
        for msg in conversation_history:
            agent = msg.get("source", "Unknown")
            iteration = msg.get("iteration", 1)

            if agent not in agent_messages:
                agent_messages[agent] = []

            agent_messages[agent].append({
                "content": msg.get("content", ""),
                "iteration": iteration
            })

        # Display traces by agent
        agent_icons = {
            "Planner": "📋",
            "Researcher": "🔍",
            "Writer": "✍️",
            "Critic": "✅"
        }

        for agent, messages in agent_messages.items():
            icon = agent_icons.get(agent, "•")
            st.markdown(f"### {icon} {agent.upper()} ({len(messages)} message(s))")

            for i, msg in enumerate(messages, 1):
                content = msg.get("content", "")
                iteration = msg.get("iteration", 1)

                # Show iteration if > 1
                iter_label = f" [Iteration {iteration}]" if iteration > 1 else ""

                with st.container():
                    st.markdown(f"**Message {i}{iter_label}:**")

                    # Truncate and display content
                    preview = content[:500] + "..." if len(content) > 500 else content
                    st.text_area(
                        "",
                        value=preview,
                        height=100,
                        key=f"trace_{agent}_{i}",
                        disabled=True,
                        label_visibility="collapsed"
                    )

                    # Show handoff signals
                    handoff_signals = []
                    if "PLAN COMPLETE" in content:
                        handoff_signals.append("✓ PLAN COMPLETE")
                    if "RESEARCH COMPLETE" in content:
                        handoff_signals.append("✓ RESEARCH COMPLETE")
                    if "DRAFT COMPLETE" in content:
                        handoff_signals.append("✓ DRAFT COMPLETE")
                    if "APPROVED" in content or "TERMINATE" in content:
                        handoff_signals.append("✓ APPROVED/TERMINATE")
                    if "NEEDS REVISION" in content:
                        handoff_signals.append("⚠ NEEDS REVISION")

                    if handoff_signals:
                        st.markdown(f"**Handoff Signals:** {', '.join(handoff_signals)}")

                    st.divider()


def display_sidebar():
    """Display sidebar with settings and statistics."""
    with st.sidebar:
        st.title("⚙️ Settings")

        # Show traces toggle
        st.session_state.show_traces = st.checkbox(
            "Show Agent Traces",
            value=st.session_state.show_traces
        )

        # Show safety log toggle
        st.session_state.show_safety_log = st.checkbox(
            "Show Safety Log",
            value=st.session_state.show_safety_log
        )

        st.divider()

        st.title("📊 Statistics")

        # Get statistics from history
        total_queries = len(st.session_state.history)
        total_safety_events = sum(
            len(result.get("metadata", {}).get("safety_events", []))
            for item in st.session_state.history
            if item.get("result", {}).get("metadata", {}).get("safety_events")
        )

        st.metric("Total Queries", total_queries)
        st.metric("Safety Events", total_safety_events)

        st.divider()

        # Clear history button
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

        # About section
        st.divider()
        st.markdown("### About")
        config = load_config()
        system_name = config.get("system", {}).get("name", "Research Assistant")
        topic = config.get("system", {}).get("topic", "General")
        st.markdown(f"**System:** {system_name}")
        st.markdown(f"**Topic:** {topic}")


def display_history():
    """Display query history."""
    if not st.session_state.history:
        return

    with st.expander("📜 Query History", expanded=False):
        for i, item in enumerate(reversed(st.session_state.history), 1):
            timestamp = item.get("timestamp", "")
            query = item.get("query", "")
            st.markdown(f"**{i}.** [{timestamp}] {query}")


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Multi-Agent Research Assistant",
        page_icon="🤖",
        layout="wide"
    )

    initialize_session_state()

    # Header
    st.title("🤖 Multi-Agent Research Assistant")
    st.markdown("Ask me anything about your research topic!")

    # Sidebar
    display_sidebar()

    # Main area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Query input
        query = st.text_area(
            "Enter your research query:",
            height=100,
            placeholder="e.g., What are the latest developments in explainable AI for novice users?"
        )

        # Submit button
        if st.button("🔍 Search", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner("Processing your query..."):
                    # Process query
                    result = asyncio.run(process_query(query))

                    # Add to history
                    st.session_state.history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "query": query,
                        "result": result
                    })

                    # Display result
                    st.divider()
                    display_response(result)
            else:
                st.warning("Please enter a query.")

        # History
        display_history()

    with col2:
        st.markdown("### 💡 Example Queries")
        examples = [
            "What are the key principles of user-centered design?",
            "Explain recent advances in AR usability research",
            "Compare different approaches to AI transparency",
            "What are ethical considerations in AI for education?",
        ]

        for example in examples:
            if st.button(example, use_container_width=True):
                st.session_state.example_query = example
                st.rerun()

        # If example was clicked, populate the text area
        if 'example_query' in st.session_state:
            st.info(f"Example query selected: {st.session_state.example_query}")
            del st.session_state.example_query

        st.divider()

        st.markdown("### ℹ️ How It Works")
        st.markdown("""
        1. **Planner** breaks down your query
        2. **Researcher** gathers evidence
        3. **Writer** synthesizes findings
        4. **Critic** verifies quality
        5. **Safety** checks ensure appropriate content
        """)

    # Safety log (if enabled)
    if st.session_state.show_safety_log:
        st.divider()
        st.markdown("### 🛡️ Safety Event Log")

        # Collect all safety events from history
        all_safety_events = []
        for item in st.session_state.history:
            result = item.get("result", {})
            safety_events = extract_safety_events(result)
            if safety_events:
                all_safety_events.extend([
                    {
                        **event,
                        "query": item.get("query", ""),
                        "timestamp": item.get("timestamp", "")
                    }
                    for event in safety_events
                ])

        if all_safety_events:
            for event in all_safety_events:
                event_type = event.get("type", "unknown")
                reason = event.get("reason", "Safety event detected")
                query = event.get("query", "")
                timestamp = event.get("timestamp", "")

                with st.container():
                    if event_type == "blocked":
                        st.error(f"⚠️ **BLOCKED** - {reason}")
                    elif event_type == "sanitized":
                        st.warning(f"🔧 **SANITIZED** - {reason}")
                    else:
                        st.info(f"ℹ️ **{event_type.upper()}** - {reason}")

                    st.caption(f"Query: {query[:100]}... | Time: {timestamp}")
                    st.divider()
        else:
            st.info("✅ No safety events recorded. All queries passed safety checks.")


if __name__ == "__main__":
    main()
