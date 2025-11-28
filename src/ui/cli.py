"""
Command Line Interface
Interactive CLI for the multi-agent research system.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from typing import Dict, Any
import yaml
import logging
from dotenv import load_dotenv

from src.autogen_orchestrator import AutoGenOrchestrator

# Load environment variables
load_dotenv()

class CLI:
    """
    Command-line interface for the research assistant.

    TODO: YOUR CODE HERE
    - Implement interactive prompt loop
    - Display agent traces clearly
    - Show citations and sources
    - Indicate safety events (blocked/sanitized)
    - Handle user commands (help, quit, clear, etc.)
    - Format output nicely
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize CLI.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup logging
        self._setup_logging()

        # Initialize AutoGen orchestrator
        try:
            self.orchestrator = AutoGenOrchestrator(self.config)
            self.logger = logging.getLogger("cli")
            self.logger.info("AutoGen orchestrator initialized successfully")
        except Exception as e:
            self.logger = logging.getLogger("cli")
            self.logger.error(f"Failed to initialize orchestrator: {e}")
            raise

        self.running = True
        self.query_count = 0
        self.verbose_mode = False

    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})
        log_level = log_config.get("level", "INFO")
        log_format = log_config.get(
            "format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format
        )

    async def run(self):
        """
        Main CLI loop.

        TODO: YOUR CODE HERE
        - Implement interactive loop
        - Handle user input
        - Process queries through orchestrator
        - Display results
        - Handle errors gracefully
        """
        self._print_welcome()

        while self.running:
            try:
                # Get user input
                query = input("\nEnter your research query (or 'help' for commands): ").strip()

                if not query:
                    continue

                # Handle commands
                if query.lower() in ['quit', 'exit', 'q']:
                    self._print_goodbye()
                    break
                elif query.lower() == 'help':
                    self._print_help()
                    continue
                elif query.lower() == 'clear':
                    self._clear_screen()
                    continue
                elif query.lower() == 'stats':
                    self._print_stats()
                    continue
                elif query.lower() == 'verbose':
                    self._toggle_verbose()
                    continue

                # Process query
                print("\n" + "=" * 70)
                print("Processing your query...")
                print("=" * 70)

                try:
                    # Process through orchestrator (synchronous call, not async)
                    result = self.orchestrator.process_query(query)
                    self.query_count += 1

                    # Display result
                    self._display_result(result)

                except Exception as e:
                    print(f"\nError processing query: {e}")
                    logging.exception("Error processing query")

            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                self._print_goodbye()
                break
            except Exception as e:
                print(f"\nError: {e}")
                logging.exception("Error in CLI loop")

    def _print_welcome(self):
        """Print welcome message."""
        print("=" * 70)
        print(f"  {self.config['system']['name']}")
        print(f"  Topic: {self.config['system']['topic']}")
        print("=" * 70)
        print("\nWelcome! Ask me anything about your research topic.")
        print("Type 'help' for available commands, or 'quit' to exit.\n")

    def _print_help(self):
        """Print help message."""
        print("\nAvailable commands:")
        print("  help    - Show this help message")
        print("  clear   - Clear the screen")
        print("  stats   - Show system statistics")
        print("  verbose - Toggle verbose mode (show agent traces)")
        print("  quit    - Exit the application")
        print("\nOr enter a research query to get started!")
        print("\nTip: Use 'verbose' to see detailed agent execution traces.")

    def _print_goodbye(self):
        """Print goodbye message."""
        print("\nThank you for using the Multi-Agent Research Assistant!")
        print("Goodbye!\n")

    def _clear_screen(self):
        """Clear the terminal screen."""
        import subprocess  # nosec B404
        # Use subprocess instead of os.system for better security
        subprocess.run(['clear' if sys.platform != 'win32' else 'cls'], shell=False, check=False)  # nosec B603 - Safe: no user input

    def _print_stats(self):
        """Print system statistics."""
        print("\nSystem Statistics:")
        print(f"  Queries processed: {self.query_count}")
        print(f"  System: {self.config.get('system', {}).get('name', 'Unknown')}")
        print(f"  Topic: {self.config.get('system', {}).get('topic', 'Unknown')}")
        print(f"  Model: {self.config.get('models', {}).get('default', {}).get('name', 'Unknown')}")

    def _display_result(self, result: Dict[str, Any]):
        """Display query result with formatting."""
        print("\n" + "=" * 70)
        print("RESPONSE")
        print("=" * 70)

        # Check for errors
        if "error" in result:
            print(f"\n❌ Error: {result['error']}")
            if result.get("metadata", {}).get("error_type"):
                print(f"   Error Type: {result['metadata']['error_type']}")
            return

        # Display response
        response = result.get("response", "")
        print(f"\n{response}\n")

        # Display workflow stages if available
        workflow_stages = result.get("workflow_stages", [])
        if workflow_stages:
            print("\n" + "-" * 70)
            print("🔄 WORKFLOW STAGES")
            print("-" * 70)
            stage_icons = {
                "planning": "📋",
                "researching": "🔍",
                "writing": "✍️",
                "critiquing": "✅",
                "revising": "🔄"
            }
            for stage in workflow_stages:
                icon = stage_icons.get(stage, "•")
                print(f"  {icon} {stage.replace('_', ' ').title()}")

        # Extract and display citations from conversation
        citations, sources = self._extract_citations_and_sources(result)
        if citations or sources:
            print("\n" + "-" * 70)
            print("📚 CITATIONS & SOURCES")
            print("-" * 70)

            if sources:
                print("\n  Sources Found:")
                for i, source in enumerate(sources[:10], 1):
                    source_type = source.get("type", "unknown")
                    title = source.get("title", "Unknown")
                    url = source.get("url", "")
                    if url:
                        print(f"    [{i}] {source_type.upper()}: {title}")
                        print(f"        {url}")
                    else:
                        print(f"    [{i}] {source_type.upper()}: {title}")

            if citations:
                print("\n  URLs Referenced:")
                for i, citation in enumerate(citations, 1):
                    print(f"    [{i}] {citation}")

        # Display metadata
        metadata = result.get("metadata", {})
        if metadata:
            print("\n" + "-" * 70)
            print("📊 METADATA")
            print("-" * 70)
            print(f"  • Messages exchanged: {metadata.get('num_messages', 0)}")
            print(f"  • Sources gathered: {metadata.get('num_sources', 0)}")
            print(f"  • Agents involved: {', '.join(metadata.get('agents_involved', []))}")

            if metadata.get("revision_count", 0) > 0:
                print(f"  • Revisions performed: {metadata.get('revision_count', 0)}")

            if metadata.get("iterations"):
                print(f"  • Iterations: {metadata.get('iterations', 1)}")

        # Display safety events if any
        safety_events = self._extract_safety_events(result)
        if safety_events:
            print("\n" + "-" * 70)
            print("🛡️ SAFETY EVENTS")
            print("-" * 70)
            for event in safety_events:
                event_type = event.get("type", "unknown")
                if event_type == "blocked":
                    print(f"  ⚠️  BLOCKED: {event.get('reason', 'Content blocked by safety system')}")
                elif event_type == "sanitized":
                    print(f"  🔧 SANITIZED: {event.get('reason', 'Content sanitized by safety system')}")
                else:
                    print(f"  ℹ️  {event_type.upper()}: {event.get('reason', 'Safety event detected')}")

        # Display agent traces
        if self._should_show_traces():
            self._display_agent_traces(result)

        print("=" * 70 + "\n")

    def _extract_citations_and_sources(self, result: Dict[str, Any]) -> tuple:
        """
        Extract citations/URLs and sources from conversation history.

        Returns:
            Tuple of (citations: list, sources: list)
        """
        citations = []
        sources = []
        seen_urls = set()

        import re

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

    def _extract_safety_events(self, result: Dict[str, Any]) -> list:
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

    def _should_show_traces(self) -> bool:
        """Check if agent traces should be displayed."""
        # Check both config and user preference
        return self.verbose_mode or self.config.get("ui", {}).get("verbose", False)

    def _toggle_verbose(self):
        """Toggle verbose mode."""
        self.verbose_mode = not self.verbose_mode
        status = "enabled" if self.verbose_mode else "disabled"
        print(f"\nVerbose mode {status}. Agent traces will {'now' if self.verbose_mode else 'no longer'} be displayed.")

    def _display_agent_traces(self, result: Dict[str, Any]):
        """Display detailed agent execution traces."""
        conversation_history = result.get("conversation_history", [])
        if not conversation_history:
            return

        print("\n" + "-" * 70)
        print("🔍 AGENT EXECUTION TRACES")
        print("-" * 70)

        # Group messages by agent
        agent_messages = {}
        for msg in conversation_history:
            agent = msg.get("source", "Unknown")
            iteration = msg.get("iteration", 1)
            timestamp = msg.get("timestamp", 0)

            if agent not in agent_messages:
                agent_messages[agent] = []

            agent_messages[agent].append({
                "content": msg.get("content", ""),
                "iteration": iteration,
                "timestamp": timestamp
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
            print(f"\n{icon} {agent.upper()} ({len(messages)} message(s)):")
            print("-" * 70)

            for i, msg in enumerate(messages, 1):
                content = msg.get("content", "")
                iteration = msg.get("iteration", 1)

                # Show iteration if > 1
                iter_label = f" [Iteration {iteration}]" if iteration > 1 else ""

                # Truncate and format content
                preview = content[:300] + "..." if len(content) > 300 else content
                preview = preview.replace("\n", " ").strip()

                print(f"\n  Message {i}{iter_label}:")
                print(f"  {preview}")

                # Show handoff signals
                if "PLAN COMPLETE" in content:
                    print("  ✓ Handoff: PLAN COMPLETE")
                elif "RESEARCH COMPLETE" in content:
                    print("  ✓ Handoff: RESEARCH COMPLETE")
                elif "DRAFT COMPLETE" in content:
                    print("  ✓ Handoff: DRAFT COMPLETE")
                elif "APPROVED" in content or "TERMINATE" in content:
                    print("  ✓ Handoff: APPROVED/TERMINATE")
                elif "NEEDS REVISION" in content:
                    print("  ⚠ Handoff: NEEDS REVISION")


def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Assistant CLI"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )

    args = parser.parse_args()

    # Run CLI
    cli = CLI(config_path=args.config)
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
