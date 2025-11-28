"""
Main Entry Point
Can be used to run the system or evaluation.

Usage:
  python main.py --mode cli           # Run CLI interface
  python main.py --mode web           # Run web interface
  python main.py --mode evaluate      # Run evaluation
"""

import argparse
import asyncio
import sys
from pathlib import Path


def run_cli(config_path="config.yaml"):
    """Run CLI interface."""
    # Temporarily modify sys.argv to prevent CLI from parsing --mode
    # The CLI's ArgumentParser only accepts --config, not --mode
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0], '--config', config_path]

    try:
        from src.ui.cli import CLI
        cli = CLI(config_path=config_path)
        asyncio.run(cli.run())
    finally:
        sys.argv = original_argv


def run_web():
    """Run web interface."""
    import subprocess  # nosec B404
    print("Starting Streamlit web interface...")
    subprocess.run(["streamlit", "run", "src/ui/streamlit_app.py"])  # nosec B603, B607 - Safe: hardcoded command, no user input


async def run_evaluation():
    """Run system evaluation."""
    import yaml
    from dotenv import load_dotenv
    from src.autogen_orchestrator import AutoGenOrchestrator

    # Load environment variables
    load_dotenv()

    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Initialize AutoGen orchestrator
    print("Initializing AutoGen orchestrator...")
    orchestrator = AutoGenOrchestrator(config)

    # For now, run a simple test query
    # TODO: Integrate with SystemEvaluator for full evaluation
    print("\n" + "=" * 70)
    print("RUNNING TEST QUERY")
    print("=" * 70)

    test_query = "What are the key principles of accessible user interface design?"
    print(f"\nQuery: {test_query}\n")

    result = orchestrator.process_query(test_query)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nResponse:\n{result.get('response', 'No response generated')}")
    print(f"\nMetadata:")
    print(f"  - Messages: {result.get('metadata', {}).get('num_messages', 0)}")
    print(f"  - Sources: {result.get('metadata', {}).get('num_sources', 0)}")

    print("\n" + "=" * 70)
    print("Note: Full evaluation with SystemEvaluator can be implemented")
    print("=" * 70)


def run_autogen():
    """Run AutoGen example."""
    import subprocess  # nosec B404
    print("Running AutoGen example...")
    subprocess.run([sys.executable, "example_autogen.py"])  # nosec B603 - Safe: hardcoded command, no user input


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Multi-Agent Research Assistant")
    parser.add_argument("--mode", choices=["cli", "web", "evaluate", "autogen"], default="autogen", help="Mode to run")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    if args.mode == "cli":
        run_cli(config_path=args.config)
    elif args.mode == "web":
        run_web()
    elif args.mode == "evaluate":
        asyncio.run(run_evaluation())
    elif args.mode == "autogen":
        run_autogen()


if __name__ == "__main__":
    main()
