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
    """Run system evaluation using SystemEvaluator."""
    import yaml
    from dotenv import load_dotenv
    from src.autogen_orchestrator import AutoGenOrchestrator
    from src.evaluation.evaluator import SystemEvaluator
    from pathlib import Path

    # Load environment variables
    load_dotenv()

    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Check if evaluation is enabled
    eval_config = config.get("evaluation", {})
    if not eval_config.get("enabled", True):
        print("Evaluation is disabled in config.yaml")
        return

    # Initialize AutoGen orchestrator
    print("Initializing AutoGen orchestrator...")
    try:
        orchestrator = AutoGenOrchestrator(config)
        print("✓ Orchestrator initialized successfully\n")
    except Exception as e:
        print(f"✗ Failed to initialize orchestrator: {e}")
        print("Evaluation will continue with placeholder responses")
        orchestrator = None

    # Initialize evaluator
    print("Initializing SystemEvaluator...")
    evaluator = SystemEvaluator(config, orchestrator=orchestrator)
    print("✓ Evaluator initialized\n")

    # Determine test queries file
    test_queries_path = "data/example_queries.json"
    if not Path(test_queries_path).exists():
        print(f"✗ Test queries file not found: {test_queries_path}")
        print("Please ensure example_queries.json exists in the data/ directory")
        return

    print("=" * 70)
    print("RUNNING SYSTEM EVALUATION")
    print("=" * 70)
    print(f"\nTest queries file: {test_queries_path}")
    print(f"Number of queries: {eval_config.get('num_test_queries', 'all')}")
    print(f"Judge prompts: {eval_config.get('num_judge_prompts', 2)}")
    print(f"Evaluation criteria: {len(eval_config.get('criteria', []))}")
    print("\nThis may take several minutes depending on the number of queries...\n")

    # Run evaluation
    try:
        report = await evaluator.evaluate_system(test_queries_path)

        # Display summary
        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE")
        print("=" * 70)

        summary = report.get("summary", {})
        print(f"\nTotal Queries: {summary.get('total_queries', 0)}")
        print(f"Successful: {summary.get('successful', 0)}")
        print(f"Failed: {summary.get('failed', 0)}")
        print(f"Success Rate: {summary.get('success_rate', 0.0):.2%}")

        scores = report.get("scores", {})
        print(f"\nOverall Average Score: {scores.get('overall_average', 0.0):.3f}")

        print("\nScores by Criterion:")
        for criterion, score in scores.get("by_criterion", {}).items():
            print(f"  {criterion}: {score:.3f}")

        # Show best and worst results
        best = report.get("best_result")
        worst = report.get("worst_result")

        if best:
            print(f"\nBest Result:")
            print(f"  Query: {best.get('query', '')[:60]}...")
            print(f"  Score: {best.get('score', 0.0):.3f}")

        if worst:
            print(f"\nWorst Result:")
            print(f"  Query: {worst.get('query', '')[:60]}...")
            print(f"  Score: {worst.get('score', 0.0):.3f}")

        print("\n" + "=" * 70)
        print("Detailed results saved to outputs/ directory")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


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
