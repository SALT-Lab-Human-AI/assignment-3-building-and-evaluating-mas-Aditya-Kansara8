"""
Main Entry Point
Can be used to run the system or evaluation.

Usage:
  python main.py --mode cli           # Run CLI interface
  python main.py --mode web           # Run web interface
  python main.py --mode evaluate      # Run evaluation
  python main.py --mode demo          # Run single-query end-to-end demo
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

        # Give time for async cleanup tasks to complete
        # This helps prevent "Event loop is closed" errors
        await asyncio.sleep(0.5)

    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


def run_autogen():
    """Run AutoGen example."""
    import subprocess  # nosec B404
    print("Running AutoGen example...")
    subprocess.run([sys.executable, "example_autogen.py"])  # nosec B603 - Safe: hardcoded command, no user input


async def run_demo(query: str = None):
    """
    Run a single-query end-to-end demo: query → agents → synthesis → judge scoring.

    This demonstrates the full pipeline:
    1. Query is processed through all agents (Planner → Researcher → Writer → Critic)
    2. Final synthesis is generated
    3. Judge evaluates the response using multiple criteria
    4. Results are displayed with scores and reasoning
    """
    import yaml
    from dotenv import load_dotenv
    from src.autogen_orchestrator import AutoGenOrchestrator
    from src.evaluation.judge import LLMJudge
    import json
    from datetime import datetime

    # Load environment variables
    load_dotenv()

    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Default query if not provided
    if not query:
        query = "What are the key principles of explainable AI for novice users?"

    print("=" * 70)
    print("END-TO-END DEMO: Query → Agents → Synthesis → Judge Scoring")
    print("=" * 70)
    print(f"\nQuery: {query}\n")

    # Step 1: Initialize orchestrator
    print("Step 1: Initializing Multi-Agent System...")
    try:
        orchestrator = AutoGenOrchestrator(config)
        print("✓ Orchestrator initialized successfully\n")
    except Exception as e:
        print(f"✗ Failed to initialize orchestrator: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 2: Process query through agents
    print("=" * 70)
    print("Step 2: Processing Query Through Agents")
    print("=" * 70)
    print("\nWorkflow: Planner → Researcher → Writer → Critic\n")
    print("This may take 1-3 minutes depending on query complexity...\n")

    try:
        result = orchestrator.process_query(query)

        if "error" in result:
            print(f"✗ Error processing query: {result.get('error')}")
            print(f"Response: {result.get('response', 'No response')}")
            return

        # Display agent workflow
        print("\n" + "-" * 70)
        print("AGENT WORKFLOW COMPLETE")
        print("-" * 70)
        print(f"\nWorkflow Stages: {' → '.join(result.get('workflow_stages', []))}")
        print(f"Messages Exchanged: {result.get('metadata', {}).get('num_messages', 0)}")
        print(f"Sources Gathered: {result.get('metadata', {}).get('num_sources', 0)}")
        print(f"Revisions: {result.get('metadata', {}).get('revision_count', 0)}")
        print(f"Agents Involved: {', '.join(result.get('metadata', {}).get('agents_involved', []))}")

        # Display final response
        response = result.get("response", "")
        print("\n" + "-" * 70)
        print("FINAL SYNTHESIS")
        print("-" * 70)
        print(f"\n{response[:500]}{'...' if len(response) > 500 else ''}\n")

    except Exception as e:
        print(f"✗ Error during agent processing: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 3: Evaluate with judge
    print("=" * 70)
    print("Step 3: Evaluating Response with LLM-as-a-Judge")
    print("=" * 70)
    print("\nEvaluating across multiple criteria with independent judge prompts...\n")

    try:
        judge = LLMJudge(config)

        # Extract sources from metadata
        sources = result.get("metadata", {}).get("sources", [])
        if not sources:
            # Try to extract from research findings
            research_findings = result.get("metadata", {}).get("research_findings", [])
            sources = [{"content": f} for f in research_findings] if research_findings else []

        evaluation = await judge.evaluate(
            query=query,
            response=response,
            sources=sources,
            ground_truth=None
        )

        # Display evaluation results
        print("\n" + "-" * 70)
        print("JUDGE EVALUATION RESULTS")
        print("-" * 70)

        overall_score = evaluation.get("overall_score", 0.0)
        print(f"\nOverall Score: {overall_score:.3f} / 1.000")

        print("\nScores by Criterion:")
        criterion_scores = evaluation.get("criterion_scores", {})
        for criterion, score_data in criterion_scores.items():
            score = score_data.get("score", 0.0)
            weight = next((c.get("weight", 0.0) for c in config.get("evaluation", {}).get("criteria", []) if c.get("name") == criterion), 0.0)
            print(f"  • {criterion.replace('_', ' ').title()}: {score:.3f} (weight: {weight:.2f})")
            reasoning = score_data.get("reasoning", "")
            if reasoning:
                # Show first judge's reasoning (truncated)
                first_reasoning = reasoning.split("\n\n")[0] if "\n\n" in reasoning else reasoning
                print(f"    Reasoning: {first_reasoning[:150]}{'...' if len(first_reasoning) > 150 else ''}")

        print("\n" + "-" * 70)
        print("DEMO COMPLETE")
        print("-" * 70)
        print("\nSummary:")
        print(f"  Query: {query[:60]}{'...' if len(query) > 60 else ''}")
        print(f"  Response Length: {len(response)} characters")
        print(f"  Overall Score: {overall_score:.3f}")
        print(f"  Workflow Stages: {len(result.get('workflow_stages', []))}")
        print(f"  Agents Involved: {len(result.get('metadata', {}).get('agents_involved', []))}")

        # Save demo results
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        demo_file = output_dir / f"demo_{timestamp}.json"

        demo_result = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "evaluation": evaluation,
            "metadata": result.get("metadata", {}),
            "workflow_stages": result.get("workflow_stages", [])
        }

        with open(demo_file, 'w') as f:
            json.dump(demo_result, f, indent=2)

        print(f"\n✓ Full results saved to: {demo_file}")

    except Exception as e:
        print(f"✗ Error during judge evaluation: {e}")
        import traceback
        traceback.print_exc()
        return


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Multi-Agent Research Assistant")
    parser.add_argument("--mode", choices=["cli", "web", "evaluate", "autogen", "demo"], default="autogen", help="Mode to run")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--query", default=None, help="Query for demo mode")
    args = parser.parse_args()

    if args.mode == "cli":
        run_cli(config_path=args.config)
    elif args.mode == "web":
        run_web()
    elif args.mode == "evaluate":
        # Suppress asyncio cleanup errors that occur after evaluation completes
        # These are harmless cleanup errors from AutoGen's HTTP clients

        # Set up custom exception handler to suppress event loop cleanup errors
        def handle_exception(loop, context):
            exception = context.get('exception')
            if isinstance(exception, RuntimeError) and 'Event loop is closed' in str(exception):
                # Suppress this specific error - it's a harmless cleanup issue
                return
            # For other exceptions, use default handler
            try:
                if hasattr(loop, 'default_exception_handler') and loop.default_exception_handler:
                    loop.default_exception_handler(context)
                else:
                    # Fallback: log to stderr
                    print(f"Exception in event loop: {context}", file=sys.stderr)
            except Exception:
                # If default handler fails, just log
                print(f"Exception in event loop: {context}", file=sys.stderr)

        try:
            # Create new event loop with custom exception handler
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.set_exception_handler(handle_exception)

            # Run evaluation
            loop.run_until_complete(run_evaluation())

            # Give a moment for cleanup tasks
            try:
                pending = asyncio.all_tasks(loop)
                if pending:
                    # Wait briefly for cleanup, but don't block forever
                    loop.run_until_complete(asyncio.wait_for(
                        asyncio.gather(*pending, return_exceptions=True),
                        timeout=1.0
                    ))
            except (asyncio.TimeoutError, Exception):
                pass  # Ignore cleanup timeout/errors
        finally:
            # Close the loop
            try:
                loop.close()
            except Exception:
                pass  # Ignore errors during loop closure
    elif args.mode == "autogen":
        run_autogen()
    elif args.mode == "demo":
        asyncio.run(run_demo(query=args.query))


if __name__ == "__main__":
    main()
