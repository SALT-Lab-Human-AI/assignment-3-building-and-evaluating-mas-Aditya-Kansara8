#!/bin/bash
# End-to-End Demo Script
# Runs a single query through the full pipeline: query → agents → synthesis → judge scoring

set -e

# Default query
QUERY="${1:-What are the key principles of explainable AI for novice users?}"

echo "=========================================="
echo "Multi-Agent Research System - Demo"
echo "=========================================="
echo ""
echo "This demo will:"
echo "  1. Process query through all agents (Planner → Researcher → Writer → Critic)"
echo "  2. Generate final synthesis"
echo "  3. Evaluate response with LLM-as-a-Judge"
echo ""
echo "Query: $QUERY"
echo ""
echo "Starting demo..."
echo ""

# Run the demo
python main.py --mode demo --query "$QUERY"

echo ""
echo "=========================================="
echo "Demo complete! Check outputs/ for full results."
echo "=========================================="
