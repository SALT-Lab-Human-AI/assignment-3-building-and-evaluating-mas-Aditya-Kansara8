"""
Agent Implementations

This package contains all agent implementations for the multi-agent research system.

Agents:
- BaseAgent: Base class for all agents
- PlannerAgent: Task planning agent
- ResearcherAgent: Evidence gathering agent
- WriterAgent: Response synthesis agent
- CriticAgent: Quality evaluation agent
"""

from src.agents.base_agent import BaseAgent
from src.agents.planner_agent import PlannerAgent, create_planner_agent
from src.agents.researcher_agent import ResearcherAgent, create_researcher_agent
from src.agents.writer_agent import WriterAgent, create_writer_agent
from src.agents.critic_agent import CriticAgent, create_critic_agent

__all__ = [
    "BaseAgent",
    "PlannerAgent",
    "ResearcherAgent",
    "WriterAgent",
    "CriticAgent",
    "create_planner_agent",
    "create_researcher_agent",
    "create_writer_agent",
    "create_critic_agent",
]
