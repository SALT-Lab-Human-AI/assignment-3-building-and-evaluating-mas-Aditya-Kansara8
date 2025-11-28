"""
Base Agent Class

This module provides a base class for all research agents in the system.
All specialized agents (Planner, Researcher, Writer, Critic) inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging


class BaseAgent(ABC):
    """
    Base class for all research agents.

    This provides common functionality and defines the interface that all agents must implement.
    Agents can be used independently or integrated with frameworks like AutoGen.
    """

    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str = "",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the base agent.

        Args:
            name: Agent name (e.g., "Planner", "Researcher")
            role: Agent role description
            system_prompt: System prompt for the agent (empty string uses default)
            config: Optional configuration dictionary
        """
        self.name = name
        self.role = role
        self.config = config or {}
        self.logger = logging.getLogger(f"agents.{name.lower()}")

        # Set system prompt (use default if empty)
        if system_prompt and system_prompt.strip():
            self.system_prompt = system_prompt.strip()
        else:
            self.system_prompt = self._get_default_prompt()

        self.logger.info(f"Initialized {self.name} agent")

    @abstractmethod
    def _get_default_prompt(self) -> str:
        """
        Get the default system prompt for this agent.

        Returns:
            Default system prompt string
        """
        pass

    @abstractmethod
    def process(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process input and return results.

        This is the main method that each agent implements to perform its specific task.

        Args:
            input_data: Input to process (query, research findings, draft, etc.)
            context: Optional context dictionary with additional information

        Returns:
            Dictionary with:
            - output: The agent's output/response
            - metadata: Additional information about the processing
        """
        pass

    def get_system_prompt(self) -> str:
        """Get the current system prompt."""
        return self.system_prompt

    def set_system_prompt(self, prompt: str):
        """Update the system prompt."""
        self.system_prompt = prompt.strip()
        self.logger.info(f"Updated system prompt for {self.name}")

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this agent.

        Returns:
            Dictionary with agent name, role, and configuration
        """
        return {
            "name": self.name,
            "role": self.role,
            "system_prompt": self.system_prompt[:100] + "..." if len(self.system_prompt) > 100 else self.system_prompt,
        }
