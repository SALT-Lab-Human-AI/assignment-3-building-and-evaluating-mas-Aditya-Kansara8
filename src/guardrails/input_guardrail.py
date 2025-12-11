"""
Input Guardrail
Checks user inputs for safety violations.
"""

from typing import Dict, Any, List
import logging

# Try to import Guardrails AI
try:
    from guardrails import Guard
    GUARDRAILS_AVAILABLE = True
    # Note: guardrails-ai 0.6.8+ uses a different validator API
    # We'll use fallback validators that work regardless
    GUARDRAILS_VALIDATORS_AVAILABLE = False
except ImportError:
    GUARDRAILS_AVAILABLE = False
    GUARDRAILS_VALIDATORS_AVAILABLE = False
    Guard = None


class InputGuardrail:
    """
    Guardrail for checking input safety.

    TODO: YOUR CODE HERE
    - Integrate with Guardrails AI or NeMo Guardrails
    - Define validation rules
    - Implement custom validators
    - Handle different types of violations
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize input guardrail.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("safety.input_guardrail")

        # Initialize guardrail framework
        # Note: guardrails-ai 0.6.8+ has a different API structure
        # We use fallback validators that are more reliable
        if GUARDRAILS_AVAILABLE and GUARDRAILS_VALIDATORS_AVAILABLE:
            try:
                # Try to use Guardrails AI if validators are available
                self.guard = Guard()
                self.logger.info("Guardrails AI input guard initialized (basic)")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Guardrails AI: {e}")
                self.guard = None
        else:
            self.guard = None
            if GUARDRAILS_AVAILABLE:
                self.logger.info("Guardrails AI installed but using fallback validators (newer API)")
            else:
                self.logger.info("Guardrails AI not available. Using fallback validation.")

    def validate(self, query: str) -> Dict[str, Any]:
        """
        Validate input query.

        Args:
            query: User input to validate

        Returns:
            Validation result

        TODO: YOUR CODE HERE
        - Implement validation logic
        - Check for toxic language
        - Check for prompt injection attempts
        - Check query length and format
        - Check for off-topic queries
        """
        violations = []

        # Use Guardrails AI if available
        if self.guard is not None:
            try:
                result = self.guard.validate(query)
                if not result.validation_passed:
                    # Guardrails AI returns validation results
                    if hasattr(result, 'errors') and result.errors:
                        violations.extend([
                            {
                                "validator": "guardrails",
                                "reason": str(error),
                                "severity": "high"
                            }
                            for error in result.errors
                        ])
                    elif hasattr(result, 'error') and result.error:
                        violations.append({
                            "validator": "guardrails",
                            "reason": str(result.error),
                            "severity": "high"
                        })
            except Exception as e:
                # If validation fails, log and continue with fallback
                self.logger.warning(f"Guardrails validation error: {e}")
                violations.append({
                    "validator": "guardrails_error",
                    "reason": f"Validation error: {str(e)}",
                    "severity": "medium"
                })

        # Fallback validation checks
        if len(query) < 5:
            violations.append({
                "validator": "length",
                "reason": "Query too short (minimum 5 characters)",
                "severity": "low"
            })

        if len(query) > 2000:
            violations.append({
                "validator": "length",
                "reason": "Query too long (maximum 2000 characters)",
                "severity": "medium"
            })

        # Check for prompt injection
        injection_violations = self._check_prompt_injection(query)
        violations.extend(injection_violations)

        # Check for relevance (optional)
        relevance_violations = self._check_relevance(query)
        violations.extend(relevance_violations)

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "sanitized_input": query  # Could be modified version
        }

    def _check_toxic_language(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for toxic/harmful language.

        TODO: YOUR CODE HERE Implement toxicity detection
        """
        violations = []
        # Implement toxicity check
        return violations

    def _check_prompt_injection(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for prompt injection attempts.

        TODO: YOUR CODE HERE Implement prompt injection detection
        """
        violations = []
        # Check for common prompt injection patterns
        injection_patterns = [
            "ignore previous instructions",
            "disregard",
            "forget everything",
            "system:",
            "sudo",
        ]

        for pattern in injection_patterns:
            if pattern.lower() in text.lower():
                violations.append({
                    "validator": "prompt_injection",
                    "reason": f"Potential prompt injection: {pattern}",
                    "severity": "high"
                })

        return violations

    def _check_relevance(self, query: str) -> List[Dict[str, Any]]:
        """
        Check if query is relevant to the system's purpose.

        TODO: YOUR CODE HERE Implement relevance checking
        """
        violations = []
        # Check if query is about HCI research (or configured topic)
        return violations
