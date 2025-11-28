"""
Safety Manager
Coordinates safety guardrails and logs safety events.
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import json

# Try to import Guardrails AI
try:
    from guardrails import Guard
    # Note: guardrails-ai 0.6.8+ uses a different API structure
    # Validators may need to be registered or used differently
    # We'll use a fallback approach that works with or without specific validators
    GUARDRAILS_AVAILABLE = True
    GUARDRAILS_VALIDATORS_AVAILABLE = False

    # Try to check if validators are available in the expected format
    try:
        from guardrails.validators import register_validator
        # Check if we can use the validator system
        GUARDRAILS_VALIDATORS_AVAILABLE = True
    except ImportError:
        GUARDRAILS_VALIDATORS_AVAILABLE = False
except ImportError:
    GUARDRAILS_AVAILABLE = False
    GUARDRAILS_VALIDATORS_AVAILABLE = False
    Guard = None

from .input_guardrail import InputGuardrail
from .output_guardrail import OutputGuardrail


class SafetyManager:
    """
    Manages safety guardrails for the multi-agent system.

    TODO: YOUR CODE HERE
    - Integrate with Guardrails AI or NeMo Guardrails
    - Define safety policies
    - Implement logging of safety events
    - Handle different violation types with appropriate responses
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize safety manager.

        Args:
            config: Safety configuration
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.log_events = config.get("log_events", True)
        self.logger = logging.getLogger("safety")

        # Safety event log
        self.safety_events: List[Dict[str, Any]] = []

        # Prohibited categories
        self.prohibited_categories = config.get("prohibited_categories", [
            "harmful_content",
            "personal_attacks",
            "misinformation",
            "off_topic_queries"
        ])

        # Violation response strategy
        self.on_violation = config.get("on_violation", {})

        # Initialize guardrail framework
        self.framework = config.get("framework", "guardrails").lower()

        # Initialize input and output guardrails (these handle their own Guardrails AI integration)
        self.input_guardrail = InputGuardrail(config)
        self.output_guardrail = OutputGuardrail(config)

        # Initialize Guardrails AI guards if available (for direct use in safety_manager)
        # Note: guardrails-ai 0.6.8+ has a different API, so we use fallback validators
        if GUARDRAILS_AVAILABLE and self.framework == "guardrails":
            if GUARDRAILS_VALIDATORS_AVAILABLE:
                try:
                    # Try to use Guardrails AI with validators if available
                    # This would work with older versions or if validators are properly registered
                    self.input_guard = Guard()
                    self.output_guard = Guard()
                    self.logger.info("Guardrails AI framework initialized (using basic Guard)")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize Guardrails AI: {e}. Using fallback implementation.")
                    self.input_guard = None
                    self.output_guard = None
            else:
                # Guardrails AI is installed but validators aren't in expected format
                # Use fallback implementation which is more robust
                self.logger.info("Guardrails AI installed but using fallback validators (newer API structure)")
                self.input_guard = None
                self.output_guard = None
        else:
            if not GUARDRAILS_AVAILABLE:
                self.logger.warning("Guardrails AI not available. Install with: pip install guardrails-ai")
            self.input_guard = None
            self.output_guard = None

    def check_input_safety(self, query: str) -> Dict[str, Any]:
        """
        Check if input query is safe to process.

        Args:
            query: User query to check

        Returns:
            Dictionary with 'safe' boolean and optional 'violations' list
        """
        if not self.enabled:
            return {"safe": True}

        violations = []
        sanitized_query = query

        # Use Guardrails AI if available
        if self.input_guard is not None:
            try:
                result = self.input_guard.validate(query)
                if not result.validation_passed:
                    # Guardrails AI returns validation results
                    if hasattr(result, 'errors') and result.errors:
                        violations.extend([
                            {
                                "category": "guardrails_validation",
                                "reason": str(error),
                                "severity": "high"
                            }
                            for error in result.errors
                        ])
                    elif hasattr(result, 'error') and result.error:
                        violations.append({
                            "category": "guardrails_validation",
                            "reason": str(result.error),
                            "severity": "high"
                        })
            except Exception as e:
                # If guardrails validation fails, log and continue with fallback
                self.logger.warning(f"Guardrails AI validation error: {e}. Using fallback checks.")
                violations.append({
                    "category": "validation_error",
                    "reason": f"Guardrails validation failed: {str(e)}",
                    "severity": "medium"
                })

        # Use InputGuardrail for additional checks
        input_validation = self.input_guardrail.validate(query)
        if not input_validation.get("valid", True):
            violations.extend(input_validation.get("violations", []))

        # Check for prohibited categories
        for category in self.prohibited_categories:
            category_violations = self._check_prohibited_category(query, category)
            violations.extend(category_violations)

        is_safe = len(violations) == 0

        # Log safety event
        if not is_safe and self.log_events:
            self._log_safety_event("input", query, violations, is_safe)

        return {
            "safe": is_safe,
            "violations": violations,
            "sanitized_query": sanitized_query
        }

    def check_output_safety(self, response: str) -> Dict[str, Any]:
        """
        Check if output response is safe to return.

        Args:
            response: Generated response to check

        Returns:
            Dictionary with 'safe' boolean and optional 'violations' list
        """
        if not self.enabled:
            return {"safe": True, "response": response}

        violations = []
        sanitized_response = response

        # Use Guardrails AI if available
        if self.output_guard is not None:
            try:
                result = self.output_guard.validate(response)
                if not result.validation_passed:
                    # Guardrails AI returns validation results
                    if hasattr(result, 'errors') and result.errors:
                        violations.extend([
                            {
                                "category": "guardrails_validation",
                                "reason": str(error),
                                "severity": "high"
                            }
                            for error in result.errors
                        ])
                    elif hasattr(result, 'error') and result.error:
                        violations.append({
                            "category": "guardrails_validation",
                            "reason": str(result.error),
                            "severity": "high"
                        })
            except Exception as e:
                # If guardrails validation fails, log and continue with fallback
                self.logger.warning(f"Guardrails AI validation error: {e}. Using fallback checks.")
                violations.append({
                    "category": "validation_error",
                    "reason": f"Guardrails validation failed: {str(e)}",
                    "severity": "medium"
                })

        # Use OutputGuardrail for additional checks
        output_validation = self.output_guardrail.validate(response)
        if not output_validation.get("valid", True):
            violations.extend(output_validation.get("violations", []))
            # Use sanitized output if available
            if "sanitized_output" in output_validation:
                sanitized_response = output_validation["sanitized_output"]

        is_safe = len(violations) == 0

        # Log safety event
        if not is_safe and self.log_events:
            self._log_safety_event("output", response, violations, is_safe)

        result = {
            "safe": is_safe,
            "violations": violations,
            "response": sanitized_response if is_safe else response
        }

        # Apply sanitization if configured
        if not is_safe:
            action = self.on_violation.get("action", "refuse")
            if action == "sanitize":
                result["response"] = self._sanitize_response(sanitized_response, violations)
            elif action == "refuse":
                result["response"] = self.on_violation.get(
                    "message",
                    "I cannot provide this response due to safety policies."
                )

        return result

    def _sanitize_response(self, response: str, violations: List[Dict[str, Any]]) -> str:
        """
        Sanitize response by removing or redacting unsafe content.

        Args:
            response: Original response
            violations: List of violations found

        Returns:
            Sanitized response
        """
        sanitized = response

        # Redact based on violation types
        for violation in violations:
            category = violation.get("category", "")
            if "pii" in category.lower() or "personal" in category.lower():
                # Redact PII using output guardrail
                output_validation = self.output_guardrail.validate(response)
                if "sanitized_output" in output_validation:
                    sanitized = output_validation["sanitized_output"]
            elif "toxic" in category.lower() or "harmful" in category.lower():
                # Replace harmful content with placeholder
                sanitized = sanitized.replace(
                    violation.get("reason", ""),
                    "[Content removed due to safety policy]"
                )

        return sanitized

    def _check_prohibited_category(self, content: str, category: str) -> List[Dict[str, Any]]:
        """
        Check if content violates a prohibited category.

        Args:
            content: Content to check
            category: Category to check against

        Returns:
            List of violations
        """
        violations = []

        if category == "harmful_content":
            harmful_keywords = ["violence", "harmful", "dangerous", "illegal"]
            for keyword in harmful_keywords:
                if keyword.lower() in content.lower():
                    violations.append({
                        "category": category,
                        "reason": f"Contains potentially harmful content: {keyword}",
                        "severity": "high"
                    })

        elif category == "personal_attacks":
            attack_patterns = ["you are", "you're", "stupid", "idiot", "hate"]
            for pattern in attack_patterns:
                if pattern.lower() in content.lower():
                    violations.append({
                        "category": category,
                        "reason": f"Contains personal attack language: {pattern}",
                        "severity": "medium"
                    })

        elif category == "misinformation":
            # This would ideally use fact-checking, but for now we check for obvious red flags
            misinformation_patterns = ["definitely false", "proven wrong", "debunked"]
            for pattern in misinformation_patterns:
                if pattern.lower() in content.lower():
                    violations.append({
                        "category": category,
                        "reason": f"Potential misinformation detected: {pattern}",
                        "severity": "high"
                    })

        elif category == "off_topic_queries":
            # Check if query is completely off-topic (this is context-dependent)
            # For HCI research system, off-topic might be non-research queries
            off_topic_indicators = ["weather", "sports", "cooking recipe"]
            for indicator in off_topic_indicators:
                if indicator.lower() in content.lower() and len(content.split()) < 10:
                    violations.append({
                        "category": category,
                        "reason": f"Query appears off-topic: {indicator}",
                        "severity": "low"
                    })

        return violations

    def _log_safety_event(
        self,
        event_type: str,
        content: str,
        violations: List[Dict[str, Any]],
        is_safe: bool
    ):
        """
        Log a safety event.

        Args:
            event_type: "input" or "output"
            content: The content that was checked
            violations: List of violations found
            is_safe: Whether content passed safety checks
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "safe": is_safe,
            "violations": violations,
            "content_preview": content[:100] + "..." if len(content) > 100 else content
        }

        self.safety_events.append(event)
        self.logger.warning(f"Safety event: {event_type} - safe={is_safe}")

        # Write to safety log file if configured
        log_file = self.config.get("safety_log_file")
        if log_file and self.log_events:
            try:
                with open(log_file, "a") as f:
                    f.write(json.dumps(event) + "\n")
            except Exception as e:
                self.logger.error(f"Failed to write safety log: {e}")

    def get_safety_events(self) -> List[Dict[str, Any]]:
        """Get all logged safety events."""
        return self.safety_events

    def get_safety_stats(self) -> Dict[str, Any]:
        """
        Get statistics about safety events.

        Returns:
            Dictionary with safety statistics
        """
        total = len(self.safety_events)
        input_events = sum(1 for e in self.safety_events if e["type"] == "input")
        output_events = sum(1 for e in self.safety_events if e["type"] == "output")
        violations = sum(1 for e in self.safety_events if not e["safe"])

        return {
            "total_events": total,
            "input_checks": input_events,
            "output_checks": output_events,
            "violations": violations,
            "violation_rate": violations / total if total > 0 else 0
        }

    def clear_events(self):
        """Clear safety event log."""
        self.safety_events = []
