"""
Output Guardrail
Checks system outputs for safety violations.
"""

from typing import Dict, Any, List
import re
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


class OutputGuardrail:
    """
    Guardrail for checking output safety.

    TODO: YOUR CODE HERE
    - Integrate with Guardrails AI or NeMo Guardrails
    - Check for harmful content in responses
    - Verify factual consistency
    - Detect potential misinformation
    - Remove PII (personal identifiable information)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize output guardrail.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger("safety.output_guardrail")

        # Initialize guardrail framework
        # Note: guardrails-ai 0.6.8+ has a different API structure
        # We use fallback validators that are more reliable
        if GUARDRAILS_AVAILABLE and GUARDRAILS_VALIDATORS_AVAILABLE:
            try:
                # Try to use Guardrails AI if validators are available
                self.guard = Guard()
                self.logger.info("Guardrails AI output guard initialized (basic)")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Guardrails AI: {e}")
                self.guard = None
        else:
            self.guard = None
            if GUARDRAILS_AVAILABLE:
                self.logger.info("Guardrails AI installed but using fallback validators (newer API)")
            else:
                self.logger.info("Guardrails AI not available. Using fallback validation.")

    def validate(self, response: str, sources: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate output response.

        Args:
            response: Generated response to validate
            sources: Optional list of sources used (for fact-checking)

        Returns:
            Validation result

        TODO: YOUR CODE HERE
        - Implement validation logic
        - Check for harmful content
        - Check for PII
        - Verify claims against sources
        - Check for bias
        """
        violations = []
        sanitized_output = response

        # Use Guardrails AI if available
        if self.guard is not None:
            try:
                result = self.guard.validate(response)
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
        pii_violations = self._check_pii(response)
        violations.extend(pii_violations)

        harmful_violations = self._check_harmful_content(response)
        violations.extend(harmful_violations)

        if sources:
            consistency_violations = self._check_factual_consistency(response, sources)
            violations.extend(consistency_violations)

        # Sanitize if violations found
        if violations:
            sanitized_output = self._sanitize(response, violations)

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "sanitized_output": sanitized_output
        }

    def _check_pii(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for personally identifiable information.

        TODO: YOUR CODE HERE Implement comprehensive PII detection
        """
        violations = []

        # Simple regex patterns for common PII
        patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        }

        for pii_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                violations.append({
                    "validator": "pii",
                    "pii_type": pii_type,
                    "reason": f"Contains {pii_type}",
                    "severity": "high",
                    "matches": matches
                })

        return violations

    def _check_harmful_content(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for harmful or inappropriate content.

        TODO: YOUR CODE HERE Implement harmful content detection
        """
        violations = []

        # Placeholder - should use proper toxicity detection
        harmful_keywords = ["violent", "harmful", "dangerous"]
        for keyword in harmful_keywords:
            if keyword in text.lower():
                violations.append({
                    "validator": "harmful_content",
                    "reason": f"May contain harmful content: {keyword}",
                    "severity": "medium"
                })

        return violations

    def _check_factual_consistency(
        self,
        response: str,
        sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Check if response is consistent with sources.

        TODO: YOUR CODE HERE Implement fact-checking logic
        This could use LLM-based verification
        """
        violations = []

        # Placeholder - this is complex and could use LLM
        # to verify claims against sources

        return violations

    def _check_bias(self, text: str) -> List[Dict[str, Any]]:
        """
        Check for biased language.

        TODO: YOUR CODE HERE Implement bias detection
        """
        violations = []
        # Implement bias detection
        return violations

    def _sanitize(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """
        Sanitize text by removing/redacting violations.

        TODO: YOUR CODE HERE Implement sanitization logic
        """
        sanitized = text

        # Redact PII
        for violation in violations:
            if violation.get("validator") == "pii":
                for match in violation.get("matches", []):
                    sanitized = sanitized.replace(match, "[REDACTED]")

        return sanitized
