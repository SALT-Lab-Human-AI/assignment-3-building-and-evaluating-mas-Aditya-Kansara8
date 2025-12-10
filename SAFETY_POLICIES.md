# Safety Policies and Guardrails

This document details the safety guardrail policies implemented in the Multi-Agent Research System, including prohibited categories and response strategies.

## Overview

The system implements comprehensive safety guardrails to detect and handle unsafe or inappropriate content at both input and output stages. The safety system uses a combination of Guardrails AI framework (when available) and custom policy-based filtering to ensure content safety.

## Prohibited Categories

The system monitors and blocks content in the following four categories:

### 1. Harmful Content
**Description**: Content that promotes violence, harm, dangerous activities, or illegal actions.

**Detection Methods**:
- Keyword-based detection for terms like: "violence", "harmful", "dangerous", "illegal"
- Pattern matching for explicit harmful language
- Severity: **High**

**Examples of Blocked Content**:
- Queries requesting information on how to cause harm
- Responses containing instructions for dangerous activities
- Content promoting illegal actions

### 2. Personal Attacks
**Description**: Content that contains personal attacks, insults, or hateful language directed at individuals or groups.

**Detection Methods**:
- Pattern matching for attack language: "you are", "you're", "stupid", "idiot", "hate"
- Detection of ad hominem arguments
- Severity: **Medium**

**Examples of Blocked Content**:
- Insulting language directed at users
- Personal attacks in generated responses
- Hate speech or discriminatory content

### 3. Misinformation
**Description**: Content that contains known false information, debunked claims, or deliberately misleading statements.

**Detection Methods**:
- Pattern matching for misinformation indicators: "definitely false", "proven wrong", "debunked"
- Fact-checking integration (when available)
- Severity: **High**

**Examples of Blocked Content**:
- Responses containing debunked scientific claims
- Deliberately false information
- Conspiracy theories or unsubstantiated claims

### 4. Off-Topic Queries
**Description**: Queries that are completely unrelated to the system's purpose (HCI research).

**Detection Methods**:
- Context-based detection for non-research queries
- Pattern matching for off-topic indicators: "weather", "sports", "cooking recipe"
- Length-based heuristics (very short queries with off-topic keywords)
- Severity: **Low**

**Examples of Blocked Content**:
- Weather queries
- Sports scores
- Cooking recipes
- Other non-research related queries

## Response Strategies

When a safety violation is detected, the system implements one of the following response strategies based on the violation type and severity:

### 1. Refuse (Default Strategy)
**Action**: The system refuses to process the request and returns a polite refusal message.

**When Used**:
- High-severity violations (harmful content, misinformation)
- Medium-severity violations (personal attacks)
- When the violation cannot be safely sanitized

**Response Message**:
```
"I cannot process this request due to safety policies."
```

**Implementation**:
- Input guardrails: Query is blocked before processing
- Output guardrails: Response is not returned to the user

### 2. Sanitize
**Action**: The system removes or redacts unsafe content while preserving safe portions of the response.

**When Used**:
- Low-severity violations that can be safely removed
- Partial violations where most content is safe
- When specific unsafe phrases can be identified and removed

**Implementation**:
- Unsafe keywords are replaced with `[REDACTED]`
- Violation-specific content is removed
- Safe portions of the response are preserved

**Example**:
```
Original: "This method is definitely false and proven wrong."
Sanitized: "This method is [REDACTED] and [REDACTED]."
```

### 3. Redirect (Future Implementation)
**Action**: The system redirects the user to a safe alternative or suggests a related safe query.

**When Used**:
- Off-topic queries that could be reformulated
- Queries that are close to acceptable but need refinement

**Status**: Currently configured but not fully implemented in the current version.

## Safety Event Logging

All safety events are logged with the following information:

- **Timestamp**: When the event occurred
- **Event Type**: "input" or "output"
- **Safe Status**: Whether content passed safety checks
- **Violations**: List of detected violations with categories and reasons
- **Content Preview**: First 100 characters of the checked content

### Log Locations

1. **In-Memory Log**: Stored in `SafetyManager.safety_events` during runtime
2. **File Log**: Written to `logs/safety_events.log` (configured in `config.yaml`)
3. **User Interface**: Displayed in both CLI and web interfaces when safety events occur

### Log Format

```json
{
  "timestamp": "2024-01-15T10:30:45.123456",
  "type": "input",
  "safe": false,
  "violations": [
    {
      "category": "harmful_content",
      "reason": "Contains potentially harmful content: violence",
      "severity": "high"
    }
  ],
  "content_preview": "How to cause violence..."
}
```

## Configuration

Safety policies are configured in `config.yaml`:

```yaml
safety:
  enabled: true
  framework: "guardrails"  # or "nemo_guardrails"
  log_events: true

  prohibited_categories:
    - "harmful_content"
    - "personal_attacks"
    - "misinformation"
    - "off_topic_queries"

  on_violation:
    action: "refuse"  # or "sanitize" or "redirect"
    message: "I cannot process this request due to safety policies."
```

## Integration Points

### Input Guardrails
- **Location**: `src/guardrails/input_guardrail.py`
- **When Applied**: Before query processing begins
- **Purpose**: Prevent unsafe queries from entering the system

### Output Guardrails
- **Location**: `src/guardrails/output_guardrail.py`
- **When Applied**: Before returning final response to user
- **Purpose**: Ensure generated responses are safe

### Safety Manager
- **Location**: `src/guardrails/safety_manager.py`
- **Purpose**: Coordinates input and output guardrails, manages logging, and implements response strategies

## User Interface Integration

Safety events are communicated to users through:

1. **CLI Interface**: Safety events are displayed with clear indicators:
   - `⚠️ BLOCKED`: Content was refused
   - `🔧 SANITIZED`: Content was sanitized
   - `ℹ️ INFO`: Other safety events

2. **Web Interface (Streamlit)**:
   - Safety events appear in an expandable section
   - Color-coded indicators (error for blocked, warning for sanitized)
   - Safety log viewable in sidebar

## Limitations and Future Work

### Current Limitations

1. **Keyword-Based Detection**: Current implementation relies primarily on keyword matching, which may have false positives/negatives
2. **Limited Fact-Checking**: Misinformation detection is basic and could be enhanced with fact-checking APIs
3. **Context Sensitivity**: Off-topic detection may incorrectly flag legitimate research queries
4. **No Learning**: System does not learn from false positives/negatives

### Future Improvements

1. **Enhanced ML Models**: Integrate more sophisticated content classification models
2. **Fact-Checking Integration**: Connect to fact-checking APIs for better misinformation detection
3. **Context-Aware Detection**: Improve off-topic detection with better understanding of research context
4. **User Feedback Loop**: Allow users to report false positives/negatives for system improvement
5. **Customizable Policies**: Allow users to configure sensitivity levels for different categories

## References

- Guardrails AI Documentation: https://docs.guardrailsai.com/
- NeMo Guardrails: https://docs.nvidia.com/nemo/guardrails/
- Safety Best Practices: Based on industry standards for AI safety and content moderation
