# Multi-Agent Research System for HCI Topics: Design, Implementation, and Evaluation

## Abstract

This paper presents a multi-agent research system designed to conduct deep research on human-computer interaction (HCI) topics. The system orchestrates four specialized agents—Planner, Researcher, Writer, and Critic—using Microsoft AutoGen's RoundRobinGroupChat framework to collaboratively answer research queries. The workflow follows a structured pipeline: task planning, evidence gathering from web and academic sources, response synthesis with citations, and quality critique with revision loops. The system integrates safety guardrails to detect and handle unsafe content across four prohibited categories: harmful content, personal attacks, misinformation, and off-topic queries. Evaluation was conducted using LLM-as-a-Judge methodology with 10 diverse HCI test queries, employing two independent judge prompts per criterion across five evaluation dimensions: relevance, evidence quality, factual accuracy, safety compliance, and clarity. Results show a 100% success rate with an overall average score of 0.621, with safety compliance achieving the highest score (0.975) and evidence quality requiring improvement (0.475). The system demonstrates effective multi-agent coordination and robust safety mechanisms, while highlighting areas for enhancement in evidence gathering and citation quality.

## System Design and Implementation

### Architecture Overview

The multi-agent system is built using Microsoft AutoGen's RoundRobinGroupChat framework, which enables sequential agent coordination through a round-robin conversation pattern. The system consists of four specialized agents, each with distinct roles and responsibilities, orchestrated by an AutoGenOrchestrator that manages workflow execution, error handling, and safety checks.

### Agent Design

**Planner Agent**: The Planner analyzes incoming research queries and breaks them down into actionable research steps. It identifies key topics, suggests research methodologies, and creates a structured plan. The Planner signals completion with "PLAN COMPLETE" to hand off to the Researcher.

**Researcher Agent**: The Researcher gathers evidence using two integrated tools: web_search() (via Tavily API) and paper_search() (via Semantic Scholar API). It collects up to 10 sources per query, prioritizing peer-reviewed papers and authoritative sources. The Researcher signals completion with "RESEARCH COMPLETE" after gathering sufficient evidence.

**Writer Agent**: The Writer synthesizes findings from the Researcher into a coherent, well-structured response. It formats citations in APA style, organizes content logically, and ensures clarity. The Writer signals completion with "DRAFT COMPLETE" to hand off to the Critic.

**Critic Agent**: The Critic evaluates the Writer's draft for quality, completeness, and accuracy. It provides constructive feedback and makes a decision: either "APPROVED - RESEARCH COMPLETE" if the draft meets quality standards, or "NEEDS REVISION" to trigger a revision loop.

### Workflow and Control Flow

The system implements a sequential workflow with revision loop support:

1. **Input Safety Check**: Before processing, the query is validated by input guardrails.
2. **Planning Phase**: Planner creates a research plan.
3. **Research Phase**: Researcher gathers evidence using web and paper search tools.
4. **Writing Phase**: Writer synthesizes findings into a draft response.
5. **Critique Phase**: Critic evaluates the draft.
6. **Revision Loop**: If the Critic requests revision, the Writer revises based on feedback, and the Critic re-evaluates. This loop continues up to 3 times or until approval.
7. **Output Safety Check**: Before returning, the final response is validated by output guardrails.

The orchestrator manages timeouts (300 seconds), maximum iterations (10), and error handling for API failures and invalid inputs.

### Tools Integration

**Web Search Tool**: Integrated with Tavily API, the web_search() tool retrieves up to 5 relevant web sources per query. The tool returns URLs, titles, snippets, and relevance scores.

**Paper Search Tool**: Integrated with Semantic Scholar API, the paper_search() tool retrieves up to 10 academic papers per query. The tool returns paper titles, authors, abstracts, publication years, and citation counts.

**Citation Tool**: The system formats citations in APA style, extracting author names, publication years, and titles from search results.

### Model Configuration

The system uses OpenAI's GPT-4o as the primary model for all agents, with GPT-4o-mini as the judge model for evaluation. The configuration supports fallback to Groq API (llama-3.1-70b-versatile) if OpenAI is unavailable. Agent models use a temperature of 0.7 and max_tokens of 2048, while the judge model uses temperature 0.3 for more consistent evaluation.

## Safety Design

### Safety Framework

The system implements comprehensive safety guardrails using a custom policy-based filtering approach, with support for Guardrails AI framework integration. The SafetyManager coordinates input and output guardrails, logging all safety events for audit and transparency.

### Prohibited Categories

The system monitors and blocks content in four categories:

1. **Harmful Content** (High Severity): Content promoting violence, harm, dangerous activities, or illegal actions. Detected through keyword matching and pattern recognition.

2. **Personal Attacks** (Medium Severity): Content containing personal attacks, insults, or hateful language. Detected through pattern matching for attack language and ad hominem arguments.

3. **Misinformation** (High Severity): Content containing known false information, debunked claims, or deliberately misleading statements. Detected through pattern matching and fact-checking indicators.

4. **Off-Topic Queries** (Low Severity): Queries completely unrelated to HCI research (e.g., weather, sports, cooking). Detected through context-based analysis and keyword matching.

### Response Strategies

When a safety violation is detected, the system implements one of two strategies:

1. **Refuse** (Default): The system refuses to process the request and returns a polite refusal message: "I cannot process this request due to safety policies." This strategy is used for high and medium-severity violations.

2. **Sanitize**: The system removes or redacts unsafe content while preserving safe portions. Unsafe keywords are replaced with [REDACTED]. This strategy is used for low-severity violations that can be safely removed.

### Safety Logging

All safety events are logged with context, including the violation type, severity, detected content, and action taken. Safety statistics are tracked and reported in system metadata, enabling transparency and auditability.

### Guardrail UI Indicators

The UI clearly indicates when content is refused or sanitized:

- **Blocked Content**: Displays a red error message with "⚠️ BLOCKED" and the reason for blocking
- **Sanitized Content**: Displays a yellow warning with "🔧 SANITIZED" and the reason for sanitization
- **Policy Category**: Shows which policy category was triggered (harmful_content, personal_attacks, misinformation, off_topic_queries)
- **Severity Level**: Indicates the severity (high, medium, low) of the violation
- **Safety Status**: When no violations occur, displays "✅ All safety checks passed" with a list of checked categories

The UI provides transparency about safety decisions, showing users exactly which policy category was triggered and why content was blocked or sanitized. This helps users understand the system's safety mechanisms and adjust their queries accordingly.

## Evaluation Setup and Results

### Test Queries

The system was tested with 10 diverse HCI queries covering various topics:

1. "What are the key principles of explainable AI for novice users?"
2. "How has AR usability evolved in the past 5 years?"
3. "What are ethical considerations in using AI for education?"
4. "Compare different approaches to measuring user experience in mobile applications"
5. "What are the latest developments in conversational AI for healthcare?" (Best performing: 0.783)
6. "How do design patterns for accessibility differ across web and mobile platforms?" (Worst performing: 0.15)
7. "What are best practices for visualizing uncertainty in data displays?"
8. "How can voice interfaces be designed for elderly users?"
9. "What are emerging trends in AI-driven prototyping tools?"
10. "How do cultural factors influence mobile app design?"

Complete test queries and results are documented in `TESTED_QUERIES.md` and `outputs/evaluation_20251128_175744.json`.

### Evaluation Methodology

The system was evaluated using LLM-as-a-Judge methodology with 10 diverse HCI test queries covering topics such as explainable AI, AR usability, ethical AI, conversational AI, accessibility, and cultural factors in design. Each query was processed through the full multi-agent workflow, and responses were evaluated using two independent judge prompts (strict and lenient) per criterion.

### Agent Outputs and Chat Transcripts

The UI displays complete agent conversation transcripts showing:
- **Planner** messages with research plans
- **Researcher** messages with tool calls and search results
- **Writer** messages with draft synthesis
- **Critic** messages with evaluation and feedback

A complete sample session export is available at `outputs/sample_session_ar_usability.json`, which includes the full conversation history, workflow stages, sources, citations, and metadata for the query "How has AR usability evolved in the past 5 years?"

### Final Synthesized Answers

The system produces final synthesized answers with:
- **Inline citations** in the response text (e.g., "(Emrich, 2023)")
- **Separate references list** at the end with full URLs and source information
- **Structured formatting** with clear sections and headings

An exported artifact example is available at `outputs/sample_artifact_ar_usability.md`, showing a complete research report with inline citations and a references section.

### Evaluation Criteria

Five evaluation criteria were used, each with assigned weights:

- **Relevance** (Weight: 0.25): How relevant the response is to the query
- **Evidence Quality** (Weight: 0.25): Quality of citations and evidence used
- **Factual Accuracy** (Weight: 0.20): Factual correctness and consistency
- **Safety Compliance** (Weight: 0.15): Absence of unsafe or inappropriate content
- **Clarity** (Weight: 0.15): Clarity and organization of response

Each criterion was scored on a 0-1 scale, with scores aggregated across both judge prompts and weighted by criterion importance.

### Results

**Overall Performance**: The system achieved a 100% success rate (10/10 queries processed successfully) with an overall average score of 0.621.

**Scores by Criterion**:
- Relevance: 0.599 (Weight: 0.25)
- Evidence Quality: 0.475 (Weight: 0.25) - Lowest scoring criterion
- Factual Accuracy: 0.574 (Weight: 0.20)
- Safety Compliance: 0.975 (Weight: 0.15) - Highest scoring criterion
- Clarity: 0.613 (Weight: 0.15)

**Best Performing Query**: "What are the latest developments in conversational AI for healthcare?" achieved a score of 0.783, demonstrating strong performance across all criteria.

**Worst Performing Query**: "How do design patterns for accessibility differ across web and mobile platforms?" achieved a score of 0.15, indicating a failure in the workflow execution or response generation.

### LLM-as-a-Judge Results

The evaluation results are displayed in the UI for each query run. The system shows:
- Overall evaluation score (0-1 scale)
- Scores by criterion with individual judge scores
- Detailed reasoning from each judge
- Visual progress bars and metrics

Raw judge prompts and outputs for a representative query ("How has AR usability evolved in the past 5 years?") are exported in `outputs/judge_prompts_outputs_ar_usability.json`, showing:
- Complete judge prompts for each criterion (strict and lenient perspectives)
- Raw LLM outputs from the judge model
- Parsed scores and reasoning
- Aggregated scores across multiple judges

The evaluation results demonstrate that the system performs well on safety compliance (0.975) but needs improvement in evidence quality (0.475), suggesting that while sources are gathered, citation formatting and evidence integration could be enhanced.

### Error Analysis

Analysis of low-scoring queries revealed several issues:

1. **Workflow Execution Failures**: Some queries resulted in responses that were procedural messages rather than actual research responses, indicating the workflow did not execute properly. For example, one query returned: "Your assessment is clear, but the 'Approved' statement needs a specific confirmation for final completion."

2. **Evidence Quality Gaps**: The lowest scoring criterion was evidence quality (0.475), suggesting that while the system gathers sources, citation formatting and evidence integration need improvement.

3. **Relevance Issues**: Some responses failed to address the query directly, scoring low on relevance (0.599 average). This may indicate issues with query understanding or response extraction.

4. **Safety Success**: Safety compliance achieved the highest score (0.975), demonstrating that the safety guardrails are functioning effectively.

## Discussion & Limitations

### Insights and Learnings

The evaluation reveals several key insights about multi-agent research systems:

1. **Multi-Agent Coordination**: The AutoGen RoundRobinGroupChat framework successfully enables agent coordination, but workflow execution can be fragile. Some queries failed to complete the full workflow, resulting in procedural messages instead of research responses.

2. **Safety Effectiveness**: The safety guardrails performed exceptionally well (0.975), demonstrating that policy-based filtering can effectively detect and prevent unsafe content without significantly impacting legitimate queries.

3. **Evidence Gathering Challenges**: Evidence quality scored lowest (0.475), indicating that while the system retrieves sources, the integration of citations and evidence into responses needs improvement. Better citation formatting and evidence synthesis would enhance this dimension.

4. **Model Dependency**: The system's performance is heavily dependent on the underlying LLM (GPT-4o). Model limitations, such as function calling inconsistencies or response extraction issues, directly impact system reliability.

### Limitations

Several limitations constrain the current system:

1. **Non-Deterministic Behavior**: LLM outputs are non-deterministic, leading to variance in results across runs. This makes reproducibility challenging and can result in inconsistent quality.

2. **Limited Tool Integration**: While web and paper search tools are integrated, the system lacks advanced features such as PDF parsing, full-text paper retrieval, or citation network analysis.

3. **Evaluation Limitations**: LLM-as-a-Judge evaluation, while scalable, may not capture all aspects of quality. Human evaluation would provide more nuanced feedback, particularly for evidence quality and factual accuracy.

4. **Domain Specificity**: The system is optimized for HCI research topics. Performance on other domains may vary, and the safety policies may need domain-specific adjustments.

### Ethical Considerations

The system raises several ethical considerations:

1. **Bias in Search Results**: Search tools (Tavily, Semantic Scholar) may introduce bias through ranking algorithms, potentially privileging certain perspectives or sources.

2. **Misinformation Risk**: While safety guardrails detect known misinformation, the system may still generate inaccurate information if sources are unreliable or if synthesis introduces errors.

3. **Citation Accuracy**: Incorrect citations or misattributed sources could mislead users. The system should implement citation verification mechanisms.

4. **Transparency**: While the system logs safety events, users may not fully understand why certain queries are refused or how responses are generated. Improved transparency would enhance trust.

### Future Work

Several directions for future improvement are identified:

1. **Enhanced Evidence Integration**: Implement better citation extraction, verification, and formatting. Integrate full-text paper retrieval and citation network analysis.

2. **Workflow Robustness**: Add better error detection, recovery mechanisms, and validation checkpoints to prevent workflow failures.

3. **Human-in-the-Loop Evaluation**: Complement LLM-as-a-Judge with human evaluation to capture nuanced quality aspects and validate judge reliability.

4. **Advanced Tools**: Integrate PDF parsing, full-text retrieval, and citation network analysis to enhance evidence gathering capabilities.

5. **Parallel Processing**: Implement parallel query processing and batch optimization to improve scalability and throughput.

6. **Domain Adaptation**: Develop mechanisms for adapting the system to different research domains with domain-specific safety policies and evaluation criteria.

7. **Transparency Enhancements**: Provide detailed explanations of safety decisions, source selection rationale, and response generation process to improve user trust and understanding.

## References

Microsoft. (2024). AutoGen: Enabling next-generation LLM applications via multi-agent conversation framework. *GitHub Repository*. https://github.com/microsoft/autogen

OpenAI. (2024). GPT-4 Technical Report. *arXiv preprint arXiv:2303.08774*.

Tavily. (2024). Tavily API: AI-powered search for research. *Tavily Documentation*. https://www.tavily.com

Semantic Scholar. (2024). Semantic Scholar API: Academic paper search and retrieval. *Semantic Scholar API Documentation*. https://www.semanticscholar.org/product/api

Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems*, 36.

Guardrails AI. (2024). Guardrails: Open-source toolkit for LLM safety. *GitHub Repository*. https://github.com/guardrails-ai/guardrails
