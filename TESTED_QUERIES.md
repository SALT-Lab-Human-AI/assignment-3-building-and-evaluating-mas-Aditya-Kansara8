# Tested Queries

This document lists all queries that have been tested with the multi-agent research system.

## Evaluation Queries (10 queries tested on November 28, 2025)

The following 10 queries were used for the comprehensive evaluation reported in `Final_Report.md`:

1. **"What are the key principles of explainable AI for novice users?"**
   - Category: explainable_ai
   - Expected topics: transparency, interpretability, user understanding, trust
   - Evaluation score: 0.27

2. **"How has AR usability evolved in the past 5 years?"**
   - Category: ar_usability
   - Expected topics: interaction techniques, user experience, hardware improvements, application domains
   - Evaluation score: See evaluation results

3. **"What are ethical considerations in using AI for education?"**
   - Category: ai_ethics
   - Expected topics: bias, privacy, accessibility, transparency, student autonomy
   - Evaluation score: See evaluation results

4. **"Compare different approaches to measuring user experience in mobile applications"**
   - Category: ux_measurement
   - Expected topics: questionnaires, analytics, user testing, physiological measures
   - Evaluation score: See evaluation results

5. **"What are the latest developments in conversational AI for healthcare?"**
   - Category: conversational_ai
   - Expected topics: chatbots, patient interaction, clinical decision support, privacy concerns
   - Evaluation score: 0.783 (Best performing query)

6. **"How do design patterns for accessibility differ across web and mobile platforms?"**
   - Category: accessibility
   - Expected topics: screen readers, touch interfaces, WCAG guidelines, responsive design
   - Evaluation score: 0.15 (Worst performing query)

7. **"What are best practices for visualizing uncertainty in data displays?"**
   - Category: data_visualization
   - Expected topics: visual encoding, uncertainty representation, user comprehension, design guidelines
   - Evaluation score: See evaluation results

8. **"How can voice interfaces be designed for elderly users?"**
   - Category: voice_interfaces
   - Expected topics: age-related challenges, speech recognition, error handling, user preferences
   - Evaluation score: See evaluation results

9. **"What are emerging trends in AI-driven prototyping tools?"**
   - Category: ai_prototyping
   - Expected topics: generative design, automation, designer collaboration, tool capabilities
   - Evaluation score: See evaluation results

10. **"How do cultural factors influence mobile app design?"**
    - Category: cross_cultural_design
    - Expected topics: localization, cultural dimensions, user preferences, design adaptations
    - Evaluation score: See evaluation results

## Additional Test Queries

The following queries have also been tested during development and demonstration:

- **"Recent Advances in AR Usability Research"** (Demo query - see demo screenshots)
- **"What are the key principles of user-centered design?"**
- **"Explain recent advances in AR usability research"**
- **"Compare different approaches to AI transparency"**

## Evaluation Results Summary

- **Total Queries Tested**: 10 (evaluation) + additional demo queries
- **Success Rate**: 100% (10/10 queries processed successfully)
- **Overall Average Score**: 0.621
- **Best Score**: 0.783 ("What are the latest developments in conversational AI for healthcare?")
- **Worst Score**: 0.15 ("How do design patterns for accessibility differ across web and mobile platforms?")

## Detailed Results

For detailed evaluation results, see:
- `outputs/evaluation_20251128_175744.json` - Complete evaluation results with all scores and reasoning
- `outputs/evaluation_summary_20251128_175744.txt` - Human-readable summary
- `Final_Report.md` - Comprehensive analysis and discussion

## Sample Session Data

A complete session export for the query "Recent Advances in Usability Research" is available at:
- `outputs/sample_session_recent_advances_usability.json` - Full conversation history and metadata
