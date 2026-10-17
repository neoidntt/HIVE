# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes and enforce scientific rigor. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution, reproducibility, and scientific accuracy over speed. 

## 1. Think Before Coding
**Don't assume. Don't hide confusion. Surface tradeoffs.**
- State your assumptions explicitly. If uncertain about the biological context or data structure, ask.
- If multiple interpretations exist (e.g., handling missing time-series data), present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First
**Minimum code that solves the problem. Nothing speculative.**
- No features beyond what was asked.
- No abstractions for single-use analytical scripts.
- No "flexibility" or "configurability" that wasn't requested.
- If you write 200 lines and it could be 50, rewrite it. 
- Ask yourself: "Would a senior PI or bioinformatician say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes
**Touch only what you must. Clean up only your own mess.**
- Don't "improve" adjacent code, comments, or formatting.
- Match existing style, even if you'd do it differently.
- Remove imports/variables/functions that YOUR changes made unused.
- The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution
**Define success criteria. Loop until verified.**
- "Analyze rhythmicity" → "Implement established time-series method, test with mock circadian data, verify output."
- For multi-step tasks, state a brief plan:
  1. [Step] → verify: [check]
  2. [Step] → verify: [check]
- Strong success criteria let you loop independently. Weak criteria require constant clarification.

## 5. Scientific Rigor & Ethical Research Practices
**Ensure reproducibility and data integrity at all times.**
- **Never mutate raw data:** All preprocessing, detrending, or filtering must generate new variables or files. Raw data must remain untouched.
- **Reproducibility:** Always set explicit random seeds for stochastic processes (e.g., permutations, machine learning models). 
- **Methodological Transparency:** Comment your code with the rationale for specific statistical choices (e.g., "Using Lomb-Scargle here due to uneven sampling intervals").
- **Literature Grounding:** Rely strictly on established, peer-reviewed computational methods for circadian analysis. Do not invent proprietary algorithms or hallucinate statistical methods.

## 6. Anti-Rabbit Hole & Confidence Checks
**Fail fast. Verify ruthlessly.**
- **Timeboxing:** If you find yourself applying a third sequential fix to the same block of code without success, STOP. Revert to the last working state, summarize the failure loop, and ask for user input.
- **Sanity Checks:** Before finalizing data transformations, implement basic assertions (e.g., ensuring time vectors are monotonically increasing, checking that phase outputs fall within expected boundaries).
- **Self-Correction:** After writing a complex function, read it back line-by-line against the original prompt before confirming completion.