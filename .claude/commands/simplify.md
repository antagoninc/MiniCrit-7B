# /simplify

Act as a code simplification specialist. Review the specified file or recent changes and simplify the architecture.

## Objectives

1. **Reduce complexity**
   - Flatten nested conditionals
   - Extract repeated logic into functions
   - Remove dead code and unused imports
   - Simplify overly clever one-liners into readable code

2. **Improve readability**
   - Add clarifying comments only where logic is non-obvious
   - Use meaningful variable names
   - Break long functions into smaller, focused ones
   - Ensure consistent formatting

3. **Maintain correctness**
   - Do NOT change behavior
   - Run existing tests after changes
   - If no tests exist, add basic sanity checks

## Process

1. Analyze the target file(s)
2. Identify top 3-5 simplification opportunities
3. Present proposed changes with rationale
4. Apply changes incrementally
5. Run tests after each significant change
6. Report final diff summary

## Example Usage

```
/simplify src/generators/healthcare.py
/simplify --recent  # simplify files changed in last commit
/simplify --all     # scan entire src/ for opportunities
```

## Guidelines

- Prefer boring, readable code over clever code
- Functions should do one thing
- Max function length: ~30 lines (guideline, not rule)
- If a comment explains "what" instead of "why", the code needs simplification
