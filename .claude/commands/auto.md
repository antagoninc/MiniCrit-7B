# /auto

Run in autonomous mode. Execute the task fully without asking for confirmation at each step.

## Behavior

1. **Plan first** — Outline all steps before starting
2. **Execute continuously** — Don't stop for confirmation between steps
3. **Self-verify** — Run tests/linters after changes
4. **Fix issues** — If tests fail, fix and retry (max 3 attempts)
5. **Stop only if:**
   - Task complete and tests pass
   - Error you can't resolve after 3 attempts
   - Need clarification (ambiguous spec)
   - Would violate a rule in CLAUDE.md

## Usage
```
/auto Implement the batch processor with full test coverage
/auto Fix all failing tests
/auto Refactor the generator module
```

## Safety Rails

Claude will still stop if:
- About to delete files
- Encounters CLAUDE.md rule violation
- Task is ambiguous
- Changes outside project directory
