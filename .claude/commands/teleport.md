# /teleport

Export the current session context so it can be imported into another Claude instance (web ↔ terminal).

## What Gets Exported

1. **Current task description** - What we're working on
2. **Files modified** - List of changed files with brief descriptions
3. **Decisions made** - Key architectural or implementation choices
4. **Blockers/Questions** - Any unresolved issues
5. **Next steps** - What needs to happen next

## Output Format

```markdown
## Teleport Context: [Project Name]
**Timestamp:** 2026-01-07T10:30:00Z
**Branch:** feature/new-generator

### Current Task
Implementing batch processing for healthcare dataset generator.

### Files Modified
- `src/generators/healthcare/batch.py` - New batch processor
- `src/generators/healthcare/config.py` - Added batch_size param
- `tests/test_batch.py` - Unit tests for batch processing

### Decisions Made
- Using chunked processing to handle memory limits
- Batch size default: 10,000 records
- Progress saved to checkpoint file every batch

### Open Questions
- Should we parallelize across CPU cores?
- What's the target memory ceiling?

### Next Steps
1. Add progress bar for long-running batches
2. Implement resume from checkpoint
3. Add integration test with 1M records
```

## Usage

```
/teleport           # generate context export
/teleport --import  # prompt to paste context from another session
```

## Tips

- Run before switching from terminal to web or vice versa
- Include in handoff when assigning to another Claude instance
- Keep context concise - aim for <500 words
