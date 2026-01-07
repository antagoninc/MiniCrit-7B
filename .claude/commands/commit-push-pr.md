# /commit-push-pr

Commit all staged changes, push to remote, and create a pull request.

## Steps

1. Check for staged changes with `git status`
2. If no staged changes, stage all modified files with `git add -A`
3. Generate a concise commit message based on the diff:
   - First line: imperative mood, max 72 chars
   - If needed, add blank line then bullet points for details
4. Commit with the generated message
5. Push to the current branch (create upstream if needed)
6. Create a pull request using `gh pr create`:
   - Title: same as commit first line
   - Body: summary of changes with any relevant context
   - Set as draft if tests haven't run yet

## Example Usage

```
/commit-push-pr
```

## Notes

- If there are merge conflicts, stop and report them
- If push fails, check if branch is behind and suggest rebase
- Always verify the PR was created successfully and provide the URL
