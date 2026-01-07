# /verify

Run end-to-end verification of the application. Test the UI, APIs, and core functionality to ensure everything works before shipping.

## Verification Checklist

### 1. Unit Tests
```bash
pytest tests/unit/ -v --tb=short
```
- All tests must pass
- Report any failures with context

### 2. Integration Tests
```bash
pytest tests/integration/ -v --tb=short
```
- Test API endpoints
- Test database operations
- Test external service integrations (mocked)

### 3. Type Checking
```bash
mypy src/ --strict
```
- No type errors allowed
- Report any issues

### 4. Linting
```bash
ruff check src/
black --check src/
```
- Must pass all lint rules
- Auto-fix if requested

### 5. Security Scan
```bash
bandit -r src/ -ll
safety check
```
- No high-severity issues
- Report medium issues for review

### 6. Manual Verification (if UI exists)
- Start the application
- Test critical user flows
- Verify error handling
- Check loading states

## Output

Provide a verification report:

```
✅ Unit Tests: 47/47 passed
✅ Integration Tests: 12/12 passed
✅ Type Check: No errors
✅ Lint: Clean
⚠️ Security: 1 medium issue (SQL in logging.py:45)
✅ Manual: All flows working

VERDICT: Ready to ship (address security warning in next PR)
```

## Example Usage

```
/verify              # run full verification
/verify --quick      # skip integration tests
/verify --fix        # auto-fix lint issues
```
