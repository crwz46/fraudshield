# test-driven-development

## Description
Red-Green-Refactor: write failing test first, make it pass, then clean up.

## Process
1. **Red** — Write a failing test that defines the expected behavior
2. **Green** — Write minimal code to make the test pass (no extra features)
3. **Refactor** — Clean up code while keeping tests green
4. **Commit** — Only after all tests pass

## Anti-Rationalization
- "I'll add tests after it works" → No. Tests drive the design, not the other way.
- "This is too simple to test" → If it has logic, it needs a test.
- "The test coverage is already good" → Every new behavior needs its own test.

## Verification
- All tests pass before any commit
