# debugging-and-error-recovery

## Description
Structured debugging: reproduce, isolate, fix, guard.

## Process
1. **Reproduce** — Get a reliable way to trigger the error
2. **Localize** — Find the exact root cause (not just the symptom)
3. **Fix** — Apply the minimal change that resolves it
4. **Guard** — Add test/assertion to prevent regression

## Anti-Rationalization
- "I know what the bug is, let me just fix it" — Without reproduction, you can't verify the fix.
- "This error is random / intermittent" — Find the pattern, then fix.
- "Just wrap it in try/except" — Swallowing errors hides bugs.

## Verification
- Bug is fixed AND regression test is added. Error no longer reproducible.
