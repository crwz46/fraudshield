# code-review-and-quality

## Description
Structured code review before merging: correctness, design, readability, security, testing.

## Process
1. **Correctness** — Does the code do what it's supposed to?
2. **Design** — Is it well-structured? Follows project conventions?
3. **Readability** — Can another engineer understand it in 30 seconds?
4. **Security** — Any injection, auth, or data exposure issues?
5. **Testing** — Are there tests for the new behavior? Do they cover edge cases?

## Anti-Rationalization
- "I'll fix it in the next PR" → Fix it now or write a ticket.
- "It follows the same pattern as the existing code" → The existing pattern might be wrong.
- "The tests pass, good enough" → Tests only cover what you thought to test.

## Verification
- All 5 axes checked. No unresolved comments.
