# AGENTS.md — Engineering Rules

## Core Principles
- **Spec before code**: Never implement without a written spec for new features/projects
- **Tests are proof**: Every change must have passing tests before commit
- **One slice at a time**: Implement in thin vertical slices, commit after each
- **Verify before ship**: Run lint + typecheck + tests before declaring done
- **Seems right is never enough**: Every claim needs evidence (test output, build log, runtime data)

## Skill Auto-Discovery
The agent detects intent and loads the matching skill automatically:

| Intent | Skill |
|--------|-------|
| New feature / project | spec-driven-development |
| Writing / fixing code | test-driven-development |
| Reviewing a PR | code-review-and-quality |
| Handling auth / secrets / input | security-and-hardening |
| Debugging a failure | debugging-and-error-recovery |

## Anti-Rationalization
Common excuses the agent must NOT accept:
- "I'll add tests later" → No. Tests ship with the code.
- "This is too small for a spec" → If it touches more than 1 file, write a spec.
- "It works on my machine" → Run the actual tests.
- "We can refactor later" → No. Clean as you go.

## Verification Gates
Every task must pass: `pytest` + lint check (if applicable). No exceptions.
