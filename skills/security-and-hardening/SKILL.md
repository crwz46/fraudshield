# security-and-hardening

## Description
Security review: input validation, auth, secrets, data exposure.

## Process
1. **Input validation** — All user inputs validated (type, range, injection)
2. **Auth** — Endpoints properly secured; tokens expire; no hardcoded creds
3. **Secrets** — No secrets in code; use env vars / .env
4. **Data exposure** — No leaking PII in logs, errors, or responses
5. **Dependencies** — Known vulnerabilities checked

## Anti-Rationalization
- "It's just a demo/portfolio project" → Bad habits carry over to real work.
- "Nobody will find this endpoint" → Security by obscurity is not security.
- "I'll add auth later" → Ship with auth from day one.

## Verification
- No secrets in code. Input validated at boundary. Auth enforced on every protected endpoint.
