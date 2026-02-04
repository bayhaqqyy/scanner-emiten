# AGENTS.md — scanner-emiten-main

## GOAL
All modules must work end-to-end with AI-first output:
- scalping (5m)
- swing (1D)
- bsjp
- bpjs
- fundamental

AI is the primary decision maker and must generate long, detailed evaluations.

## UNIVERSAL OUTPUT (AIReport)
All modules must return JSON that matches the universal AIReport schema:
- module, symbol, timeframe, status, data_quality
- ai_decision: score, confidence, setup_type
- plan (optional)
- evaluation_long: thesis(2 paragraphs), why_this(>=7 bullets), risks(>=7 bullets),
  scenarios(>=2), what_to_watch_next(>=7 checklist)
- evidence: derived features used
- generated_at, ai_raw_json_valid, error

## HARD GUARDRAILS
- Never crash on empty/missing data.
- Validate AI output JSON schema. If invalid: do repair prompt once.
- No secrets in logs.
- Add caching (swing TTL 30m, scalping TTL 1-3m).
- Add rate limits / universe limits to avoid yfinance overload.

## DONE CRITERIA
- Smoke tests pass for all endpoints
- Each endpoint returns valid AIReport JSON
- AI output is long & structured (evaluation_long filled)
- Failure cases return status="error" with message
