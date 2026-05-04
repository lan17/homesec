# Cursor Filter Signature

> 8 nodes · cohesion 0.43

## Key Concepts

- **drops stale cursor history when filter signature changes** (4 connections) — `ui/src/features/clips/cursorState.test.ts`
- **historyForSignature** (4 connections) — `ui/src/features/clips/cursorState.test.ts`
- **rewinds previous cursor and keeps filters intact** (4 connections) — `ui/src/features/clips/cursorState.test.ts`
- **keeps filter signature scoped while moving across keyset pages** (3 connections) — `ui/src/features/clips/cursorState.test.ts`
- **pushCursorHistory** (3 connections) — `ui/src/features/clips/cursorState.test.ts`
- **resetCursorHistory** (3 connections) — `ui/src/features/clips/cursorState.test.ts`
- **popCursorHistory** (2 connections) — `ui/src/features/clips/cursorState.test.ts`
- **historyForSignature** (1 connections) — `ui/src/features/clips/ClipsPage.tsx`

## Relationships

- No strong cross-community connections detected

## Source Files

- `ui/src/features/clips/ClipsPage.tsx`
- `ui/src/features/clips/cursorState.test.ts`

## Audit Trail

- EXTRACTED: 22 (92%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 2 (8%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*