# Json Rejects Backend

> 8 nodes · cohesion 0.25

## Key Concepts

- **parseSourceConfigJson** (5 connections) — `ui/src/features/cameras/forms.test.ts`
- **returns valid JSON templates for every backend option** (3 connections) — `ui/src/features/cameras/forms.test.ts`
- **CAMERA_BACKEND_OPTIONS** (1 connections) — `ui/src/features/cameras/forms.test.ts`
- **defaultSourceConfigForBackend** (1 connections) — `ui/src/features/cameras/forms.test.ts`
- **parses JSON object payloads used for camera create requests** (1 connections) — `ui/src/features/cameras/forms.test.ts`
- **rejects empty config text** (1 connections) — `ui/src/features/cameras/forms.test.ts`
- **rejects JSON arrays because source config must be an object** (1 connections) — `ui/src/features/cameras/forms.test.ts`
- **rejects malformed JSON payloads** (1 connections) — `ui/src/features/cameras/forms.test.ts`

## Relationships

- No strong cross-community connections detected

## Source Files

- `ui/src/features/cameras/forms.test.ts`

## Audit Trail

- EXTRACTED: 14 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*