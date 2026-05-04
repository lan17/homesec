# Step Persisted Wizard

> 14 nodes · cohesion 0.15

## Key Concepts

- **renderWizard** (10 connections) — `ui/src/features/setup/useWizardState.test.tsx`
- **createWrapper** (2 connections) — `ui/src/features/setup/useWizardState.test.tsx`
- **persists and restores state across hook instances** (2 connections) — `ui/src/features/setup/useWizardState.test.tsx`
- **reset clears state and removes persisted wizard storage** (2 connections) — `ui/src/features/setup/useWizardState.test.tsx`
- **STEPS** (2 connections) — `ui/src/features/setup/useWizardState.test.tsx`
- **WIZARD_STATE_STORAGE_KEY** (2 connections) — `ui/src/features/setup/useWizardState.test.tsx`
- **goBack at step zero is a no-op** (1 connections) — `ui/src/features/setup/useWizardState.test.tsx`
- **goNext and goBack update step index and URL search param** (1 connections) — `ui/src/features/setup/useWizardState.test.tsx`
- **goToStep navigates directly to a known step id** (1 connections) — `ui/src/features/setup/useWizardState.test.tsx`
- **PropsWithChildren** (1 connections) — `ui/src/features/setup/useWizardState.test.tsx`
- **skipStep advances and marks current step as skipped without completion** (1 connections) — `ui/src/features/setup/useWizardState.test.tsx`
- **starts at step zero when no URL step and no persisted state exist** (1 connections) — `ui/src/features/setup/useWizardState.test.tsx`
- **useWizardState** (1 connections) — `ui/src/features/setup/useWizardState.test.tsx`
- **WizardStepDef** (1 connections) — `ui/src/features/setup/useWizardState.test.tsx`

## Relationships

- No strong cross-community connections detected

## Source Files

- `ui/src/features/setup/useWizardState.test.tsx`

## Audit Trail

- EXTRACTED: 28 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*