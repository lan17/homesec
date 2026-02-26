import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import type { WizardState, WizardStepDef } from './types'

export interface UpdateStepDataOptions {
  persist?: boolean
}

export interface UseWizardStateResult {
  state: WizardState
  goNext: () => void
  goBack: () => void
  skipStep: () => void
  updateStepData: (stepId: string, data: unknown, options?: UpdateStepDataOptions) => void
  markComplete: (stepId: string) => void
  reset: () => void
}

interface WizardPersistedState {
  schemaVersion: number
  currentStep: number
  stepData: Record<string, unknown>
  completedSteps: string[]
  skippedSteps: string[]
}

interface InternalWizardState {
  stepData: Record<string, unknown>
  completedSteps: Set<string>
  skippedSteps: Set<string>
}

const WIZARD_STATE_SCHEMA_VERSION = 1

export const WIZARD_STATE_STORAGE_KEY = 'homesec.setup.wizard'

function clampStepIndex(value: number, maxStepIndex: number): number {
  if (!Number.isFinite(value)) {
    return 0
  }
  if (value < 0) {
    return 0
  }
  if (value > maxStepIndex) {
    return maxStepIndex
  }
  return Math.trunc(value)
}

function parseStepParam(rawStep: string | null, maxStepIndex: number): number | null {
  if (!rawStep) {
    return null
  }
  const parsed = Number.parseInt(rawStep, 10)
  if (Number.isNaN(parsed)) {
    return null
  }
  return clampStepIndex(parsed, maxStepIndex)
}

function toUniqueKnownStepIds(values: unknown, knownStepIds: ReadonlySet<string>): string[] {
  if (!Array.isArray(values)) {
    return []
  }
  const ids = new Set<string>()
  for (const value of values) {
    if (typeof value === 'string' && knownStepIds.has(value)) {
      ids.add(value)
    }
  }
  return [...ids]
}

function toKnownStepData(
  raw: unknown,
  knownStepIds: ReadonlySet<string>,
): Record<string, unknown> {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return {}
  }
  const stepData = raw as Record<string, unknown>
  const result: Record<string, unknown> = {}
  for (const [stepId, value] of Object.entries(stepData)) {
    if (knownStepIds.has(stepId)) {
      result[stepId] = value
    }
  }
  return result
}

function loadPersistedState(knownStepIds: ReadonlySet<string>, maxStepIndex: number): WizardPersistedState | null {
  if (typeof window === 'undefined') {
    return null
  }

  const raw = window.localStorage.getItem(WIZARD_STATE_STORAGE_KEY)
  if (!raw) {
    return null
  }

  try {
    const parsed = JSON.parse(raw) as Partial<WizardPersistedState>
    if (
      !parsed ||
      typeof parsed !== 'object' ||
      parsed.schemaVersion !== WIZARD_STATE_SCHEMA_VERSION
    ) {
      return null
    }

    const currentStep = clampStepIndex(
      typeof parsed.currentStep === 'number' ? parsed.currentStep : 0,
      maxStepIndex,
    )
    const completedSteps = toUniqueKnownStepIds(parsed.completedSteps, knownStepIds)
    const skippedSteps = toUniqueKnownStepIds(parsed.skippedSteps, knownStepIds)
    const stepData = toKnownStepData(parsed.stepData, knownStepIds)

    return {
      schemaVersion: WIZARD_STATE_SCHEMA_VERSION,
      currentStep,
      stepData,
      completedSteps,
      skippedSteps,
    }
  } catch {
    return null
  }
}

function buildPersistedState(
  state: InternalWizardState,
  currentStep: number,
  nonPersistentStepIds: ReadonlySet<string>,
): WizardPersistedState {
  const persistedStepData: Record<string, unknown> = {}
  for (const [stepId, value] of Object.entries(state.stepData)) {
    if (!nonPersistentStepIds.has(stepId)) {
      persistedStepData[stepId] = value
    }
  }

  return {
    schemaVersion: WIZARD_STATE_SCHEMA_VERSION,
    currentStep,
    stepData: persistedStepData,
    completedSteps: [...state.completedSteps],
    skippedSteps: [...state.skippedSteps],
  }
}

function parsedStepParamOrPersisted(
  parsedStepFromUrl: number | null,
  persistedStep: number | undefined,
): number {
  if (parsedStepFromUrl !== null) {
    return parsedStepFromUrl
  }
  if (typeof persistedStep === 'number') {
    return persistedStep
  }
  return 0
}

export function useWizardState(steps: readonly WizardStepDef[]): UseWizardStateResult {
  const [searchParams, setSearchParams] = useSearchParams()
  const maxStepIndex = Math.max(steps.length - 1, 0)
  const stepIds = useMemo(() => steps.map((step) => step.id), [steps])
  const knownStepIds = useMemo(() => new Set(stepIds), [stepIds])
  const persistedState = useMemo(
    () => loadPersistedState(knownStepIds, maxStepIndex),
    [knownStepIds, maxStepIndex],
  )
  const initialStepFromState = parsedStepParamOrPersisted(
    parseStepParam(searchParams.get('step'), maxStepIndex),
    persistedState?.currentStep,
  )
  const [fallbackCurrentStep, setFallbackCurrentStep] = useState(initialStepFromState)
  const nonPersistentStepIdsRef = useRef<Set<string>>(new Set())
  const skipNextPersistenceRef = useRef(false)
  const persistencePausedRef = useRef(false)
  const parsedStepFromUrl = parseStepParam(searchParams.get('step'), maxStepIndex)

  const [state, setState] = useState<InternalWizardState>(() => {
    return {
      stepData: persistedState?.stepData ?? {},
      completedSteps: new Set(persistedState?.completedSteps ?? []),
      skippedSteps: new Set(persistedState?.skippedSteps ?? []),
    }
  })

  const currentStep = parsedStepFromUrl ?? fallbackCurrentStep

  function setStepFromActions(nextStep: number): void {
    const normalizedStep = clampStepIndex(nextStep, maxStepIndex)
    setFallbackCurrentStep(normalizedStep)
    const nextParams = new URLSearchParams(searchParams)
    nextParams.set('step', String(normalizedStep))
    setSearchParams(nextParams)
  }

  useEffect(() => {
    if (parsedStepFromUrl === null) {
      const nextParams = new URLSearchParams(searchParams)
      nextParams.set('step', String(currentStep))
      setSearchParams(nextParams, { replace: true })
    }
  }, [currentStep, parsedStepFromUrl, searchParams, setSearchParams])

  useEffect(() => {
    if (typeof window === 'undefined') {
      return
    }
    if (skipNextPersistenceRef.current) {
      skipNextPersistenceRef.current = false
      return
    }
    if (persistencePausedRef.current) {
      return
    }
    const serialized = buildPersistedState(state, currentStep, nonPersistentStepIdsRef.current)
    window.localStorage.setItem(WIZARD_STATE_STORAGE_KEY, JSON.stringify(serialized))
  }, [currentStep, state])

  function goNext(): void {
    persistencePausedRef.current = false
    setStepFromActions(currentStep + 1)
  }

  function goBack(): void {
    persistencePausedRef.current = false
    setStepFromActions(currentStep - 1)
  }

  function skipStep(): void {
    const activeStep = steps[currentStep]
    if (!activeStep || !activeStep.skippable) {
      return
    }
    setState((current) => {
      const skippedSteps = new Set(current.skippedSteps)
      skippedSteps.add(activeStep.id)
      return {
        ...current,
        skippedSteps,
      }
    })
    persistencePausedRef.current = false
    setStepFromActions(currentStep + 1)
  }

  const updateStepData = useCallback(
    (stepId: string, data: unknown, options?: UpdateStepDataOptions) => {
      if (!knownStepIds.has(stepId)) {
        return
      }
      if (options?.persist === false) {
        nonPersistentStepIdsRef.current.add(stepId)
      } else {
        nonPersistentStepIdsRef.current.delete(stepId)
      }
      persistencePausedRef.current = false

      setState((current) => ({
        ...current,
        stepData: {
          ...current.stepData,
          [stepId]: data,
        },
      }))
    },
    [knownStepIds],
  )

  const markComplete = useCallback(
    (stepId: string) => {
      if (!knownStepIds.has(stepId)) {
        return
      }
      persistencePausedRef.current = false

      setState((current) => {
        const completedSteps = new Set(current.completedSteps)
        completedSteps.add(stepId)
        const skippedSteps = new Set(current.skippedSteps)
        skippedSteps.delete(stepId)
        return {
          ...current,
          completedSteps,
          skippedSteps,
        }
      })
    },
    [knownStepIds],
  )

  const reset = useCallback(() => {
    nonPersistentStepIdsRef.current.clear()
    skipNextPersistenceRef.current = true
    persistencePausedRef.current = true
    if (typeof window !== 'undefined') {
      window.localStorage.removeItem(WIZARD_STATE_STORAGE_KEY)
    }
    setFallbackCurrentStep(0)
    setState({
      stepData: {},
      completedSteps: new Set<string>(),
      skippedSteps: new Set<string>(),
    })
    const nextParams = new URLSearchParams(searchParams)
    nextParams.set('step', '0')
    setSearchParams(nextParams, { replace: true })
  }, [searchParams, setSearchParams])

  return {
    state: {
      currentStep,
      stepData: state.stepData,
      completedSteps: state.completedSteps,
      skippedSteps: state.skippedSteps,
    },
    goNext,
    goBack,
    skipStep,
    updateStepData,
    markComplete,
    reset,
  }
}
