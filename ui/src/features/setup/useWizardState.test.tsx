// @vitest-environment happy-dom

import type { PropsWithChildren } from 'react'
import { act, renderHook, waitFor } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { beforeEach, describe, expect, it } from 'vitest'

import type { WizardStepDef } from './types'
import { WIZARD_STATE_STORAGE_KEY, useWizardState } from './useWizardState'

const STEPS: readonly WizardStepDef[] = [
  { id: 'welcome', title: 'Welcome', skippable: true },
  { id: 'camera', title: 'Camera', skippable: false },
  { id: 'review', title: 'Review', skippable: false },
]

function createWrapper(initialEntry = '/setup') {
  return function Wrapper({ children }: PropsWithChildren) {
    return <MemoryRouter initialEntries={[initialEntry]}>{children}</MemoryRouter>
  }
}

function renderWizard(initialEntry = '/setup') {
  return renderHook(
    () => {
      const wizard = useWizardState(STEPS)
      const location = useLocation()
      return { wizard, location }
    },
    { wrapper: createWrapper(initialEntry) },
  )
}

describe('useWizardState', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  it('starts at step zero when no URL step and no persisted state exist', async () => {
    // Given: No persisted wizard state and setup route without query params
    const { result } = renderWizard('/setup')

    // When: Wizard hook initializes
    await waitFor(() => {
      // Then: Step index is zero and URL search is synchronized
      expect(result.current.wizard.state.currentStep).toBe(0)
      expect(result.current.location.search).toBe('?step=0')
    })
  })

  it('goNext and goBack update step index and URL search param', async () => {
    // Given: A fresh wizard session at step 0
    const { result } = renderWizard('/setup')

    // When: Advancing then going back one step
    act(() => {
      result.current.wizard.goNext()
    })
    await waitFor(() => {
      expect(result.current.wizard.state.currentStep).toBe(1)
      expect(result.current.location.search).toBe('?step=1')
    })

    act(() => {
      result.current.wizard.goBack()
    })

    // Then: State and URL return to step 0
    await waitFor(() => {
      expect(result.current.wizard.state.currentStep).toBe(0)
      expect(result.current.location.search).toBe('?step=0')
    })
  })

  it('goBack at step zero is a no-op', async () => {
    // Given: A fresh wizard session at first step
    const { result } = renderWizard('/setup')

    // When: Back navigation is requested at step 0
    act(() => {
      result.current.wizard.goBack()
    })

    // Then: Current step remains unchanged
    await waitFor(() => {
      expect(result.current.wizard.state.currentStep).toBe(0)
      expect(result.current.location.search).toBe('?step=0')
    })
  })

  it('skipStep advances and marks current step as skipped without completion', async () => {
    // Given: Wizard at skippable first step
    const { result } = renderWizard('/setup')

    // When: Skipping the current step
    act(() => {
      result.current.wizard.skipStep()
    })

    // Then: Step advances and skipped set includes previous step id
    await waitFor(() => {
      expect(result.current.wizard.state.currentStep).toBe(1)
      expect(result.current.wizard.state.skippedSteps.has('welcome')).toBe(true)
      expect(result.current.wizard.state.completedSteps.has('welcome')).toBe(false)
    })
  })

  it('persists and restores state across hook instances', async () => {
    // Given: First hook instance writes wizard state to localStorage
    const first = renderWizard('/setup')
    act(() => {
      first.result.current.wizard.updateStepData('welcome', { note: 'Initial setup notes' })
      first.result.current.wizard.markComplete('welcome')
      first.result.current.wizard.goNext()
    })

    await waitFor(() => {
      const raw = window.localStorage.getItem(WIZARD_STATE_STORAGE_KEY)
      expect(raw).toBeTruthy()
      expect(first.result.current.wizard.state.currentStep).toBe(1)
    })
    first.unmount()

    // When: New hook instance mounts with same storage key
    const second = renderWizard('/setup')

    // Then: Wizard resumes from persisted step and data
    await waitFor(() => {
      expect(second.result.current.wizard.state.currentStep).toBe(1)
      expect(
        second.result.current.wizard.state.stepData['welcome'] as { note: string },
      ).toEqual({
        note: 'Initial setup notes',
      })
      expect(second.result.current.wizard.state.completedSteps.has('welcome')).toBe(true)
    })
  })

  it('reset clears state and removes persisted wizard storage', async () => {
    // Given: Wizard state has progressed and persisted values
    const { result } = renderWizard('/setup')
    act(() => {
      result.current.wizard.goNext()
      result.current.wizard.updateStepData('camera', { note: 'camera configured' })
      result.current.wizard.markComplete('camera')
    })
    await waitFor(() => {
      expect(window.localStorage.getItem(WIZARD_STATE_STORAGE_KEY)).toBeTruthy()
    })

    // When: Reset is invoked
    act(() => {
      result.current.wizard.reset()
    })

    // Then: Wizard state returns to defaults and storage is cleared
    await waitFor(() => {
      expect(result.current.wizard.state.currentStep).toBe(0)
      expect(Object.keys(result.current.wizard.state.stepData)).toHaveLength(0)
      expect(result.current.wizard.state.completedSteps.size).toBe(0)
      expect(result.current.wizard.state.skippedSteps.size).toBe(0)
      expect(result.current.location.search).toBe('?step=0')
      expect(window.localStorage.getItem(WIZARD_STATE_STORAGE_KEY)).toBeNull()
    })
  })
})
