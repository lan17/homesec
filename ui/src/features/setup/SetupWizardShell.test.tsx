// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'

import type { WizardStepDef } from './types'
import { SetupWizardShell } from './SetupWizardShell'

const STEPS: readonly WizardStepDef[] = [
  { id: 'welcome', title: 'Welcome', skippable: true },
  { id: 'camera', title: 'Camera', skippable: true },
  { id: 'review', title: 'Review', skippable: false },
]

function renderShell({
  currentStep = 0,
  completedSteps = new Set<string>(),
  skippedSteps = new Set<string>(),
  canGoNext = true,
  onNext = vi.fn(),
  onBack = vi.fn(),
  onSkip = vi.fn(),
}: {
  currentStep?: number
  completedSteps?: ReadonlySet<string>
  skippedSteps?: ReadonlySet<string>
  canGoNext?: boolean
  onNext?: () => void
  onBack?: () => void
  onSkip?: () => void
} = {}) {
  render(
    <MemoryRouter>
      <SetupWizardShell
        steps={STEPS}
        currentStep={currentStep}
        completedSteps={completedSteps}
        skippedSteps={skippedSteps}
        canGoNext={canGoNext}
        onNext={onNext}
        onBack={onBack}
        onSkip={onSkip}
      >
        <p>Step content</p>
      </SetupWizardShell>
    </MemoryRouter>,
  )
  return { onNext, onBack, onSkip }
}

describe('SetupWizardShell', () => {
  afterEach(() => {
    cleanup()
  })

  it('renders progress indicator and marks the active step', () => {
    // Given: Wizard at camera step with welcome already complete
    renderShell({
      currentStep: 1,
      completedSteps: new Set(['welcome']),
      skippedSteps: new Set(),
    })

    // When: Progress indicator is rendered
    const progress = screen.getByRole('list', { name: 'Setup steps' })
    const activeStep = within(progress).getByText('Camera').closest('li')
    const completeStep = within(progress).getByText('Welcome').closest('li')

    // Then: Active and completed statuses are reflected via CSS classes
    expect(activeStep?.className).toContain('wizard__progress-step--active')
    expect(completeStep?.className).toContain('wizard__progress-step--complete')
  })

  it('disables Back on the first step', () => {
    // Given: Wizard at the first step
    renderShell({ currentStep: 0 })

    // When: Back button is rendered
    const backButton = screen.getByRole('button', { name: 'Back' })

    // Then: Back is disabled to prevent negative step navigation
    expect((backButton as HTMLButtonElement).disabled).toBe(true)
  })

  it('hides Skip button for non-skippable steps', () => {
    // Given: Wizard at non-skippable review step
    renderShell({ currentStep: 2 })

    // When: Footer controls are rendered
    const skipButton = screen.queryByRole('button', { name: 'Skip' })

    // Then: Skip is not available on required steps
    expect(skipButton).toBeNull()
  })

  it('routes Exit to Dashboard link to root path', () => {
    // Given: Wizard shell is mounted in router context
    renderShell()

    // When: Reading the exit link
    const exitLink = screen.getByRole('link', { name: 'Exit to Dashboard' })

    // Then: Link points to root dashboard route
    expect(exitLink.getAttribute('href')).toBe('/')
  })

  it('invokes back, skip, and next callbacks from footer actions', async () => {
    // Given: Wizard with actionable footer controls
    const onNext = vi.fn()
    const onBack = vi.fn()
    const onSkip = vi.fn()
    renderShell({
      currentStep: 1,
      onNext,
      onBack,
      onSkip,
    })
    const user = userEvent.setup()

    // When: Operator clicks all three controls
    await user.click(screen.getByRole('button', { name: 'Back' }))
    await user.click(screen.getByRole('button', { name: 'Skip' }))
    await user.click(screen.getByRole('button', { name: 'Next' }))

    // Then: Each callback is invoked exactly once
    expect(onBack).toHaveBeenCalledTimes(1)
    expect(onSkip).toHaveBeenCalledTimes(1)
    expect(onNext).toHaveBeenCalledTimes(1)
  })
})
