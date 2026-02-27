import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'

import { Button } from '../../components/ui/Button'
import type { WizardStepDef } from './types'

interface SetupWizardShellProps {
  steps: readonly WizardStepDef[]
  currentStep: number
  completedSteps: ReadonlySet<string>
  skippedSteps: ReadonlySet<string>
  canGoNext: boolean
  showNext?: boolean
  nextLabel?: string
  onNext: () => void
  onBack: () => void
  onSkip: () => void
  children: ReactNode
}

function stepState(
  step: WizardStepDef,
  stepIndex: number,
  currentStep: number,
  completedSteps: ReadonlySet<string>,
  skippedSteps: ReadonlySet<string>,
): 'pending' | 'active' | 'complete' | 'skipped' {
  if (stepIndex === currentStep) {
    return 'active'
  }
  if (completedSteps.has(step.id)) {
    return 'complete'
  }
  if (skippedSteps.has(step.id)) {
    return 'skipped'
  }
  return 'pending'
}

export function SetupWizardShell({
  steps,
  currentStep,
  completedSteps,
  skippedSteps,
  canGoNext,
  showNext = true,
  nextLabel = 'Next',
  onNext,
  onBack,
  onSkip,
  children,
}: SetupWizardShellProps) {
  const activeStep = steps[currentStep]
  const backDisabled = currentStep <= 0

  return (
    <section className="wizard fade-in-up" aria-label="Setup wizard">
      <header className="wizard__header">
        <p className="wizard__kicker">HomeSec Setup Wizard</p>
        <h1 className="wizard__title">{activeStep?.title ?? 'Setup'}</h1>
        {activeStep?.subtitle ? <p className="wizard__subtitle">{activeStep.subtitle}</p> : null}
      </header>

      <ol className="wizard__progress" aria-label="Setup steps">
        {steps.map((step, index) => {
          const status = stepState(step, index, currentStep, completedSteps, skippedSteps)
          const classes = `wizard__progress-step wizard__progress-step--${status}`
          return (
            <li key={step.id} className={classes} aria-current={status === 'active' ? 'step' : undefined}>
              <span className="wizard__progress-index">{index + 1}</span>
              <span className="wizard__progress-label">{step.title}</span>
            </li>
          )
        })}
      </ol>

      <div className="wizard__body">{children}</div>

      <footer className="wizard__footer">
        <div className="wizard__footer-actions">
          <Button variant="ghost" onClick={onBack} disabled={backDisabled}>
            Back
          </Button>
          {activeStep?.skippable ? (
            <Button variant="ghost" onClick={onSkip}>
              Skip
            </Button>
          ) : null}
          {showNext ? (
            <Button onClick={onNext} disabled={!canGoNext}>
              {nextLabel}
            </Button>
          ) : null}
        </div>
        <Link className="wizard__exit-link" to="/">
          Exit to Dashboard
        </Link>
      </footer>
    </section>
  )
}
