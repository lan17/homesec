import { useNavigate } from 'react-router-dom'

import { Button } from '../../components/ui/Button'
import { SetupWizardShell } from './SetupWizardShell'
import { WelcomeStep } from './steps/WelcomeStep'
import type { WizardStepDef } from './types'
import { useWizardState } from './useWizardState'
import './wizard.css'

const WIZARD_STEPS: readonly WizardStepDef[] = [
  {
    id: 'welcome',
    title: 'Welcome',
    subtitle: 'Review prerequisites before configuring cameras and services.',
    skippable: true,
  },
  {
    id: 'camera',
    title: 'Camera',
    subtitle: 'Add first camera source or skip for now.',
    skippable: true,
  },
  {
    id: 'storage',
    title: 'Storage',
    subtitle: 'Choose clip storage backend and path.',
    skippable: true,
  },
  {
    id: 'detection',
    title: 'Detection',
    subtitle: 'Tune detection/analyzer behavior.',
    skippable: true,
  },
  {
    id: 'notifications',
    title: 'Notifications',
    subtitle: 'Configure alert destinations.',
    skippable: true,
  },
  {
    id: 'review',
    title: 'Review & Launch',
    subtitle: 'Confirm setup decisions and launch HomeSec.',
    skippable: false,
  },
]

interface StepDraft {
  note: string
}

function toStepDraft(value: unknown): StepDraft {
  if (
    value &&
    typeof value === 'object' &&
    !Array.isArray(value) &&
    typeof (value as { note?: unknown }).note === 'string'
  ) {
    return {
      note: (value as { note: string }).note,
    }
  }
  return { note: '' }
}

function statusText(isComplete: boolean, isSkipped: boolean): string {
  if (isComplete) {
    return 'Completed'
  }
  if (isSkipped) {
    return 'Skipped'
  }
  return 'Pending'
}

export function SetupPage() {
  const navigate = useNavigate()
  const { state, goNext, goBack, skipStep, updateStepData, markComplete, reset } =
    useWizardState(WIZARD_STEPS)

  const activeStep = WIZARD_STEPS[state.currentStep] ?? WIZARD_STEPS[0]
  const isComplete = state.completedSteps.has(activeStep.id)
  const isSkipped = state.skippedSteps.has(activeStep.id)
  const draft = toStepDraft(state.stepData[activeStep.id])
  const canGoNext = activeStep.skippable || isComplete
  const isLastStep = state.currentStep === WIZARD_STEPS.length - 1
  const completedCount = state.completedSteps.size
  const skippedCount = state.skippedSteps.size
  const isWelcomeStep = activeStep.id === 'welcome'

  function handleNoteChange(note: string): void {
    updateStepData(activeStep.id, { note })
  }

  function handleMarkComplete(): void {
    markComplete(activeStep.id)
  }

  function handleNext(): void {
    if (!canGoNext) {
      return
    }
    if (isLastStep) {
      reset()
      navigate('/', { replace: true })
      return
    }
    goNext()
  }

  function renderActiveStepContent() {
    if (isWelcomeStep) {
      return <WelcomeStep isComplete={isComplete} onComplete={handleMarkComplete} />
    }

    return (
      <section className="wizard-step-card">
        <header className="wizard-step-card__header">
          <p className="wizard-step-card__status">
            Step status: {statusText(isComplete, isSkipped)}
          </p>
          {!isComplete ? (
            <Button variant="ghost" onClick={handleMarkComplete}>
              Mark step complete
            </Button>
          ) : null}
        </header>

        <label className="field-label" htmlFor="wizard-step-note">
          Step notes (non-secret)
        </label>
        <textarea
          id="wizard-step-note"
          className="input wizard-step-card__textarea"
          value={draft.note}
          onChange={(event) => handleNoteChange(event.target.value)}
          placeholder="Capture setup decisions for this step."
        />
        <p className="subtle">
          Security: credentials and secrets are not persisted in browser localStorage.
        </p>
      </section>
    )
  }

  return (
    <main className="setup-page">
      <SetupWizardShell
        steps={WIZARD_STEPS}
        currentStep={state.currentStep}
        completedSteps={state.completedSteps}
        skippedSteps={state.skippedSteps}
        canGoNext={canGoNext}
        nextLabel={isLastStep ? 'Launch' : 'Next'}
        onNext={handleNext}
        onBack={goBack}
        onSkip={skipStep}
      >
        {renderActiveStepContent()}

        <section className="wizard-summary-card">
          <h2 className="wizard-summary-card__title">Session summary</h2>
          <dl className="wizard-summary-card__grid">
            <div>
              <dt>Current step</dt>
              <dd>
                {state.currentStep + 1} / {WIZARD_STEPS.length}
              </dd>
            </div>
            <div>
              <dt>Completed</dt>
              <dd>{completedCount}</dd>
            </div>
            <div>
              <dt>Skipped</dt>
              <dd>{skippedCount}</dd>
            </div>
          </dl>
          {isLastStep ? (
            <p className="subtle">
              Launch clears persisted wizard state and returns to the dashboard.
            </p>
          ) : null}
        </section>
      </SetupWizardShell>
    </main>
  )
}
