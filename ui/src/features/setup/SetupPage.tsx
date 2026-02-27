import { useNavigate } from 'react-router-dom'

import { Button } from '../../components/ui/Button'
import type { CameraCreate } from '../../api/generated/types'
import type { DetectionStepData } from '../settings/detection/types'
import type { StorageFormState } from '../settings/storage/types'
import { SetupWizardShell } from './SetupWizardShell'
import { CameraStep } from './steps/CameraStep'
import { DetectionStep } from './steps/DetectionStep'
import { StorageStep } from './steps/StorageStep'
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

interface CameraStepDraft {
  camera: CameraCreate
}

interface StorageStepDraft {
  storage: StorageFormState
}

interface DetectionStepDraft {
  detection: DetectionStepData
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

function toCameraStepDraft(value: unknown): CameraStepDraft | null {
  if (
    value &&
    typeof value === 'object' &&
    !Array.isArray(value) &&
    'camera' in value
  ) {
    const cameraValue = (value as { camera: unknown }).camera
    if (cameraValue && typeof cameraValue === 'object' && !Array.isArray(cameraValue)) {
      return {
        camera: cameraValue as CameraCreate,
      }
    }
  }
  return null
}

function toStorageStepDraft(value: unknown): StorageStepDraft | null {
  if (
    value &&
    typeof value === 'object' &&
    !Array.isArray(value) &&
    'storage' in value
  ) {
    const storageValue = (value as { storage: unknown }).storage
    if (
      storageValue &&
      typeof storageValue === 'object' &&
      !Array.isArray(storageValue) &&
      'backend' in storageValue &&
      'config' in storageValue
    ) {
      const typed = storageValue as { backend: unknown; config: unknown }
      if (
        (typed.backend === 'local' || typed.backend === 'dropbox')
        && typed.config
        && typeof typed.config === 'object'
        && !Array.isArray(typed.config)
      ) {
        return {
          storage: {
            backend: typed.backend,
            config: typed.config as Record<string, unknown>,
          },
        }
      }
    }
  }
  return null
}

function toDetectionStepDraft(value: unknown): DetectionStepDraft | null {
  if (
    value
    && typeof value === 'object'
    && !Array.isArray(value)
    && 'detection' in value
  ) {
    const detectionValue = (value as { detection: unknown }).detection
    if (
      detectionValue
      && typeof detectionValue === 'object'
      && !Array.isArray(detectionValue)
      && 'filter' in detectionValue
      && 'vlm' in detectionValue
    ) {
      return {
        detection: detectionValue as DetectionStepData,
      }
    }
  }
  return null
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
  const isCameraStep = activeStep.id === 'camera'
  const isStorageStep = activeStep.id === 'storage'
  const isDetectionStep = activeStep.id === 'detection'
  const cameraStepDraft = toCameraStepDraft(state.stepData.camera)
  const storageStepDraft = toStorageStepDraft(state.stepData.storage)
  const detectionStepDraft = toDetectionStepDraft(state.stepData.detection)

  function handleNoteChange(note: string): void {
    updateStepData(activeStep.id, { note })
  }

  function handleMarkComplete(): void {
    markComplete(activeStep.id)
  }

  function handleCameraStepUpdateData(camera: CameraCreate): void {
    updateStepData('camera', { camera }, { persist: false })
  }

  function handleCameraStepComplete(): void {
    markComplete('camera')
    goNext()
  }

  function handleCameraStepSkip(): void {
    skipStep()
  }

  function handleStorageStepUpdateData(storage: StorageFormState): void {
    updateStepData('storage', { storage })
  }

  function handleStorageStepComplete(): void {
    markComplete('storage')
    goNext()
  }

  function handleStorageStepSkip(): void {
    skipStep()
  }

  function handleDetectionStepUpdateData(detection: DetectionStepData): void {
    updateStepData('detection', { detection })
  }

  function handleDetectionStepComplete(): void {
    markComplete('detection')
    goNext()
  }

  function handleDetectionStepSkip(): void {
    skipStep()
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
    if (isCameraStep) {
      return (
        <CameraStep
          existingCameraNames={
            cameraStepDraft ? [cameraStepDraft.camera.name] : []
          }
          onUpdateData={handleCameraStepUpdateData}
          onComplete={handleCameraStepComplete}
          onSkip={handleCameraStepSkip}
        />
      )
    }
    if (isStorageStep) {
      return (
        <StorageStep
          initialData={storageStepDraft?.storage ?? null}
          onUpdateData={handleStorageStepUpdateData}
          onComplete={handleStorageStepComplete}
          onSkip={handleStorageStepSkip}
        />
      )
    }
    if (isDetectionStep) {
      return (
        <DetectionStep
          initialData={detectionStepDraft?.detection ?? null}
          onUpdateData={handleDetectionStepUpdateData}
          onComplete={handleDetectionStepComplete}
          onSkip={handleDetectionStepSkip}
        />
      )
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
