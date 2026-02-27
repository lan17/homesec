import { useNavigate } from 'react-router-dom'

import type { CameraCreate } from '../../api/generated/types'
import type { DetectionStepData } from '../settings/detection/types'
import type { NotificationStepData } from '../settings/notifiers/types'
import type { StorageFormState } from '../settings/storage/types'
import { SetupWizardShell } from './SetupWizardShell'
import type { ReviewWizardDrafts } from './review'
import {
  parseCameraStepDraft,
  parseDetectionStepDraft,
  parseNotificationStepDraft,
  parseStorageStepDraft,
} from './stepDrafts'
import { CameraStep } from './steps/CameraStep'
import { DetectionStep } from './steps/DetectionStep'
import { NotificationStep } from './steps/NotificationStep'
import { ReviewStep } from './steps/ReviewStep'
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

export function SetupPage() {
  const navigate = useNavigate()
  const {
    state,
    goNext,
    goBack,
    goToStep,
    skipStep,
    updateStepData,
    markComplete,
    clearPersistedState,
  } =
    useWizardState(WIZARD_STEPS)

  const activeStep = WIZARD_STEPS[state.currentStep] ?? WIZARD_STEPS[0]
  const isComplete = state.completedSteps.has(activeStep.id)
  const isReviewStep = activeStep.id === 'review'
  const canGoNext = !isReviewStep && (activeStep.skippable || isComplete)
  const completedCount = state.completedSteps.size
  const skippedCount = state.skippedSteps.size
  const isWelcomeStep = activeStep.id === 'welcome'
  const isCameraStep = activeStep.id === 'camera'
  const isStorageStep = activeStep.id === 'storage'
  const isDetectionStep = activeStep.id === 'detection'
  const isNotificationsStep = activeStep.id === 'notifications'
  const cameraStepDraft = parseCameraStepDraft(state.stepData.camera)
  const storageStepDraft = parseStorageStepDraft(state.stepData.storage)
  const detectionStepDraft = parseDetectionStepDraft(state.stepData.detection)
  const notificationStepDraft = parseNotificationStepDraft(state.stepData.notifications)
  const reviewWizardData: ReviewWizardDrafts = {
    camera: cameraStepDraft,
    storage: storageStepDraft,
    detection: detectionStepDraft,
    notifications: notificationStepDraft,
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

  function handleNotificationStepUpdateData(notifications: NotificationStepData): void {
    updateStepData('notifications', { notifications })
  }

  function handleNotificationStepComplete(): void {
    markComplete('notifications')
    goNext()
  }

  function handleNotificationStepSkip(): void {
    skipStep()
  }

  function handleNext(): void {
    if (!canGoNext) {
      return
    }
    goNext()
  }

  function handleReviewLaunchSuccess(): void {
    clearPersistedState()
  }

  function handleGoDashboard(): void {
    clearPersistedState()
    navigate('/', { replace: true })
  }

  function renderActiveStepContent() {
    if (isWelcomeStep) {
      return <WelcomeStep isComplete={isComplete} onComplete={handleMarkComplete} />
    }
    if (isCameraStep) {
      return (
        <CameraStep
          existingCameraNames={
            cameraStepDraft ? [cameraStepDraft.name] : []
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
          initialData={storageStepDraft}
          onUpdateData={handleStorageStepUpdateData}
          onComplete={handleStorageStepComplete}
          onSkip={handleStorageStepSkip}
        />
      )
    }
    if (isDetectionStep) {
      return (
        <DetectionStep
          initialData={detectionStepDraft}
          onUpdateData={handleDetectionStepUpdateData}
          onComplete={handleDetectionStepComplete}
          onSkip={handleDetectionStepSkip}
        />
      )
    }
    if (isNotificationsStep) {
      return (
        <NotificationStep
          initialData={notificationStepDraft}
          onUpdateData={handleNotificationStepUpdateData}
          onComplete={handleNotificationStepComplete}
          onSkip={handleNotificationStepSkip}
        />
      )
    }
    if (isReviewStep) {
      return (
        <ReviewStep
          wizardData={reviewWizardData}
          skippedSteps={state.skippedSteps}
          onGoToStep={goToStep}
          onLaunchSuccess={handleReviewLaunchSuccess}
          onGoDashboard={handleGoDashboard}
        />
      )
    }

    return (
      <section className="wizard-step-card">
        <p className="subtle">Unknown setup step.</p>
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
        showNext={!isReviewStep}
        nextLabel="Next"
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
          {isReviewStep ? (
            <p className="subtle">
              Launch clears persisted wizard state and returns to the dashboard.
            </p>
          ) : null}
        </section>
      </SetupWizardShell>
    </main>
  )
}
