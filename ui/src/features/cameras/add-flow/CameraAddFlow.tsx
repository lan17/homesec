import { useMemo, useState } from 'react'

import type { CameraCreate } from '../../../api/generated/types'
import { Button } from '../../../components/ui/Button'
import { Card } from '../../../components/ui/Card'
import type { CameraAddFlowOnComplete } from '../actions'
import { BackendPicker } from './BackendPicker'
import { CAMERA_ADD_BACKENDS, suggestCameraName } from './backends'
import { ConfirmStep } from './ConfirmStep'
import { TestConnectionStep } from './TestConnectionStep'
import { useCameraAddState } from './useCameraAddState'
import { describeCameraCreateError, validateCameraName } from './validation'

interface CameraAddFlowProps {
  existingCameraNames: readonly string[]
  defaultApplyChangesImmediately?: boolean
  onApplyChangesImmediatelyChange?: (value: boolean) => void
  onComplete: CameraAddFlowOnComplete
  onDone: () => void
  onCancel: () => void
}

export function CameraAddFlow({
  existingCameraNames,
  defaultApplyChangesImmediately = false,
  onApplyChangesImmediatelyChange,
  onComplete,
  onDone,
  onCancel,
}: CameraAddFlowProps) {
  const cameraAddState = useCameraAddState()
  const [applyChangesImmediately, setApplyChangesImmediately] = useState(defaultApplyChangesImmediately)
  const [validationError, setValidationError] = useState<string | null>(null)
  const [createError, setCreateError] = useState<string | null>(null)
  const [submitPending, setSubmitPending] = useState(false)

  const backendDef =
    cameraAddState.state.backend === null
      ? null
      : CAMERA_ADD_BACKENDS[cameraAddState.state.backend]
  const backendStepCount = backendDef?.steps.length ?? 1

  const configureStepTitle = useMemo(() => {
    if (!backendDef) {
      return null
    }
    return backendDef.steps[cameraAddState.state.backendStepIndex]?.title ?? null
  }, [backendDef, cameraAddState.state.backendStepIndex])

  function handleApplyChangesImmediatelyChange(value: boolean): void {
    setApplyChangesImmediately(value)
    onApplyChangesImmediatelyChange?.(value)
  }

  function handleCancel(): void {
    cameraAddState.reset()
    setValidationError(null)
    setCreateError(null)
    onCancel()
  }

  function handleSelectBackend(backendId: keyof typeof CAMERA_ADD_BACKENDS): void {
    const selectedBackend = CAMERA_ADD_BACKENDS[backendId]
    cameraAddState.selectBackend({
      backend: selectedBackend.id,
      defaultConfig: selectedBackend.defaultConfig,
      suggestedName: suggestCameraName(selectedBackend.suggestNamePrefix, existingCameraNames),
    })
    setValidationError(null)
    setCreateError(null)
  }

  function handleNext(): void {
    if (!backendDef) {
      return
    }

    if (cameraAddState.state.step === 'configure') {
      const maybeError = backendDef.validateStep(
        cameraAddState.state.backendStepIndex,
        cameraAddState.state.config,
      )
      if (maybeError) {
        setValidationError(maybeError)
        return
      }
    }

    setValidationError(null)
    cameraAddState.goNext(backendStepCount)
  }

  function handleBack(): void {
    if (!backendDef) {
      return
    }
    setValidationError(null)
    cameraAddState.goBack(backendStepCount)
  }

  async function handleCreateCamera(): Promise<void> {
    if (!backendDef) {
      return
    }

    const cameraNameError = validateCameraName(cameraAddState.state.cameraName)
    if (cameraNameError) {
      setCreateError(cameraNameError)
      return
    }

    const payload: CameraCreate = {
      name: cameraAddState.state.cameraName.trim(),
      enabled: true,
      ...backendDef.buildCameraSource(cameraAddState.state.config),
    }

    setCreateError(null)
    setSubmitPending(true)
    try {
      const createResult = await onComplete(payload, { applyChangesImmediately })
      if (!createResult.ok) {
        setCreateError(describeCameraCreateError(createResult.error))
        return
      }
      cameraAddState.reset()
      setValidationError(null)
      setCreateError(null)
      onDone()
    } finally {
      setSubmitPending(false)
    }
  }

  const stage = cameraAddState.state.step

  return (
    <Card title="Add Camera" subtitle="Choose a backend, test connectivity, and confirm creation.">
      {stage === 'pick-backend' ? (
        <BackendPicker
          isMutating={submitPending}
          onSelect={handleSelectBackend}
          onCancel={handleCancel}
        />
      ) : null}

      {stage === 'configure' && backendDef ? (
        <section className="inline-form camera-add-flow">
          <h3 className="camera-add-flow__title">{configureStepTitle ?? 'Configure source'}</h3>
          {(() => {
            const StepComponent = backendDef.steps[cameraAddState.state.backendStepIndex]?.component
            if (!StepComponent) {
              return <p className="error-text">Backend step configuration is missing.</p>
            }
            return (
              <StepComponent
                config={cameraAddState.state.config}
                onChange={(nextConfig) => {
                  cameraAddState.updateConfig(nextConfig)
                  setValidationError(null)
                }}
                stepIndex={cameraAddState.state.backendStepIndex}
                onSuggestedNameChange={(cameraName) => {
                  cameraAddState.setCameraName(cameraName)
                }}
              />
            )
          })()}

          {validationError ? <p className="error-text">{validationError}</p> : null}

          <div className="inline-form__actions">
            <Button variant="ghost" onClick={handleBack}>
              Back
            </Button>
            <Button variant="ghost" onClick={handleCancel}>
              Cancel
            </Button>
            <Button onClick={handleNext}>Next</Button>
          </div>
        </section>
      ) : null}

      {stage === 'test' && backendDef ? (
        <section className="inline-form camera-add-flow">
          <TestConnectionStep
            request={backendDef.buildTestRequest(cameraAddState.state.config)}
            result={cameraAddState.state.testResult}
            onResult={(result) => {
              cameraAddState.setTestResult(result)
            }}
          />

          <div className="inline-form__actions">
            <Button variant="ghost" onClick={handleBack}>
              Back
            </Button>
            <Button variant="ghost" onClick={handleCancel}>
              Cancel
            </Button>
            <Button onClick={handleNext}>Continue</Button>
          </div>
        </section>
      ) : null}

      {stage === 'confirm' && backendDef ? (
        <section className="inline-form camera-add-flow">
          <ConfirmStep
            backendLabel={backendDef.label}
            cameraName={cameraAddState.state.cameraName}
            config={cameraAddState.state.config}
            applyChangesImmediately={applyChangesImmediately}
            testResult={cameraAddState.state.testResult}
            submitPending={submitPending}
            createError={createError}
            onCameraNameChange={(cameraName) => {
              cameraAddState.setCameraName(cameraName)
              setCreateError(null)
            }}
            onApplyChangesImmediatelyChange={handleApplyChangesImmediatelyChange}
            onSubmit={() => {
              void handleCreateCamera()
            }}
          />

          <div className="inline-form__actions">
            <Button variant="ghost" onClick={handleBack}>
              Back
            </Button>
            <Button variant="ghost" onClick={handleCancel}>
              Cancel
            </Button>
          </div>
        </section>
      ) : null}
    </Card>
  )
}

