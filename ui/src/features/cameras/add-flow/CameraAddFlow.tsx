import { useState, type FormEvent } from 'react'

import type { CameraCreate } from '../../../api/generated/types'
import { isAPIError } from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import type { CameraAddFlowOnComplete } from '../actions'
import {
  defaultSourceConfigForBackend,
  parseSourceConfigJson,
  type CameraBackend,
} from '../forms'
import { OnvifDiscoveryWizard } from '../components/OnvifDiscoveryWizard'
import { BackendPicker } from './BackendPicker'
import { ManualCameraConfigureStep } from './ManualCameraConfigureStep'
import type { CameraAddBackend } from './types'

interface CameraAddFlowProps {
  defaultApplyChangesImmediately?: boolean
  onApplyChangesImmediatelyChange?: (value: boolean) => void
  onComplete: CameraAddFlowOnComplete
  onCancel: () => void
}

type CameraAddFlowStage = 'pick-backend' | 'configure-manual' | 'configure-onvif'

function describeCreateError(error: unknown): string {
  if (isAPIError(error) && error.errorCode === 'CAMERA_ALREADY_EXISTS') {
    return 'Camera name already exists. Choose a different name and retry.'
  }
  if (error instanceof Error && error.message.trim().length > 0) {
    return `Create camera failed: ${error.message}`
  }
  return 'Create camera failed. Review the page error details and retry.'
}

interface ManualCameraDraft {
  name: string
  enabled: boolean
  backend: CameraBackend
  sourceConfigRaw: string
}

function createManualCameraDraft(backend: CameraBackend): ManualCameraDraft {
  return {
    name: '',
    enabled: true,
    backend,
    sourceConfigRaw: defaultSourceConfigForBackend(backend),
  }
}

export function CameraAddFlow({
  defaultApplyChangesImmediately = false,
  onApplyChangesImmediatelyChange,
  onComplete,
  onCancel,
}: CameraAddFlowProps) {
  const [applyChangesImmediately, setApplyChangesImmediately] = useState(defaultApplyChangesImmediately)
  const [stage, setStage] = useState<CameraAddFlowStage>('pick-backend')
  const [manualDraft, setManualDraft] = useState<ManualCameraDraft>(createManualCameraDraft('rtsp'))
  const [manualError, setManualError] = useState<string | null>(null)
  const [submitPending, setSubmitPending] = useState(false)
  const isMutating = submitPending

  function handleApplyChangesImmediatelyChange(value: boolean): void {
    setApplyChangesImmediately(value)
    onApplyChangesImmediatelyChange?.(value)
  }

  async function submitCamera(payload: CameraCreate) {
    setSubmitPending(true)
    try {
      return await onComplete(payload, { applyChangesImmediately })
    } finally {
      setSubmitPending(false)
    }
  }

  function handleSelectBackend(backend: CameraAddBackend): void {
    setManualError(null)
    if (backend === 'onvif') {
      setStage('configure-onvif')
      return
    }
    setManualDraft(createManualCameraDraft(backend))
    setStage('configure-manual')
  }

  async function handleManualSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault()

    const normalizedName = manualDraft.name.trim()
    if (!normalizedName) {
      setManualError('Camera name is required.')
      return
    }

    const parsedSourceConfig = parseSourceConfigJson(manualDraft.sourceConfigRaw)
    if (!parsedSourceConfig.ok) {
      setManualError(parsedSourceConfig.message)
      return
    }

    const createResult = await submitCamera(
      {
        name: normalizedName,
        enabled: manualDraft.enabled,
        source_backend: manualDraft.backend,
        source_config: parsedSourceConfig.value,
      },
    )

    if (!createResult.ok) {
      setManualError(describeCreateError(createResult.error))
      return
    }

    setManualError(null)
    setStage('pick-backend')
    onCancel()
  }

  if (stage === 'configure-onvif') {
    return (
      <OnvifDiscoveryWizard
        applyChangesImmediately={applyChangesImmediately}
        createPending={submitPending}
        isMutating={isMutating}
        onCreateCamera={submitCamera}
        onClose={() => {
          setStage('pick-backend')
          onCancel()
        }}
      />
    )
  }

  return (
    <Card title="Add Camera" subtitle="Reusable flow for manual and ONVIF camera onboarding">
      {stage === 'pick-backend' ? (
        <BackendPicker
          applyChangesImmediately={applyChangesImmediately}
          isMutating={isMutating}
          onSelect={handleSelectBackend}
          onApplyChangesImmediatelyChange={handleApplyChangesImmediatelyChange}
          onCancel={onCancel}
        />
      ) : null}

      {stage === 'configure-manual' ? (
        <ManualCameraConfigureStep
          backend={manualDraft.backend}
          cameraName={manualDraft.name}
          cameraEnabled={manualDraft.enabled}
          sourceConfigRaw={manualDraft.sourceConfigRaw}
          errorMessage={manualError}
          applyChangesImmediately={applyChangesImmediately}
          isMutating={isMutating}
          createPending={submitPending}
          onSubmit={(event) => {
            void handleManualSubmit(event)
          }}
          onBack={() => {
            setManualError(null)
            setStage('pick-backend')
          }}
          onCancel={onCancel}
          onCameraNameChange={(name) => {
            setManualDraft((current) => ({ ...current, name }))
            setManualError(null)
          }}
          onCameraEnabledChange={(enabled) => {
            setManualDraft((current) => ({ ...current, enabled }))
          }}
          onSourceConfigChange={(sourceConfigRaw) => {
            setManualDraft((current) => ({ ...current, sourceConfigRaw }))
            setManualError(null)
          }}
          onApplyChangesImmediatelyChange={handleApplyChangesImmediatelyChange}
          onResetTemplate={() => {
            setManualDraft((current) => ({
              ...current,
              sourceConfigRaw: defaultSourceConfigForBackend(current.backend),
            }))
            setManualError(null)
          }}
        />
      ) : null}
    </Card>
  )
}
