import { useState } from 'react'

import type { CameraCreate } from '../../../../api/generated/types'
import { Button } from '../../../../components/ui/Button'
import { OnvifDiscoveryWizard } from '../../components/OnvifDiscoveryWizard'
import type { BackendFormStepProps } from './types'
import { readString } from './configReaders'

export function OnvifForm({ config, onChange, onSuggestedNameChange }: BackendFormStepProps) {
  const selectedRtspUrl = readString(config, 'rtsp_url', '')
  const [showWizard, setShowWizard] = useState(selectedRtspUrl.trim().length === 0)

  async function handleResolvedCandidate(payload: CameraCreate) {
    onChange({
      ...config,
      ...payload.source_config,
      output_dir:
        typeof payload.source_config.output_dir === 'string'
          ? payload.source_config.output_dir
          : './recordings',
    })
    onSuggestedNameChange(payload.name)
    setShowWizard(false)
    return { ok: true as const }
  }

  return (
    <div className="inline-form">
      {selectedRtspUrl ? (
        <div className="camera-add-flow__step-hint">
          <p className="subtle">
            Selected stream URI: <span className="camera-mono">{selectedRtspUrl}</span>
          </p>
        </div>
      ) : (
        <p className="subtle">
          Complete ONVIF discovery to select a stream profile before continuing.
        </p>
      )}

      {showWizard ? null : (
        <Button
          type="button"
          variant="ghost"
          onClick={() => {
            setShowWizard(true)
          }}
        >
          {selectedRtspUrl ? 'Choose different stream' : 'Open ONVIF discovery'}
        </Button>
      )}

      {showWizard ? (
        <OnvifDiscoveryWizard
          applyChangesImmediately={false}
          createPending={false}
          isMutating={false}
          submitLabel="Use selected stream"
          showApplyChangesSummary={false}
          onCreateCamera={handleResolvedCandidate}
          onClose={() => {
            // Keep operator on current configure step; parent controls overall flow navigation.
            setShowWizard(false)
          }}
        />
      ) : null}
    </div>
  )
}
