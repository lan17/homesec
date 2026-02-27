import type { CameraCreate } from '../../../../api/generated/types'
import { OnvifDiscoveryWizard } from '../../components/OnvifDiscoveryWizard'
import type { BackendFormStepProps } from './types'

function readString(config: Record<string, unknown>, key: string, fallback: string): string {
  const value = config[key]
  return typeof value === 'string' ? value : fallback
}

export function OnvifForm({ config, onChange, onSuggestedNameChange }: BackendFormStepProps) {
  const selectedRtspUrl = readString(config, 'rtsp_url', '')

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
    return { ok: true as const }
  }

  return (
    <div className="inline-form">
      {selectedRtspUrl ? (
        <p className="subtle">
          Selected stream URI: <span className="camera-mono">{selectedRtspUrl}</span>
        </p>
      ) : (
        <p className="subtle">
          Complete ONVIF discovery to select a stream profile before continuing.
        </p>
      )}

      <OnvifDiscoveryWizard
        applyChangesImmediately={false}
        createPending={false}
        isMutating={false}
        submitLabel="Use selected stream"
        showApplyChangesSummary={false}
        onCreateCamera={handleResolvedCandidate}
        onClose={() => {
          // Keep operator on current configure step; parent controls overall flow navigation.
        }}
      />
    </div>
  )
}

