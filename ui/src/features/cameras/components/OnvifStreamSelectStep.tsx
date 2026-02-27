import type { DiscoveredCameraResponse, ProbeResponse } from '../../../api/generated/types'
import { Button } from '../../../components/ui/Button'

interface OnvifStreamSelectStepProps {
  camera: DiscoveredCameraResponse
  probeResult: ProbeResponse
  selectedProfileToken: string | null
  cameraName: string
  createPending: boolean
  isMutating: boolean
  applyChangesImmediately: boolean
  showApplyChangesSummary?: boolean
  submitLabel?: string
  error: string | null
  onSelectProfile: (profileToken: string) => void
  onCameraNameChange: (value: string) => void
  onCreate: () => void
  onBack: () => void
  onCancel: () => void
}

export function OnvifStreamSelectStep({
  camera,
  probeResult,
  selectedProfileToken,
  cameraName,
  createPending,
  isMutating,
  applyChangesImmediately,
  showApplyChangesSummary = true,
  submitLabel = 'Create camera',
  error,
  onSelectProfile,
  onCameraNameChange,
  onCreate,
  onBack,
  onCancel,
}: OnvifStreamSelectStepProps) {
  const canSubmit = selectedProfileToken !== null && cameraName.trim().length > 0 && !isMutating
  const hasSelectableProfiles = probeResult.profiles.some((profile) => profile.stream_uri !== null)

  return (
    <form
      className="onvif-wizard__step inline-form"
      onSubmit={(event) => {
        event.preventDefault()
        onCreate()
      }}
    >
      <div className="onvif-step-header">
        <h3 className="onvif-step-title">Step 3: Select stream and create camera</h3>
        <div className="inline-form__actions">
          <Button variant="ghost" onClick={onBack} disabled={isMutating}>
            Back
          </Button>
          <Button variant="ghost" onClick={onCancel} disabled={isMutating}>
            Cancel
          </Button>
        </div>
      </div>

      <div className="onvif-device-info">
        <p className="muted">
          Camera: <span className="camera-mono">{camera.ip}</span>
        </p>
        <p className="muted">
          Device:{' '}
          <span className="camera-mono">
            {probeResult.device.manufacturer} {probeResult.device.model}
          </span>
        </p>
        <p className="subtle">
          Firmware: {probeResult.device.firmware_version || 'n/a'} | Serial:{' '}
          {probeResult.device.serial_number || 'n/a'}
        </p>
      </div>

      {!hasSelectableProfiles ? (
        <p className="error-text">
          Probe succeeded, but no usable stream URI was returned. Retry with different credentials or
          choose another camera.
        </p>
      ) : null}

      <div className="onvif-profile-list">
        {probeResult.profiles.map((profile) => {
          const disabled = profile.stream_uri === null
          const selected = selectedProfileToken === profile.token
          const classNames = [
            'onvif-profile-card',
            selected ? 'onvif-profile-card--selected' : '',
            disabled ? 'onvif-profile-card--disabled' : '',
          ]
            .filter(Boolean)
            .join(' ')

          return (
            <button
              key={profile.token}
              type="button"
              className={classNames}
              disabled={disabled || isMutating}
              onClick={() => {
                onSelectProfile(profile.token)
              }}
            >
              <p className="onvif-profile-card__title">{profile.name}</p>
              <p className="muted onvif-profile-card__line">
                {profile.video_encoding ?? 'unknown'} | {profile.width ?? '?'}x{profile.height ?? '?'} |{' '}
                {profile.frame_rate_limit ?? '?'} fps
              </p>
              <p className="subtle onvif-profile-card__line">
                Bitrate: {profile.bitrate_limit_kbps ?? '?'} kbps
              </p>
              {profile.stream_uri ? (
                <p className="subtle onvif-profile-card__line">
                  URI: <span className="camera-mono">{profile.stream_uri}</span>
                </p>
              ) : (
                <p className="error-text onvif-profile-card__line">
                  Unavailable: {profile.stream_error || 'No stream URI returned'}
                </p>
              )}
            </button>
          )
        })}
      </div>

      <label className="field-label" htmlFor="onvif-camera-name">
        Camera name
        <input
          id="onvif-camera-name"
          className="input"
          type="text"
          value={cameraName}
          onChange={(event) => {
            onCameraNameChange(event.target.value)
          }}
          disabled={isMutating}
        />
      </label>

      {showApplyChangesSummary ? (
        <p className="subtle">
          Apply changes immediately: {applyChangesImmediately ? 'enabled' : 'disabled'}.
        </p>
      ) : null}
      {error ? <p className="error-text">{error}</p> : null}

      <div className="inline-form__actions">
        <Button type="submit" disabled={!canSubmit}>
          {createPending ? 'Submitting...' : submitLabel}
        </Button>
      </div>
    </form>
  )
}
