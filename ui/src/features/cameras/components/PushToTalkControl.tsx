import { useRef } from 'react'

import { isAPIError } from '../../../api/client'
import type { TalkStatusResponse } from '../../../api/generated/types'
import { Button } from '../../../components/ui/Button'
import { StatusBadge } from '../../../components/ui/StatusBadge'
import { TechnicalDetailsDisclosure } from '../../../components/ui/TechnicalDetailsDisclosure'
import { describeUnknownError } from '../../shared/errorPresentation'
import { usePushToTalk } from '../hooks/usePushToTalk'

interface PushToTalkControlProps {
  cameraName: string
}

type TalkUiState = 'idle' | 'connecting' | 'talking' | 'microphone_blocked' | 'unavailable' | 'stopping'

function talkTone(state: TalkUiState): 'healthy' | 'degraded' | 'unhealthy' | 'unknown' {
  switch (state) {
    case 'talking':
      return 'healthy'
    case 'connecting':
    case 'stopping':
      return 'degraded'
    case 'microphone_blocked':
    case 'unavailable':
      return 'unhealthy'
    default:
      return 'unknown'
  }
}

function talkLabel(state: TalkUiState): string {
  switch (state) {
    case 'talking':
      return 'Talking'
    case 'connecting':
      return 'Connecting'
    case 'stopping':
      return 'Stopping'
    case 'microphone_blocked':
      return 'Microphone blocked'
    case 'unavailable':
      return 'Talk unavailable'
    default:
      return 'Hold to talk'
  }
}

function buttonLabel(state: TalkUiState): string {
  switch (state) {
    case 'connecting':
      return 'Connecting...'
    case 'talking':
      return 'Talking'
    case 'microphone_blocked':
      return 'Microphone blocked'
    case 'unavailable':
      return 'Talk unavailable'
    case 'stopping':
      return 'Stopping...'
    default:
      return 'Hold to talk'
  }
}

function codecList(codecs: string[] | undefined): string {
  return codecs && codecs.length > 0 ? codecs.join(', ') : 'none'
}

function technicalCapabilityMessage(status: TalkStatusResponse | null): string | null {
  if (!status) {
    return null
  }
  switch (status.capability) {
    case 'unsupported_codec':
      return `Camera talkback codec is not supported. Offered: ${codecList(status.offered_codecs)}. Supported: ${codecList(status.supported_codecs)}.`
    case 'unsupported':
      return status.last_error ?? 'This camera does not advertise a talkback channel.'
    case 'config_error':
      return status.last_error ?? status.backend_reason ?? 'Talkback configuration is invalid.'
    case 'error':
      return status.last_error
        ? `Talkback capability check failed: ${status.last_error}`
        : 'Talkback capability check failed.'
    case 'probing':
    case 'unknown':
      return 'Checking talkback capability...'
    default:
      return null
  }
}

function isMicrophoneBlocked(error: unknown): boolean {
  if (!error || typeof error !== 'object') {
    return false
  }
  const name = 'name' in error && typeof error.name === 'string' ? error.name : ''
  const message = 'message' in error && typeof error.message === 'string' ? error.message : ''
  return (
    name === 'NotAllowedError'
    || name === 'PermissionDeniedError'
    || name === 'SecurityError'
    || /permission denied|notallowed|not allowed/i.test(message)
  )
}

function isTalkUnavailable(status: TalkStatusResponse | null, state: string | undefined): boolean {
  if (!status) {
    return false
  }
  if (
    status.capability === 'disabled'
    || status.capability === 'unsupported'
    || status.capability === 'unsupported_codec'
    || status.capability === 'config_error'
    || status.capability === 'error'
  ) {
    return true
  }
  return state === 'disabled' || state === 'unsupported' || state === 'error'
}

function uiState({
  error,
  isStarting,
  isStopping,
  isStreaming,
  status,
  state,
}: {
  error: unknown
  isStarting: boolean
  isStopping: boolean
  isStreaming: boolean
  status: TalkStatusResponse | null
  state: string | undefined
}): TalkUiState {
  if (isMicrophoneBlocked(error)) {
    return 'microphone_blocked'
  }
  if (isStreaming) {
    return 'talking'
  }
  if (isStarting || state === 'starting') {
    return 'connecting'
  }
  if (isStopping || state === 'stopping') {
    return 'stopping'
  }
  if (isTalkUnavailable(status, state) || state === 'temporarily_unavailable' || state === 'active') {
    return 'unavailable'
  }
  return 'idle'
}

function userMessage({
  error,
  isPending,
  status,
  state,
  talkState,
}: {
  error: unknown
  isPending: boolean
  status: TalkStatusResponse | null
  state: string | undefined
  talkState: TalkUiState
}): string | null {
  if (talkState === 'microphone_blocked') {
    return 'Microphone blocked. Allow microphone access in your browser and try again.'
  }
  if (error) {
    if (isAPIError(error) && error.errorCode === 'TALK_SESSION_ALREADY_ACTIVE') {
      return 'Another talk session is already active.'
    }
    return describeUnknownError(error)
  }
  if (isPending) {
    return 'Checking talk availability...'
  }
  if (talkState === 'talking') {
    return null
  }
  switch (state) {
    case 'temporarily_unavailable':
      return 'Talk is busy. Try again in a moment.'
    case 'active':
      return 'Another talk session is already active.'
    default:
      return isTalkUnavailable(status, state) ? 'Talk unavailable on this camera.' : null
  }
}

function technicalDetails(status: TalkStatusResponse | null, error: unknown): string[] {
  const details: string[] = []
  const capability = technicalCapabilityMessage(status)
  if (capability) {
    details.push(capability)
  }
  if (status?.backend_reason) {
    details.push(`Backend detail: ${status.backend_reason}`)
  }
  if (status?.selected_codec) {
    details.push(`Selected codec: ${status.selected_codec}`)
  }
  if (error && !isMicrophoneBlocked(error)) {
    details.push(`Last error: ${describeUnknownError(error)}`)
  }
  return [...new Set(details)]
}

export function PushToTalkControl({ cameraName }: PushToTalkControlProps) {
  const {
    canStart,
    error,
    isPending,
    isStarting,
    isStopping,
    isStreaming,
    refreshStatus,
    session,
    start,
    status,
    stop,
  } = usePushToTalk(cameraName)
  const keyHeldRef = useRef(false)
  const pointerHeldRef = useRef(false)
  const disabled = !canStart && !isStarting && !isStreaming
  const state = isStreaming ? 'active' : (status?.state ?? session?.state)
  const talkState = uiState({ error, isStarting, isStopping, isStreaming, status, state })
  const message = userMessage({ error, isPending, status, state, talkState })
  const details = technicalDetails(status, error)
  const talkButtonClassName = talkState === 'talking'
    ? 'push-to-talk__button push-to-talk__button--active'
    : 'push-to-talk__button'

  const beginTalk = (): void => {
    if (!canStart) {
      return
    }
    void start()
  }

  const endTalk = (): void => {
    void stop()
  }

  return (
    <section className="push-to-talk" aria-label="Push to talk">
      <header className="push-to-talk__header">
        <div>
          <p className="camera-preview__title">Push to talk</p>
          <p className="camera-preview__subtitle">Hold the button to speak through this camera.</p>
        </div>
        <StatusBadge tone={talkTone(talkState)}>{talkLabel(talkState)}</StatusBadge>
      </header>

      {message ? <p className="camera-preview__message">{message}</p> : null}
      {talkState === 'talking' ? (
        <p className="push-to-talk__hint">Talking. Release to stop.</p>
      ) : null}

      <div className="inline-form__actions push-to-talk__actions">
        <Button
          className={talkButtonClassName}
          aria-pressed={isStarting || isStreaming}
          disabled={disabled}
          onPointerDown={(event) => {
            if (event.button !== 0 || disabled) {
              return
            }
            event.preventDefault()
            pointerHeldRef.current = true
            if (event.currentTarget.setPointerCapture) {
              event.currentTarget.setPointerCapture(event.pointerId)
            }
            beginTalk()
          }}
          onPointerUp={(event) => {
            if (!pointerHeldRef.current) {
              return
            }
            event.preventDefault()
            pointerHeldRef.current = false
            if (event.currentTarget.releasePointerCapture) {
              event.currentTarget.releasePointerCapture(event.pointerId)
            }
            endTalk()
          }}
          onPointerCancel={() => {
            pointerHeldRef.current = false
            endTalk()
          }}
          onLostPointerCapture={() => {
            if (pointerHeldRef.current) {
              pointerHeldRef.current = false
              endTalk()
            }
          }}
          onKeyDown={(event) => {
            if ((event.key !== ' ' && event.key !== 'Enter') || event.repeat || disabled) {
              return
            }
            event.preventDefault()
            keyHeldRef.current = true
            beginTalk()
          }}
          onKeyUp={(event) => {
            if ((event.key !== ' ' && event.key !== 'Enter') || !keyHeldRef.current) {
              return
            }
            event.preventDefault()
            keyHeldRef.current = false
            endTalk()
          }}
        >
          {buttonLabel(talkState)}
        </Button>
        <Button
          variant="ghost"
          onClick={() => {
            void refreshStatus()
          }}
          disabled={isPending || isStarting || isStopping}
        >
          Refresh talk
        </Button>
      </div>

      {details.length > 0 ? (
        <TechnicalDetailsDisclosure summary="Technical talk details">
          <ul className="technical-details__list">
            {details.map((detail) => (
              <li key={detail}>{detail}</li>
            ))}
          </ul>
        </TechnicalDetailsDisclosure>
      ) : null}
    </section>
  )
}
