import { useRef } from 'react'

import { isAPIError } from '../../../api/client'
import { Button } from '../../../components/ui/Button'
import { StatusBadge } from '../../../components/ui/StatusBadge'
import { describeUnknownError } from '../../shared/errorPresentation'
import { usePushToTalk } from '../hooks/usePushToTalk'

interface PushToTalkControlProps {
  cameraName: string
}

function talkTone(state: string | undefined): 'healthy' | 'degraded' | 'unhealthy' | 'unknown' {
  switch (state) {
    case 'active':
      return 'healthy'
    case 'starting':
    case 'stopping':
    case 'temporarily_unavailable':
      return 'degraded'
    case 'disabled':
    case 'unsupported':
    case 'error':
      return 'unhealthy'
    default:
      return 'unknown'
  }
}

function talkLabel(state: string | undefined): string {
  switch (state) {
    case 'active':
      return 'TALKING'
    case 'starting':
      return 'CONNECTING'
    case 'stopping':
      return 'STOPPING'
    case 'disabled':
      return 'DISABLED'
    case 'unsupported':
      return 'UNSUPPORTED'
    case 'temporarily_unavailable':
      return 'BUSY'
    case 'error':
      return 'ERROR'
    default:
      return 'IDLE'
  }
}

function buttonLabel(isStarting: boolean, isStreaming: boolean, isStopping: boolean): string {
  if (isStopping) {
    return 'Stopping...'
  }
  if (isStarting) {
    return 'Connecting...'
  }
  if (isStreaming) {
    return 'Release to stop'
  }
  return 'Hold to talk'
}

function blockedMessage(state: string | undefined): string | null {
  switch (state) {
    case 'disabled':
      return 'Push-to-talk is disabled for this camera.'
    case 'unsupported':
      return 'This camera source does not support talkback.'
    case 'temporarily_unavailable':
      return 'Talkback is temporarily unavailable. Try again in a moment.'
    case 'active':
      return 'Another talk session is already active.'
    default:
      return null
  }
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
  const message = (() => {
    if (error) {
      if (isAPIError(error) && error.errorCode === 'TALK_SESSION_ALREADY_ACTIVE') {
        return 'Another talk session is already active.'
      }
      return describeUnknownError(error)
    }
    if (isPending) {
      return 'Checking talkback capability...'
    }
    return blockedMessage(state)
  })()

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
          <p className="camera-preview__subtitle">Hold the mic button to stream browser audio to the camera speaker.</p>
        </div>
        <StatusBadge tone={talkTone(state)}>{talkLabel(state)}</StatusBadge>
      </header>

      {message ? <p className="camera-preview__message">{message}</p> : null}

      <div className="inline-form__actions">
        <Button
          className={isStreaming ? 'push-to-talk__button push-to-talk__button--active' : 'push-to-talk__button'}
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
          {buttonLabel(isStarting, isStreaming, isStopping)}
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
    </section>
  )
}
