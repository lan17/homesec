import { useEffect, useMemo, useRef, useState } from 'react'
import Hls from 'hls.js'

import { isAPIError } from '../../../api/client'
import { Button } from '../../../components/ui/Button'
import { StatusBadge } from '../../../components/ui/StatusBadge'
import { describeUnknownError } from '../../shared/errorPresentation'
import { useCameraPreview } from '../hooks/useCameraPreview'

const PLAYLIST_POLL_DELAY_MS = 500
const PLAYLIST_POLL_MAX_ATTEMPTS = 12
const PLAYLIST_READY_MIN_SEGMENTS = 2
const PREVIEW_DISPLAY_STATUS_STATES = new Set(['starting', 'ready', 'degraded', 'stopping'])

interface CameraPreviewPanelProps {
  cameraName: string
}

function previewTone(
  state: string | undefined,
): 'healthy' | 'degraded' | 'unhealthy' | 'unknown' {
  switch (state) {
    case 'ready':
      return 'healthy'
    case 'starting':
    case 'degraded':
    case 'stopping':
      return 'degraded'
    case 'error':
      return 'unhealthy'
    default:
      return 'unknown'
  }
}

function previewLabel(state: string | undefined): string {
  switch (state) {
    case 'ready':
      return 'READY'
    case 'starting':
      return 'STARTING'
    case 'degraded':
      return 'DEGRADED'
    case 'stopping':
      return 'STOPPING'
    case 'error':
      return 'ERROR'
    default:
      return 'IDLE'
  }
}

function startLabel(statusState: string | undefined): string {
  if (statusState === 'ready' || statusState === 'degraded' || statusState === 'starting') {
    return 'Attach preview'
  }
  if (statusState === 'error') {
    return 'Retry preview'
  }
  return 'Start preview'
}

function countPlaylistSegments(playlistText: string): number {
  return playlistText
    .split(/\r?\n/)
    .filter((line) => {
      const stripped = line.trim()
      return stripped.length > 0 && !stripped.startsWith('#')
    }).length
}

export function CameraPreviewPanel({ cameraName }: CameraPreviewPanelProps) {
  const {
    error,
    isPending,
    isStarting,
    isStopping,
    playlistUrl,
    refreshStatus,
    session,
    start,
    status,
    stop,
    warning,
  } = useCameraPreview(cameraName)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const [playlistReady, setPlaylistReady] = useState(false)
  const [playerError, setPlayerError] = useState<string | null>(null)
  const effectiveState =
    session && (!status || !PREVIEW_DISPLAY_STATUS_STATES.has(status.state))
      ? session.state
      : status?.state ?? session?.state
  const viewerCount = status?.viewer_count ?? session?.viewer_count

  useEffect(() => {
    if (!playlistUrl) {
      setPlaylistReady(false)
      setPlayerError(null)
      return
    }

    let cancelled = false
    let timeoutId: number | null = null
    const controller = new AbortController()
    setPlaylistReady(false)
    setPlayerError(null)

    const pollPlaylist = async (attempt: number): Promise<void> => {
      try {
        const response = await fetch(playlistUrl, {
          cache: 'no-store',
          signal: controller.signal,
        })
        if (response.ok) {
          const playlistText = await response.text()
          if (
            !cancelled
            && countPlaylistSegments(playlistText) >= PLAYLIST_READY_MIN_SEGMENTS
          ) {
            setPlaylistReady(true)
            return
          }
        }
        if (response.status !== 404 && response.status !== 409) {
          throw new Error(`Preview playlist unavailable (${response.status})`)
        }
      } catch (nextError) {
        if (controller.signal.aborted || cancelled) {
          return
        }
        if (attempt >= PLAYLIST_POLL_MAX_ATTEMPTS) {
          setPlayerError(describeUnknownError(nextError))
          return
        }
      }

      if (attempt >= PLAYLIST_POLL_MAX_ATTEMPTS) {
        if (!cancelled) {
          setPlayerError('Preview media is still starting. Try again.')
        }
        return
      }

      timeoutId = window.setTimeout(() => {
        void pollPlaylist(attempt + 1)
      }, PLAYLIST_POLL_DELAY_MS)
    }

    void pollPlaylist(0)

    return () => {
      cancelled = true
      controller.abort()
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId)
      }
    }
  }, [playlistUrl])

  useEffect(() => {
    const video = videoRef.current
    if (!video || !playlistUrl || !playlistReady) {
      return
    }

    let hls: Hls | null = null
    setPlayerError(null)

    if (video.canPlayType('application/vnd.apple.mpegurl')) {
      video.src = playlistUrl
      void video.play().catch(() => {})
      return () => {
        video.pause()
        video.removeAttribute('src')
        video.load()
      }
    }

    if (!Hls.isSupported()) {
      setPlayerError('This browser cannot play the live preview stream.')
      return
    }

    hls = new Hls({
      enableWorker: true,
      lowLatencyMode: true,
    })
    hls.loadSource(playlistUrl)
    hls.attachMedia(video)
    hls.on(Hls.Events.MANIFEST_PARSED, () => {
      void video.play().catch(() => {})
    })
    hls.on(Hls.Events.ERROR, (_event, data) => {
      if (!data.fatal) {
        return
      }
      setPlayerError('Preview playback failed. Restart preview.')
      hls?.destroy()
      hls = null
    })

    return () => {
      hls?.destroy()
      video.pause()
      video.removeAttribute('src')
      video.load()
    }
  }, [playlistReady, playlistUrl])

  const statusMessage = useMemo(() => {
    if (warning) {
      return warning
    }
    if (playerError) {
      return playerError
    }
    if (error) {
      if (isAPIError(error) && error.errorCode === 'PREVIEW_MEDIA_UNAVAILABLE') {
        return 'Preview media is still starting.'
      }
      return describeUnknownError(error)
    }
    if (playlistUrl && !playlistReady) {
      return 'Preparing preview stream...'
    }
    if (status?.enabled === false) {
      return 'Preview is disabled for this runtime.'
    }
    return null
  }, [error, playerError, playlistReady, playlistUrl, status?.enabled, warning])

  return (
    <section className="camera-preview">
      <header className="camera-preview__header">
        <div>
          <p className="camera-preview__title">Live preview</p>
          <p className="camera-preview__subtitle">On-demand HLS stream from the active runtime.</p>
        </div>
        <div className="camera-item__badges">
          <StatusBadge tone={previewTone(effectiveState)}>{previewLabel(effectiveState)}</StatusBadge>
          {viewerCount !== null && viewerCount !== undefined ? (
            <span className="camera-chip">viewers {viewerCount}</span>
          ) : null}
        </div>
      </header>

      <div className="camera-preview__viewport">
        {playlistUrl && playlistReady && !playerError ? (
          <video
            ref={videoRef}
            className="camera-preview__video"
            controls
            muted
            autoPlay
            playsInline
          />
        ) : (
          <div className="camera-preview__placeholder">
            {statusMessage ?? 'Start preview to attach to the live stream.'}
          </div>
        )}
      </div>

      {statusMessage && playlistUrl && playlistReady && !playerError ? (
        <p className="camera-preview__message">{statusMessage}</p>
      ) : null}

      <div className="inline-form__actions">
        <Button
          onClick={() => {
            void start()
          }}
          disabled={isPending || status?.enabled === false}
        >
          {isStarting ? 'Starting...' : startLabel(effectiveState)}
        </Button>
        <Button
          variant="ghost"
          onClick={() => {
            void refreshStatus()
          }}
          disabled={isPending}
        >
          Refresh status
        </Button>
        <Button
          variant="ghost"
          onClick={() => {
            void stop()
          }}
          disabled={isStopping || (!session && (effectiveState ?? 'idle') === 'idle')}
        >
          {isStopping ? 'Stopping...' : 'Stop preview'}
        </Button>
      </div>
    </section>
  )
}
