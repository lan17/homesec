import { useEffect, useMemo, useRef, useState } from 'react'
import Hls from 'hls.js'

import { isAPIError } from '../../../api/client'
import { Button } from '../../../components/ui/Button'
import { StatusBadge } from '../../../components/ui/StatusBadge'
import { describeUnknownError } from '../../shared/errorPresentation'
import { useCameraPreview } from '../hooks/useCameraPreview'

const PLAYLIST_POLL_DELAY_MS = 500
const PLAYLIST_POLL_MAX_ATTEMPTS = 12
const PLAYBACK_RETRY_DELAY_MS = 1000
const PREVIEW_DISPLAY_STATUS_STATES = new Set(['starting', 'ready', 'degraded', 'stopping'])

type WebKitFullscreenDocument = Document & {
  webkitExitFullscreen?: () => Promise<void> | void
  webkitFullscreenElement?: Element | null
}

type WebKitFullscreenElement = HTMLElement & {
  webkitRequestFullscreen?: () => Promise<void> | void
}

type WebKitVideoElement = HTMLVideoElement & {
  webkitEnterFullscreen?: () => void
}

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

function activeFullscreenElement(): Element | null {
  const fullscreenDocument = document as WebKitFullscreenDocument
  return document.fullscreenElement ?? fullscreenDocument.webkitFullscreenElement ?? null
}

async function requestElementFullscreen(element: HTMLElement): Promise<boolean> {
  const fullscreenElement = element as WebKitFullscreenElement
  if (fullscreenElement.requestFullscreen) {
    await fullscreenElement.requestFullscreen()
    return true
  }
  if (fullscreenElement.webkitRequestFullscreen) {
    await fullscreenElement.webkitRequestFullscreen()
    return true
  }
  return false
}

async function exitActiveFullscreen(): Promise<void> {
  const fullscreenDocument = document as WebKitFullscreenDocument
  if (document.exitFullscreen) {
    await document.exitFullscreen()
    return
  }
  if (fullscreenDocument.webkitExitFullscreen) {
    await fullscreenDocument.webkitExitFullscreen()
  }
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
  const viewportRef = useRef<HTMLDivElement | null>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const [playlistReady, setPlaylistReady] = useState(false)
  const [isPreviewFullscreen, setIsPreviewFullscreen] = useState(false)
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
          if (!cancelled) {
            setPlaylistReady(true)
          }
          return
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
    const syncFullscreenState = (): void => {
      setIsPreviewFullscreen(activeFullscreenElement() === viewportRef.current)
    }

    syncFullscreenState()
    document.addEventListener('fullscreenchange', syncFullscreenState)
    document.addEventListener('webkitfullscreenchange', syncFullscreenState)

    return () => {
      document.removeEventListener('fullscreenchange', syncFullscreenState)
      document.removeEventListener('webkitfullscreenchange', syncFullscreenState)
    }
  }, [])

  useEffect(() => {
    const video = videoRef.current
    if (!video || !playlistUrl || !playlistReady) {
      return
    }

    let hls: Hls | null = null
    let keepPlaybackActive = true
    let resumeTimeoutId: number | null = null
    let resumeIntervalId: number | null = null
    setPlayerError(null)

    video.muted = true
    video.defaultMuted = true
    video.autoplay = true
    video.playsInline = true
    video.setAttribute('autoplay', '')
    video.setAttribute('muted', '')
    video.setAttribute('playsinline', '')
    video.setAttribute('webkit-playsinline', '')

    const clearResumeTimeout = (): void => {
      if (resumeTimeoutId === null) {
        return
      }
      window.clearTimeout(resumeTimeoutId)
      resumeTimeoutId = null
    }

    const clearResumeInterval = (): void => {
      if (resumeIntervalId === null) {
        return
      }
      window.clearInterval(resumeIntervalId)
      resumeIntervalId = null
    }

    const requestPlayback = (): void => {
      if (!keepPlaybackActive) {
        return
      }
      clearResumeTimeout()
      resumeTimeoutId = window.setTimeout(() => {
        resumeTimeoutId = null
        if (!keepPlaybackActive) {
          return
        }
        if (!video.paused && !video.ended) {
          return
        }
        void video.play().catch(() => {})
      }, 0)
    }

    const startPlaybackMonitor = (): void => {
      requestPlayback()
      clearResumeInterval()
      resumeIntervalId = window.setInterval(() => {
        if (!keepPlaybackActive || document.visibilityState === 'hidden') {
          return
        }
        if (video.paused || video.ended) {
          requestPlayback()
        }
      }, PLAYBACK_RETRY_DELAY_MS)
    }

    const handleVisibilityChange = (): void => {
      if (document.visibilityState === 'visible') {
        requestPlayback()
      }
    }

    const cleanupPlayback = (): void => {
      keepPlaybackActive = false
      clearResumeTimeout()
      clearResumeInterval()
      video.removeEventListener('pause', requestPlayback)
      video.removeEventListener('ended', requestPlayback)
      video.removeEventListener('loadedmetadata', requestPlayback)
      video.removeEventListener('loadeddata', requestPlayback)
      video.removeEventListener('canplay', requestPlayback)
      video.removeEventListener('canplaythrough', requestPlayback)
      video.removeEventListener('stalled', requestPlayback)
      video.removeEventListener('waiting', requestPlayback)
      document.removeEventListener('visibilitychange', handleVisibilityChange)
      hls?.destroy()
      video.pause()
      video.removeAttribute('src')
      video.load()
    }

    video.addEventListener('pause', requestPlayback)
    video.addEventListener('ended', requestPlayback)
    video.addEventListener('loadedmetadata', requestPlayback)
    video.addEventListener('loadeddata', requestPlayback)
    video.addEventListener('canplay', requestPlayback)
    video.addEventListener('canplaythrough', requestPlayback)
    video.addEventListener('stalled', requestPlayback)
    video.addEventListener('waiting', requestPlayback)
    document.addEventListener('visibilitychange', handleVisibilityChange)

    if (Hls.isSupported()) {
      hls = new Hls({
        enableWorker: true,
        lowLatencyMode: true,
      })
      hls.loadSource(playlistUrl)
      hls.attachMedia(video)
      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        startPlaybackMonitor()
      })
      hls.on(Hls.Events.ERROR, (_event, data) => {
        if (!data.fatal) {
          return
        }
        setPlayerError('Preview playback failed. Restart preview.')
        keepPlaybackActive = false
        clearResumeTimeout()
        clearResumeInterval()
        hls?.destroy()
        hls = null
      })

      startPlaybackMonitor()

      return cleanupPlayback
    }

    if (video.canPlayType('application/vnd.apple.mpegurl')) {
      video.src = playlistUrl
      startPlaybackMonitor()
      return cleanupPlayback
    }

    setPlayerError('This browser cannot play the live preview stream.')

    return cleanupPlayback
  }, [playlistReady, playlistUrl])

  const toggleFullscreen = async (): Promise<void> => {
    const viewport = viewportRef.current
    if (!viewport) {
      return
    }

    try {
      if (isPreviewFullscreen) {
        await exitActiveFullscreen()
      } else {
        const fullscreenRequested = await requestElementFullscreen(viewport)
        if (!fullscreenRequested) {
          const video = videoRef.current as WebKitVideoElement | null
          video?.webkitEnterFullscreen?.()
        }
      }
    } catch {
      return
    }

    void videoRef.current?.play().catch(() => {})
  }

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

      <div className="camera-preview__viewport" ref={viewportRef}>
        {playlistUrl && playlistReady && !playerError ? (
          <>
            <video
              ref={videoRef}
              className="camera-preview__video"
              muted
              autoPlay
              playsInline
              preload="auto"
            />
            <button
              type="button"
              className="camera-preview__fullscreen-button"
              aria-label={isPreviewFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
              title={isPreviewFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
              onClick={() => {
                void toggleFullscreen()
              }}
            >
              <span className="camera-preview__fullscreen-icon" aria-hidden="true" />
            </button>
          </>
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
