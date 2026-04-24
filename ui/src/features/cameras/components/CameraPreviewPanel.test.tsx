// @vitest-environment happy-dom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { CameraPreviewPanel } from './CameraPreviewPanel'

const DEFAULT_PLAYLIST_URL =
  'http://localhost:8081/api/v1/preview/cameras/front/playlist.m3u8?token=preview-token'

const {
  useCameraPreviewMock,
  hlsAttachMediaMock,
  hlsConstructMock,
  hlsDestroyMock,
  hlsIsSupportedMock,
  hlsLoadSourceMock,
  hlsOnMock,
} = vi.hoisted(() => ({
  useCameraPreviewMock: vi.fn(),
  hlsConstructMock: vi.fn(),
  hlsLoadSourceMock: vi.fn(),
  hlsAttachMediaMock: vi.fn(),
  hlsOnMock: vi.fn(),
  hlsDestroyMock: vi.fn(),
  hlsIsSupportedMock: vi.fn(() => true),
}))

vi.mock('../hooks/useCameraPreview', () => ({
  useCameraPreview: (...args: unknown[]) => useCameraPreviewMock(...args),
}))

vi.mock('hls.js', () => {
  function MockHls(this: Record<string, unknown>) {
    hlsConstructMock()
    this.loadSource = hlsLoadSourceMock
    this.attachMedia = hlsAttachMediaMock
    this.on = hlsOnMock
    this.destroy = hlsDestroyMock
  }

  Object.assign(MockHls, {
    isSupported: hlsIsSupportedMock,
    Events: {
      MANIFEST_PARSED: 'manifestParsed',
      ERROR: 'error',
    },
  })

  return { default: MockHls }
})

function mockReadyPreviewSession(playlistUrl: string = DEFAULT_PLAYLIST_URL) {
  useCameraPreviewMock.mockReturnValue({
    status: {
      camera_name: 'front',
      enabled: true,
      state: 'ready',
      viewer_count: 1,
      degraded_reason: null,
      last_error: null,
      idle_shutdown_at: null,
      httpStatus: 200,
    },
    session: {
      camera_name: 'front',
      state: 'ready',
      viewer_count: 1,
      token: 'preview-token',
      token_expires_at: null,
      playlist_url: '/api/v1/preview/cameras/front/playlist.m3u8?token=preview-token',
      idle_timeout_s: 30,
      warning: null,
      httpStatus: 200,
    },
    playlistUrl,
    warning: null,
    error: null,
    isPending: false,
    isStarting: false,
    isStopping: false,
    start: vi.fn(),
    stop: vi.fn(),
    refreshStatus: vi.fn(),
  })
}

describe('CameraPreviewPanel', () => {
  beforeEach(() => {
    useCameraPreviewMock.mockReset()
    hlsConstructMock.mockReset()
    hlsLoadSourceMock.mockReset()
    hlsAttachMediaMock.mockReset()
    hlsOnMock.mockReset()
    hlsDestroyMock.mockReset()
    hlsIsSupportedMock.mockReset()
    hlsIsSupportedMock.mockReturnValue(true)
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response('#EXTM3U', {
        status: 200,
        headers: { 'content-type': 'application/vnd.apple.mpegurl' },
      }),
    )
    Object.defineProperty(HTMLMediaElement.prototype, 'canPlayType', {
      configurable: true,
      value: vi.fn(() => ''),
    })
    Object.defineProperty(HTMLMediaElement.prototype, 'play', {
      configurable: true,
      value: vi.fn().mockResolvedValue(undefined),
    })
    Object.defineProperty(HTMLMediaElement.prototype, 'pause', {
      configurable: true,
      value: vi.fn(),
    })
    Object.defineProperty(HTMLMediaElement.prototype, 'load', {
      configurable: true,
      value: vi.fn(),
    })
    Object.defineProperty(HTMLElement.prototype, 'requestFullscreen', {
      configurable: true,
      value: vi.fn().mockResolvedValue(undefined),
    })
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('renders disabled preview state from the hook', () => {
    // Given: Preview is disabled for the current camera
    useCameraPreviewMock.mockReturnValue({
      status: {
        camera_name: 'front',
        enabled: false,
        state: 'idle',
        viewer_count: null,
        degraded_reason: null,
        last_error: null,
        idle_shutdown_at: null,
        httpStatus: 200,
      },
      session: null,
      playlistUrl: null,
      warning: null,
      error: null,
      isPending: false,
      isStarting: false,
      isStopping: false,
      start: vi.fn(),
      stop: vi.fn(),
      refreshStatus: vi.fn(),
    })

    // When: Rendering the preview panel
    render(<CameraPreviewPanel cameraName="front" />)

    // Then: The panel surfaces the disabled state and blocks start
    expect(screen.getByText('Preview is disabled for this runtime.')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Start preview' }).hasAttribute('disabled')).toBe(true)
  })

  it('wires start, refresh, and stop actions from the preview hook', async () => {
    // Given: A ready preview panel with active session
    const start = vi.fn().mockResolvedValue(undefined)
    const stop = vi.fn().mockResolvedValue(undefined)
    const refreshStatus = vi.fn().mockResolvedValue(undefined)
    useCameraPreviewMock.mockReturnValue({
      status: {
        camera_name: 'front',
        enabled: true,
        state: 'ready',
        viewer_count: 2,
        degraded_reason: null,
        last_error: null,
        idle_shutdown_at: null,
        httpStatus: 200,
      },
      session: {
        camera_name: 'front',
        state: 'ready',
        viewer_count: 2,
        token: 'preview-token',
        token_expires_at: null,
        playlist_url: '/api/v1/preview/cameras/front/playlist.m3u8?token=preview-token',
        idle_timeout_s: 30,
        warning: null,
        httpStatus: 200,
      },
      playlistUrl: 'http://localhost:8081/api/v1/preview/cameras/front/playlist.m3u8?token=preview-token',
      warning: null,
      error: null,
      isPending: false,
      isStarting: false,
      isStopping: false,
      start,
      stop,
      refreshStatus,
    })
    const user = userEvent.setup()

    // When: Rendering and using the preview controls
    render(<CameraPreviewPanel cameraName="front" />)
    await user.click(screen.getByRole('button', { name: 'Attach preview' }))
    await user.click(screen.getByRole('button', { name: 'Refresh status' }))
    await user.click(screen.getByRole('button', { name: 'Stop preview' }))

    // Then: The panel forwards button actions to the hook handlers
    expect(start).toHaveBeenCalledTimes(1)
    expect(refreshStatus).toHaveBeenCalledTimes(1)
    expect(stop).toHaveBeenCalledTimes(1)
    expect(screen.getByText('viewers 2')).toBeTruthy()
  })

  it('initializes hls.js playback when a playlist URL becomes ready', async () => {
    // Given: A ready preview session with a tokenized playlist URL
    mockReadyPreviewSession()
    const play = vi.mocked(HTMLMediaElement.prototype.play)

    // When: Rendering the preview panel
    render(<CameraPreviewPanel cameraName="front" />)

    // Then: The player probes the playlist URL, initializes hls.js, and requests playback
    await waitFor(() => {
      expect(globalThis.fetch).toHaveBeenCalledWith(
        'http://localhost:8081/api/v1/preview/cameras/front/playlist.m3u8?token=preview-token',
        expect.objectContaining({ cache: 'no-store' }),
      )
      expect(hlsConstructMock).toHaveBeenCalledTimes(1)
      expect(hlsLoadSourceMock).toHaveBeenCalledWith(
        'http://localhost:8081/api/v1/preview/cameras/front/playlist.m3u8?token=preview-token',
      )
      expect(hlsAttachMediaMock).toHaveBeenCalledTimes(1)
      expect(play).toHaveBeenCalled()
    })
  })

  it('uses hls.js before native HLS when both playback paths are available', async () => {
    // Given: Safari-like native HLS support and hls.js support are both available
    vi.mocked(HTMLMediaElement.prototype.canPlayType).mockReturnValue('maybe')
    mockReadyPreviewSession()

    // When: Rendering the preview panel
    render(<CameraPreviewPanel cameraName="front" />)

    // Then: The player takes the hls.js path instead of short-circuiting to native HLS
    await waitFor(() => {
      expect(hlsConstructMock).toHaveBeenCalledTimes(1)
      expect(hlsLoadSourceMock).toHaveBeenCalledWith(DEFAULT_PLAYLIST_URL)
      expect(HTMLMediaElement.prototype.canPlayType).not.toHaveBeenCalled()
    })
  })

  it('falls back to native HLS when hls.js is unavailable', async () => {
    // Given: hls.js cannot run but the browser supports native HLS playback
    hlsIsSupportedMock.mockReturnValue(false)
    vi.mocked(HTMLMediaElement.prototype.canPlayType).mockReturnValue('maybe')
    mockReadyPreviewSession()

    // When: Rendering the preview panel
    const { container } = render(<CameraPreviewPanel cameraName="front" />)

    // Then: The player assigns the playlist directly to the video element
    await waitFor(() => {
      const video = container.querySelector('video')
      expect(video?.getAttribute('src')).toBe(DEFAULT_PLAYLIST_URL)
      expect(hlsConstructMock).not.toHaveBeenCalled()
    })
  })

  it('renders attached previews with only a fullscreen playback control', async () => {
    // Given: A ready preview session with playable live media
    mockReadyPreviewSession()

    // When: Rendering the preview panel
    const { container } = render(<CameraPreviewPanel cameraName="front" />)

    // Then: The attached video does not expose browser play or pause controls
    await waitFor(() => {
      const video = container.querySelector('video')
      expect(video).toBeTruthy()
      expect(video?.hasAttribute('controls')).toBe(false)
      expect(video?.getAttribute('preload')).toBe('auto')
      expect(video?.getAttribute('muted')).toBe('')
      expect(video?.getAttribute('playsinline')).toBe('')
      expect(video?.getAttribute('webkit-playsinline')).toBe('')
      expect(screen.getByRole('button', { name: 'Enter fullscreen' })).toBeTruthy()
    })
  })

  it('requests fullscreen for the preview viewport', async () => {
    // Given: A ready preview session with a fullscreen-capable viewport
    mockReadyPreviewSession()
    const user = userEvent.setup()
    render(<CameraPreviewPanel cameraName="front" />)

    // When: Entering fullscreen from the preview overlay control
    await user.click(await screen.findByRole('button', { name: 'Enter fullscreen' }))

    // Then: The browser fullscreen API is invoked without enabling native video controls
    expect(HTMLElement.prototype.requestFullscreen).toHaveBeenCalledTimes(1)
    expect(screen.getByRole('button', { name: 'Enter fullscreen' })).toBeTruthy()
  })

  it('resumes playback when the attached video pauses', async () => {
    // Given: A ready preview session with an attached hls.js player
    mockReadyPreviewSession()
    const play = vi.mocked(HTMLMediaElement.prototype.play)
    const { container } = render(<CameraPreviewPanel cameraName="front" />)

    await waitFor(() => {
      expect(hlsAttachMediaMock).toHaveBeenCalledTimes(1)
      expect(play).toHaveBeenCalled()
    })
    play.mockClear()
    const video = container.querySelector('video')

    // When: The browser pauses the live video element
    video?.dispatchEvent(new Event('pause'))

    // Then: The panel requests playback again while the preview remains attached
    await waitFor(() => {
      expect(play).toHaveBeenCalled()
    })
  })

  it('keeps stop enabled when a session exists but status is temporarily unavailable', () => {
    // Given: An active preview session with no current status snapshot
    useCameraPreviewMock.mockReturnValue({
      status: null,
      session: {
        camera_name: 'front',
        state: 'ready',
        viewer_count: 1,
        token: 'preview-token',
        token_expires_at: null,
        playlist_url: '/api/v1/preview/cameras/front/playlist.m3u8?token=preview-token',
        idle_timeout_s: 30,
        warning: null,
        httpStatus: 200,
      },
      playlistUrl: 'http://localhost:8081/api/v1/preview/cameras/front/playlist.m3u8?token=preview-token',
      warning: null,
      error: null,
      isPending: false,
      isStarting: false,
      isStopping: false,
      start: vi.fn(),
      stop: vi.fn(),
      refreshStatus: vi.fn(),
    })

    // When: Rendering the preview panel
    render(<CameraPreviewPanel cameraName="front" />)

    // Then: Stop remains available because an active session still exists
    expect(screen.getByRole('button', { name: 'Stop preview' }).hasAttribute('disabled')).toBe(
      false,
    )
    expect(screen.getByText('READY')).toBeTruthy()
  })

  it('prefers the active session state over a stale terminal status snapshot', () => {
    // Given: A live preview session with a stale idle status snapshot
    useCameraPreviewMock.mockReturnValue({
      status: {
        camera_name: 'front',
        enabled: true,
        state: 'idle',
        viewer_count: null,
        degraded_reason: null,
        last_error: null,
        idle_shutdown_at: null,
        httpStatus: 200,
      },
      session: {
        camera_name: 'front',
        state: 'ready',
        viewer_count: 1,
        token: 'preview-token',
        token_expires_at: null,
        playlist_url: '/api/v1/preview/cameras/front/playlist.m3u8?token=preview-token',
        idle_timeout_s: 30,
        warning: null,
        httpStatus: 200,
      },
      playlistUrl: 'http://localhost:8081/api/v1/preview/cameras/front/playlist.m3u8?token=preview-token',
      warning: null,
      error: null,
      isPending: false,
      isStarting: false,
      isStopping: false,
      start: vi.fn(),
      stop: vi.fn(),
      refreshStatus: vi.fn(),
    })

    // When: Rendering the preview panel
    render(<CameraPreviewPanel cameraName="front" />)

    // Then: The session stays labeled as active instead of regressing to stale idle state
    expect(screen.getByText('READY')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Attach preview' })).toBeTruthy()
  })
})
