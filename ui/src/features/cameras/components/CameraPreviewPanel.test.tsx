// @vitest-environment happy-dom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { CameraPreviewPanel } from './CameraPreviewPanel'

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

const READY_PLAYLIST = '#EXTM3U\n#EXTINF:1.0,\nsegment_000000.ts\n#EXTINF:1.0,\nsegment_000001.ts\n'

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
      new Response(READY_PLAYLIST, {
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
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
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

    // Then: The player probes the playlist URL and initializes hls.js
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
    })
  })

  it('waits for a usable live HLS window before initializing playback', async () => {
    // Given: A ready preview session whose first playlist poll has only one segment
    const playlistUrl =
      'http://localhost:8081/api/v1/preview/cameras/front/playlist.m3u8?token=preview-token'
    vi.mocked(globalThis.fetch)
      .mockResolvedValueOnce(
        new Response('#EXTM3U\n#EXTINF:1.0,\nsegment_000000.ts\n', {
          status: 200,
          headers: { 'content-type': 'application/vnd.apple.mpegurl' },
        }),
      )
      .mockResolvedValueOnce(
        new Response(READY_PLAYLIST, {
          status: 200,
          headers: { 'content-type': 'application/vnd.apple.mpegurl' },
        }),
      )
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

    // When: Rendering the preview panel before the playlist has enough media
    render(<CameraPreviewPanel cameraName="front" />)
    await waitFor(() => {
      expect(globalThis.fetch).toHaveBeenCalledTimes(1)
    })

    // Then: The player waits instead of attaching to a thin live window
    expect(hlsConstructMock).not.toHaveBeenCalled()

    // Then: Playback initialization proceeds once a later poll sees a full enough window
    await waitFor(() => {
      expect(globalThis.fetch).toHaveBeenCalledTimes(2)
      expect(hlsConstructMock).toHaveBeenCalledTimes(1)
      expect(hlsLoadSourceMock).toHaveBeenCalledWith(playlistUrl)
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
