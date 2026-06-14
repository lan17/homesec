// @vitest-environment happy-dom

import type { PropsWithChildren } from 'react'
import { act, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, describe, expect, it, vi } from 'vitest'

type MockNativeLifecycleState = {
  isActive: boolean
  isBackgrounded: boolean
  pauseCount: number
  resumeCount: number
}

const nativeLifecycleMock = vi.hoisted(() => ({
  state: {
    isActive: true,
    isBackgrounded: false,
    pauseCount: 0,
    resumeCount: 0,
  } as MockNativeLifecycleState,
}))

vi.mock('../../../runtime/nativeAppLifecycle', () => ({
  useNativeAppLifecycleState: () => nativeLifecycleMock.state,
}))

import { apiClient } from '../../../api/client'
import { useCameraPreview } from './useCameraPreview'

const PREVIEW_TEST_NOW_MS = Date.parse('2026-04-23T12:00:00.000Z')

function resetNativeLifecycleState() {
  nativeLifecycleMock.state = {
    isActive: true,
    isBackgrounded: false,
    pauseCount: 0,
    resumeCount: 0,
  }
}

function setNativeLifecycleState(nextState: Partial<MockNativeLifecycleState>) {
  nativeLifecycleMock.state = {
    ...nativeLifecycleMock.state,
    ...nextState,
  }
}

function freezePreviewClock() {
  vi.spyOn(Date, 'now').mockReturnValue(PREVIEW_TEST_NOW_MS)
}

type Deferred<T> = {
  promise: Promise<T>
  resolve: (value: T) => void
  reject: (error: unknown) => void
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void
  let reject!: (error: unknown) => void
  const promise = new Promise<T>((nextResolve, nextReject) => {
    resolve = nextResolve
    reject = nextReject
  })
  return { promise, resolve, reject }
}

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
      mutations: {
        retry: false,
      },
    },
  })

  return function Wrapper({ children }: PropsWithChildren) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  }
}

describe('useCameraPreview', () => {
  afterEach(() => {
    resetNativeLifecycleState()
    vi.restoreAllMocks()
    vi.useRealTimers()
  })

  it('swallows start mutation rejections and exposes the failure via hook state', async () => {
    // Given: Status loads but preview activation fails
    vi.spyOn(apiClient, 'getCameraPreviewStatus').mockResolvedValue({
      camera_name: 'front',
      enabled: true,
      state: 'idle',
      viewer_count: null,
      degraded_reason: null,
      last_error: null,
      idle_shutdown_at: null,
      httpStatus: 200,
    })
    vi.spyOn(apiClient, 'ensureCameraPreviewActive').mockRejectedValue(new Error('preview boom'))

    const { result } = renderHook(() => useCameraPreview('front'), {
      wrapper: createWrapper(),
    })

    await waitFor(() => {
      expect(result.current.status?.state).toBe('idle')
    })

    // When: Starting preview from the hook
    await act(async () => {
      await expect(result.current.start()).resolves.toBeUndefined()
    })

    // Then: The hook stores the failure without surfacing an unhandled rejection
    await waitFor(() => {
      expect(result.current.error?.message).toBe('preview boom')
      expect(result.current.session).toBeNull()
    })
  })

  it('keeps a fresh preview session when the follow-up status refetch fails', async () => {
    // Given: An idle camera whose preview start succeeds but the invalidated status refresh fails
    freezePreviewClock()
    vi.spyOn(apiClient, 'getCameraPreviewStatus')
      .mockResolvedValueOnce({
        camera_name: 'front',
        enabled: true,
        state: 'idle',
        viewer_count: null,
        degraded_reason: null,
        last_error: null,
        idle_shutdown_at: null,
        httpStatus: 200,
      })
      .mockRejectedValueOnce(new Error('status refresh failed'))
    vi.spyOn(apiClient, 'ensureCameraPreviewActive').mockResolvedValue({
      camera_name: 'front',
      state: 'ready',
      viewer_count: 1,
      token: 'preview-token-1',
      token_expires_at: '2026-04-24T12:00:10.000Z',
      playlist_url: '/api/v1/preview/cameras/front/playlist.m3u8?token=preview-token-1',
      idle_timeout_s: 30,
      warning: null,
      httpStatus: 200,
    })

    const { result } = renderHook(() => useCameraPreview('front'), {
      wrapper: createWrapper(),
    })

    await waitFor(() => {
      expect(result.current.status?.state).toBe('idle')
    })

    // When: Starting preview and allowing the invalidated status refresh to fail
    await act(async () => {
      await expect(result.current.start()).resolves.toBeUndefined()
    })

    // Then: The hook keeps serving the fresh preview session instead of dropping it
    await waitFor(() => {
      expect(result.current.session?.token).toBe('preview-token-1')
      expect(result.current.playlistUrl).toContain('preview-token-1')
    })
  })

  it('does not let an older in-flight status refresh clear a newer preview session', async () => {
    // Given: A stale status refresh that started before preview attach succeeds
    freezePreviewClock()
    let releaseStaleRefresh: ((value: {
      camera_name: string
      enabled: boolean
      state: 'idle'
      viewer_count: null
      degraded_reason: null
      last_error: null
      idle_shutdown_at: null
      httpStatus: 200
    }) => void) | null = null
    const staleRefresh = new Promise<{
      camera_name: string
      enabled: boolean
      state: 'idle'
      viewer_count: null
      degraded_reason: null
      last_error: null
      idle_shutdown_at: null
      httpStatus: 200
    }>((resolve) => {
      releaseStaleRefresh = resolve
    })
    vi.spyOn(apiClient, 'getCameraPreviewStatus')
      .mockResolvedValueOnce({
        camera_name: 'front',
        enabled: true,
        state: 'idle',
        viewer_count: null,
        degraded_reason: null,
        last_error: null,
        idle_shutdown_at: null,
        httpStatus: 200,
      })
      .mockImplementationOnce(() => staleRefresh)
      .mockResolvedValueOnce({
        camera_name: 'front',
        enabled: true,
        state: 'ready',
        viewer_count: 1,
        degraded_reason: null,
        last_error: null,
        idle_shutdown_at: null,
        httpStatus: 200,
      })
    const ensurePreviewActive = vi
      .spyOn(apiClient, 'ensureCameraPreviewActive')
      .mockResolvedValue({
        camera_name: 'front',
        state: 'ready',
        viewer_count: 1,
        token: 'preview-token-1',
        token_expires_at: '2026-04-24T12:00:10.000Z',
        playlist_url: '/api/v1/preview/cameras/front/playlist.m3u8?token=preview-token-1',
        idle_timeout_s: 30,
        warning: null,
        httpStatus: 200,
      })

    const { result } = renderHook(() => useCameraPreview('front'), {
      wrapper: createWrapper(),
    })

    await waitFor(() => {
      expect(result.current.status?.state).toBe('idle')
    })

    // When: A manual refresh starts, preview attaches, then the older refresh resolves idle
    await act(async () => {
      void result.current.refreshStatus()
      await Promise.resolve()
    })

    await act(async () => {
      await expect(result.current.start()).resolves.toBeUndefined()
    })

    await act(async () => {
      releaseStaleRefresh?.({
        camera_name: 'front',
        enabled: true,
        state: 'idle',
        viewer_count: null,
        degraded_reason: null,
        last_error: null,
        idle_shutdown_at: null,
        httpStatus: 200,
      })
      await Promise.resolve()
    })

    // Then: The stale refresh cannot clear the newer session, and the queued refetch converges to ready
    await waitFor(() => {
      expect(result.current.session?.token).toBe('preview-token-1')
      expect(result.current.playlistUrl).toContain('preview-token-1')
      expect(result.current.status?.state).toBe('ready')
    })
    expect(ensurePreviewActive).toHaveBeenCalledTimes(1)
  })

  it('refreshes preview sessions before short-lived playback tokens expire', async () => {
    // Given: A ready preview session with an expiring playback token
    const nowMs = Date.parse('2026-04-23T12:00:00.000Z')
    vi.spyOn(Date, 'now').mockReturnValue(nowMs)
    const realSetTimeout = window.setTimeout.bind(window)
    const realClearTimeout = window.clearTimeout.bind(window)
    let refreshDelayMs: number | null = null
    let runScheduledRefresh: (() => void) | null = null
    vi.spyOn(window, 'setTimeout').mockImplementation(((handler, timeout, ...args) => {
      if (timeout === 5_000) {
        refreshDelayMs = timeout
        runScheduledRefresh = () => {
          if (typeof handler !== 'function') {
            throw new Error('Expected refresh timer handler to be a function')
          }
          handler(...args)
        }
        return 99
      }
      return realSetTimeout(handler, timeout, ...args)
    }) as typeof window.setTimeout)
    vi.spyOn(window, 'clearTimeout').mockImplementation(((timeoutId) => {
      if (timeoutId === 99) {
        return
      }
      realClearTimeout(timeoutId)
    }) as typeof window.clearTimeout)
    vi.spyOn(apiClient, 'getCameraPreviewStatus').mockResolvedValue({
      camera_name: 'front',
      enabled: true,
      state: 'ready',
      viewer_count: 1,
      degraded_reason: null,
      last_error: null,
      idle_shutdown_at: null,
      httpStatus: 200,
    })
    const ensurePreviewActive = vi
      .spyOn(apiClient, 'ensureCameraPreviewActive')
      .mockResolvedValueOnce({
        camera_name: 'front',
        state: 'ready',
        viewer_count: 1,
        token: 'preview-token-1',
        token_expires_at: '2026-04-23T12:00:10.000Z',
        playlist_url: '/api/v1/preview/cameras/front/playlist.m3u8?token=preview-token-1',
        idle_timeout_s: 30,
        warning: null,
        httpStatus: 200,
      })
      .mockResolvedValueOnce({
        camera_name: 'front',
        state: 'ready',
        viewer_count: 1,
        token: 'preview-token-2',
        token_expires_at: '2026-04-23T12:01:10.000Z',
        playlist_url: '/api/v1/preview/cameras/front/playlist.m3u8?token=preview-token-2',
        idle_timeout_s: 30,
        warning: null,
        httpStatus: 200,
      })

    const { result } = renderHook(() => useCameraPreview('front'), {
      wrapper: createWrapper(),
    })

    await waitFor(() => {
      expect(result.current.status?.state).toBe('ready')
    })

    // When: Starting preview and advancing to the token-refresh deadline
    await act(async () => {
      await expect(result.current.start()).resolves.toBeUndefined()
    })

    expect(ensurePreviewActive).toHaveBeenCalledTimes(1)
    expect(refreshDelayMs).toBe(5_000)
    expect(runScheduledRefresh).not.toBeNull()
    expect(result.current.playlistUrl).toContain('preview-token-1')

    // When: Running the scheduled token-refresh callback
    await act(async () => {
      runScheduledRefresh?.()
      await Promise.resolve()
    })

    // Then: The hook refreshes the session before the original token expires
    await waitFor(() => {
      expect(ensurePreviewActive).toHaveBeenCalledTimes(2)
      expect(result.current.playlistUrl).toContain('preview-token-2')
    })
  })

  it('retries preview token renewal after a transient refresh failure', async () => {
    // Given: A ready preview session whose first renewal attempt fails transiently
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-04-23T12:00:00.000Z'))
    vi.spyOn(apiClient, 'getCameraPreviewStatus').mockResolvedValue({
      camera_name: 'front',
      enabled: true,
      state: 'ready',
      viewer_count: 1,
      degraded_reason: null,
      last_error: null,
      idle_shutdown_at: null,
      httpStatus: 200,
    })
    const ensurePreviewActive = vi
      .spyOn(apiClient, 'ensureCameraPreviewActive')
      .mockResolvedValueOnce({
        camera_name: 'front',
        state: 'ready',
        viewer_count: 1,
        token: 'preview-token-1',
        token_expires_at: '2026-04-23T12:00:10.000Z',
        playlist_url: '/api/v1/preview/cameras/front/playlist.m3u8?token=preview-token-1',
        idle_timeout_s: 30,
        warning: null,
        httpStatus: 200,
      })
      .mockRejectedValueOnce(new Error('token refresh failed'))
      .mockResolvedValueOnce({
        camera_name: 'front',
        state: 'ready',
        viewer_count: 1,
        token: 'preview-token-2',
        token_expires_at: '2026-04-23T12:01:10.000Z',
        playlist_url: '/api/v1/preview/cameras/front/playlist.m3u8?token=preview-token-2',
        idle_timeout_s: 30,
        warning: null,
        httpStatus: 200,
      })

    const { result } = renderHook(() => useCameraPreview('front'), {
      wrapper: createWrapper(),
    })

    // When: Starting preview and running the scheduled token refresh twice
    await act(async () => {
      await expect(result.current.start()).resolves.toBeUndefined()
    })

    expect(ensurePreviewActive).toHaveBeenCalledTimes(1)
    expect(result.current.playlistUrl).toContain('preview-token-1')

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000)
    })

    // Then: A failed renewal schedules a short retry instead of abandoning the session
    await act(async () => {
      await Promise.resolve()
    })
    expect(ensurePreviewActive).toHaveBeenCalledTimes(2)
    expect(result.current.error?.message).toBe('token refresh failed')
    expect(result.current.session?.token).toBe('preview-token-1')

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000)
    })

    // Then: A later retry can renew the token and update playback URL
    await act(async () => {
      await Promise.resolve()
    })
    expect(ensurePreviewActive).toHaveBeenCalledTimes(3)
    expect(result.current.playlistUrl).toContain('preview-token-2')
    expect(result.current.error).toBeNull()
  })

  it('keeps post-expiry token refresh retries bounded after a failed renewal', async () => {
    // Given: A ready preview session whose renewal fails after the current token has expired
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-04-23T12:00:00.000Z'))
    vi.spyOn(apiClient, 'getCameraPreviewStatus').mockResolvedValue({
      camera_name: 'front',
      enabled: true,
      state: 'ready',
      viewer_count: 1,
      degraded_reason: null,
      last_error: null,
      idle_shutdown_at: null,
      httpStatus: 200,
    })
    const ensurePreviewActive = vi
      .spyOn(apiClient, 'ensureCameraPreviewActive')
      .mockResolvedValueOnce({
        camera_name: 'front',
        state: 'ready',
        viewer_count: 1,
        token: 'preview-token-1',
        token_expires_at: '2026-04-23T12:00:10.000Z',
        playlist_url: '/api/v1/preview/cameras/front/playlist.m3u8?token=preview-token-1',
        idle_timeout_s: 30,
        warning: null,
        httpStatus: 200,
      })
      .mockImplementationOnce(async () => {
        vi.setSystemTime(new Date('2026-04-23T12:00:11.000Z'))
        throw new Error('token refresh failed')
      })
      .mockResolvedValueOnce({
        camera_name: 'front',
        state: 'ready',
        viewer_count: 1,
        token: 'preview-token-2',
        token_expires_at: '2026-04-23T12:01:11.000Z',
        playlist_url: '/api/v1/preview/cameras/front/playlist.m3u8?token=preview-token-2',
        idle_timeout_s: 30,
        warning: null,
        httpStatus: 200,
      })

    const { result } = renderHook(() => useCameraPreview('front'), {
      wrapper: createWrapper(),
    })

    // When: Starting preview and letting the scheduled renewal fail after expiry
    await act(async () => {
      await expect(result.current.start()).resolves.toBeUndefined()
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(5_000)
    })

    // Then: The failed renewal stays bounded instead of immediately spinning another request
    await act(async () => {
      await Promise.resolve()
    })
    expect(ensurePreviewActive).toHaveBeenCalledTimes(2)
    expect(result.current.error?.message).toBe('token refresh failed')
    expect(result.current.session?.token).toBe('preview-token-1')

    // When: Advancing almost the entire retry interval
    await act(async () => {
      await vi.advanceTimersByTimeAsync(999)
    })

    // Then: No extra retry fires early
    expect(ensurePreviewActive).toHaveBeenCalledTimes(2)

    // When: Completing the bounded retry delay
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1)
    })

    // Then: The next retry runs once and can recover the preview session
    await act(async () => {
      await Promise.resolve()
    })
    expect(ensurePreviewActive).toHaveBeenCalledTimes(3)
    expect(result.current.playlistUrl).toContain('preview-token-2')
    expect(result.current.error).toBeNull()
  })

  it('drops stale preview sessions after a newer terminal runtime status', async () => {
    // Given: A started preview session whose follow-up status says the runtime has already failed it
    freezePreviewClock()
    vi.spyOn(apiClient, 'getCameraPreviewStatus')
      .mockResolvedValueOnce({
        camera_name: 'front',
        enabled: true,
        state: 'idle',
        viewer_count: null,
        degraded_reason: null,
        last_error: null,
        idle_shutdown_at: null,
        httpStatus: 200,
      })
      .mockResolvedValueOnce({
        camera_name: 'front',
        enabled: true,
        state: 'error',
        viewer_count: 0,
        degraded_reason: null,
        last_error: 'runtime worker exited with code 137',
        idle_shutdown_at: null,
        httpStatus: 200,
      })
    const ensurePreviewActive = vi
      .spyOn(apiClient, 'ensureCameraPreviewActive')
      .mockResolvedValue({
        camera_name: 'front',
        state: 'ready',
        viewer_count: 1,
        token: 'preview-token-1',
        token_expires_at: '2026-04-23T12:00:10.000Z',
        playlist_url: '/api/v1/preview/cameras/front/playlist.m3u8?token=preview-token-1',
        idle_timeout_s: 30,
        warning: null,
        httpStatus: 200,
      })

    const { result } = renderHook(() => useCameraPreview('front'), {
      wrapper: createWrapper(),
    })

    await waitFor(() => {
      expect(result.current.status?.state).toBe('idle')
    })

    // When: Starting preview and allowing the invalidated status refetch to report a terminal error
    await act(async () => {
      await expect(result.current.start()).resolves.toBeUndefined()
    })

    // Then: The stale session is dropped and its token-renewal timer is cancelled
    await waitFor(() => {
      expect(result.current.status?.state).toBe('error')
      expect(result.current.session).toBeNull()
      expect(result.current.playlistUrl).toBeNull()
    })
    expect(ensurePreviewActive).toHaveBeenCalledTimes(1)
  })

  it('keeps a dropped preview session cleared across later status refreshes', async () => {
    // Given: A preview session invalidated by a newer terminal runtime status
    freezePreviewClock()
    vi.spyOn(apiClient, 'getCameraPreviewStatus')
      .mockResolvedValueOnce({
        camera_name: 'front',
        enabled: true,
        state: 'idle',
        viewer_count: null,
        degraded_reason: null,
        last_error: null,
        idle_shutdown_at: null,
        httpStatus: 200,
      })
      .mockResolvedValueOnce({
        camera_name: 'front',
        enabled: true,
        state: 'error',
        viewer_count: 0,
        degraded_reason: null,
        last_error: 'runtime worker exited with code 137',
        idle_shutdown_at: null,
        httpStatus: 200,
      })
      .mockResolvedValueOnce({
        camera_name: 'front',
        enabled: true,
        state: 'ready',
        viewer_count: 2,
        degraded_reason: null,
        last_error: null,
        idle_shutdown_at: null,
        httpStatus: 200,
      })
    const ensurePreviewActive = vi
      .spyOn(apiClient, 'ensureCameraPreviewActive')
      .mockResolvedValue({
        camera_name: 'front',
        state: 'ready',
        viewer_count: 1,
        token: 'preview-token-1',
        token_expires_at: '2026-04-23T12:00:10.000Z',
        playlist_url: '/api/v1/preview/cameras/front/playlist.m3u8?token=preview-token-1',
        idle_timeout_s: 30,
        warning: null,
        httpStatus: 200,
      })

    const { result } = renderHook(() => useCameraPreview('front'), {
      wrapper: createWrapper(),
    })

    await waitFor(() => {
      expect(result.current.status?.state).toBe('idle')
    })

    // When: Starting preview, dropping the stale session, then refreshing status again
    await act(async () => {
      await expect(result.current.start()).resolves.toBeUndefined()
    })

    await waitFor(() => {
      expect(result.current.status?.state).toBe('error')
      expect(result.current.session).toBeNull()
      expect(result.current.playlistUrl).toBeNull()
    })

    await act(async () => {
      await expect(result.current.refreshStatus()).resolves.toMatchObject({ state: 'ready' })
    })

    // Then: The hook keeps the stale session cleared until the user explicitly re-attaches
    await waitFor(() => {
      expect(result.current.status?.state).toBe('ready')
      expect(result.current.session).toBeNull()
      expect(result.current.playlistUrl).toBeNull()
    })
    expect(ensurePreviewActive).toHaveBeenCalledTimes(1)
  })

  it('stops active preview on native background and refreshes status on resume', async () => {
    // Given: An active native app preview session
    freezePreviewClock()
    const getPreviewStatus = vi.spyOn(apiClient, 'getCameraPreviewStatus').mockResolvedValue({
      camera_name: 'front',
      enabled: true,
      state: 'ready',
      viewer_count: 1,
      degraded_reason: null,
      last_error: null,
      idle_shutdown_at: null,
      httpStatus: 200,
    })
    vi.spyOn(apiClient, 'ensureCameraPreviewActive').mockResolvedValue({
      camera_name: 'front',
      state: 'ready',
      viewer_count: 1,
      token: 'preview-token-1',
      token_expires_at: '2026-04-23T12:00:10.000Z',
      playlist_url: '/api/v1/preview/cameras/front/playlist.m3u8?token=preview-token-1',
      idle_timeout_s: 30,
      warning: null,
      httpStatus: 200,
    })
    const stopPreview = vi.spyOn(apiClient, 'stopCameraPreview').mockResolvedValue({
      accepted: true,
      state: 'stopping',
      httpStatus: 202,
    })

    const { result, rerender } = renderHook(() => useCameraPreview('front'), {
      wrapper: createWrapper(),
    })
    await waitFor(() => {
      expect(result.current.status?.state).toBe('ready')
    })
    await act(async () => {
      await result.current.start()
    })
    await waitFor(() => {
      expect(result.current.session?.token).toBe('preview-token-1')
    })

    // When: iOS backgrounds the app while preview is attached
    setNativeLifecycleState({ isActive: false, isBackgrounded: true, pauseCount: 1 })
    rerender()

    // Then: The hook stops preview and suppresses background status requests
    await waitFor(() => {
      expect(stopPreview).toHaveBeenCalledWith('front')
      expect(result.current.session).toBeNull()
    })
    const statusCallsAfterPause = getPreviewStatus.mock.calls.length
    await act(async () => {
      await expect(result.current.refreshStatus()).resolves.toBeNull()
    })
    expect(getPreviewStatus).toHaveBeenCalledTimes(statusCallsAfterPause)

    // When: iOS resumes the app
    setNativeLifecycleState({ isActive: true, isBackgrounded: false, resumeCount: 1 })
    rerender()

    // Then: The hook refreshes preview status without auto-attaching a new session
    await waitFor(() => {
      expect(getPreviewStatus.mock.calls.length).toBeGreaterThan(statusCallsAfterPause)
    })
    expect(result.current.session).toBeNull()
  })

  it('stops preview start that resolves after native background', async () => {
    // Given: Preview start is still in flight when native iOS backgrounds the app
    const previewStart = deferred<Awaited<ReturnType<typeof apiClient.ensureCameraPreviewActive>>>()
    vi.spyOn(apiClient, 'getCameraPreviewStatus').mockResolvedValue({
      camera_name: 'front',
      enabled: true,
      state: 'ready',
      viewer_count: 0,
      degraded_reason: null,
      last_error: null,
      idle_shutdown_at: null,
      httpStatus: 200,
    })
    vi.spyOn(apiClient, 'ensureCameraPreviewActive').mockReturnValue(previewStart.promise)
    const stopPreview = vi.spyOn(apiClient, 'stopCameraPreview').mockResolvedValue({
      accepted: true,
      state: 'stopping',
      httpStatus: 202,
    })

    const { result, rerender } = renderHook(() => useCameraPreview('front'), {
      wrapper: createWrapper(),
    })
    await waitFor(() => {
      expect(result.current.status?.state).toBe('ready')
    })

    let startPromise!: Promise<void>
    act(() => {
      startPromise = result.current.start()
    })
    await waitFor(() => {
      expect(apiClient.ensureCameraPreviewActive).toHaveBeenCalledWith('front')
    })

    // When: The app backgrounds before the preview start response resolves
    setNativeLifecycleState({ isActive: false, isBackgrounded: true, pauseCount: 1 })
    rerender()
    await act(async () => {
      previewStart.resolve({
        camera_name: 'front',
        state: 'ready',
        viewer_count: 1,
        token: 'preview-token-1',
        token_expires_at: '2026-04-23T12:00:10.000Z',
        playlist_url: '/api/v1/preview/cameras/front/playlist.m3u8?token=preview-token-1',
        idle_timeout_s: 30,
        warning: null,
        httpStatus: 200,
      })
      await startPromise
    })

    // Then: The late preview session is stopped instead of being stored while backgrounded
    await waitFor(() => {
      expect(stopPreview).toHaveBeenCalledWith('front')
      expect(result.current.session).toBeNull()
      expect(result.current.playlistUrl).toBeNull()
    })
  })

  it('does not let a stale preview start stop a newer resumed preview', async () => {
    // Given: A preview start begins before background and a newer start succeeds after resume
    const staleStart = deferred<Awaited<ReturnType<typeof apiClient.ensureCameraPreviewActive>>>()
    vi.spyOn(apiClient, 'getCameraPreviewStatus').mockResolvedValue({
      camera_name: 'front',
      enabled: true,
      state: 'ready',
      viewer_count: 0,
      degraded_reason: null,
      last_error: null,
      idle_shutdown_at: null,
      httpStatus: 200,
    })
    vi.spyOn(apiClient, 'ensureCameraPreviewActive')
      .mockReturnValueOnce(staleStart.promise)
      .mockResolvedValueOnce({
        camera_name: 'front',
        state: 'ready',
        viewer_count: 1,
        token: 'preview-token-new',
        token_expires_at: '2026-04-23T12:00:10.000Z',
        playlist_url: '/api/v1/preview/cameras/front/playlist.m3u8?token=preview-token-new',
        idle_timeout_s: 30,
        warning: null,
        httpStatus: 200,
      })
    const stopPreview = vi.spyOn(apiClient, 'stopCameraPreview').mockResolvedValue({
      accepted: true,
      state: 'stopping',
      httpStatus: 202,
    })

    const { result, rerender } = renderHook(() => useCameraPreview('front'), {
      wrapper: createWrapper(),
    })
    await waitFor(() => {
      expect(result.current.status?.state).toBe('ready')
    })

    let staleStartPromise!: Promise<void>
    act(() => {
      staleStartPromise = result.current.start()
    })
    await waitFor(() => {
      expect(apiClient.ensureCameraPreviewActive).toHaveBeenCalledTimes(1)
    })

    setNativeLifecycleState({ isActive: false, isBackgrounded: true, pauseCount: 1 })
    rerender()
    setNativeLifecycleState({ isActive: true, isBackgrounded: false, resumeCount: 1 })
    rerender()
    await act(async () => {
      await result.current.start()
    })
    await waitFor(() => {
      expect(result.current.session?.token).toBe('preview-token-new')
    })

    // When: The stale pre-background start resolves after the newer resumed start
    await act(async () => {
      staleStart.resolve({
        camera_name: 'front',
        state: 'ready',
        viewer_count: 1,
        token: 'preview-token-stale',
        token_expires_at: '2026-04-23T12:00:10.000Z',
        playlist_url: '/api/v1/preview/cameras/front/playlist.m3u8?token=preview-token-stale',
        idle_timeout_s: 30,
        warning: null,
        httpStatus: 200,
      })
      await staleStartPromise
    })

    // Then: The stale completion is ignored without clearing or stopping the newer session
    expect(stopPreview).not.toHaveBeenCalled()
    expect(result.current.session?.token).toBe('preview-token-new')
    expect(result.current.playlistUrl).toContain('preview-token-new')
  })

  it('does not stop preview on transient native inactive transitions before background', async () => {
    // Given: Preview is attached while iOS is active
    vi.spyOn(apiClient, 'getCameraPreviewStatus').mockResolvedValue({
      camera_name: 'front',
      enabled: true,
      state: 'ready',
      viewer_count: 1,
      degraded_reason: null,
      last_error: null,
      idle_shutdown_at: null,
      httpStatus: 200,
    })
    vi.spyOn(apiClient, 'ensureCameraPreviewActive').mockResolvedValue({
      camera_name: 'front',
      state: 'ready',
      viewer_count: 1,
      token: 'preview-token-1',
      token_expires_at: '2026-04-23T12:00:10.000Z',
      playlist_url: '/api/v1/preview/cameras/front/playlist.m3u8?token=preview-token-1',
      idle_timeout_s: 30,
      warning: null,
      httpStatus: 200,
    })
    const stopPreview = vi.spyOn(apiClient, 'stopCameraPreview').mockResolvedValue({
      accepted: true,
      state: 'stopping',
      httpStatus: 202,
    })

    const { result, rerender } = renderHook(() => useCameraPreview('front'), {
      wrapper: createWrapper(),
    })
    await waitFor(() => {
      expect(result.current.status?.state).toBe('ready')
    })
    await act(async () => {
      await result.current.start()
    })
    await waitFor(() => {
      expect(result.current.session?.token).toBe('preview-token-1')
    })

    // When: iOS becomes inactive without the pause/background event
    setNativeLifecycleState({ isActive: false, isBackgrounded: false })
    rerender()
    await act(async () => {
      await Promise.resolve()
    })

    // Then: Preview remains attached until a real background pause is observed
    expect(stopPreview).not.toHaveBeenCalled()
    expect(result.current.session?.token).toBe('preview-token-1')
  })
})
