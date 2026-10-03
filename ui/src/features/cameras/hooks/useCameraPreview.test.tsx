// @vitest-environment happy-dom

import type { PropsWithChildren } from 'react'
import { act, renderHook, waitFor } from '@testing-library/react'
import { onlineManager, QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { apiClient } from '../../../api/client'
import { useCameraPreview } from './useCameraPreview'

const PREVIEW_TEST_NOW_MS = Date.parse('2026-04-23T12:00:00.000Z')

function freezePreviewClock() {
  vi.spyOn(Date, 'now').mockReturnValue(PREVIEW_TEST_NOW_MS)
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
    vi.restoreAllMocks()
    vi.useRealTimers()
    onlineManager.setOnline(true)
  })

  it('stops a WebRTC viewer locally without force-stopping the camera publisher', async () => {
    // Given: A camera with another viewer and an active WebRTC attachment
    vi.spyOn(apiClient, 'getCameraPreviewStatus').mockResolvedValue({
      camera_name: 'front', enabled: true, state: 'ready', viewer_count: 2,
      degraded_reason: null, last_error: null, idle_shutdown_at: null, httpStatus: 200,
    })
    const visibility = vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('visible')
    const ensure = vi.spyOn(apiClient, 'ensureCameraPreviewActive').mockResolvedValue({
      camera_name: 'front', state: 'ready', viewer_count: 2, transport: 'webrtc',
      token: 'preview-token', token_expires_at: null, playlist_url: null,
      lease_expires_at: new Date(Date.now() + 60_000).toISOString(),
      signaling_url: '/api/v1/preview/cameras/front/sessions', ice_servers: [],
      idle_timeout_s: 30, warning: null, httpStatus: 200,
    })
    const forceStop = vi.spyOn(apiClient, 'stopCameraPreview')
    const { result, unmount } = renderHook(() => useCameraPreview('front'), { wrapper: createWrapper() })
    await act(async () => { await result.current.start() })
    await waitFor(() => expect(result.current.session?.transport).toBe('webrtc'))

    // When: This viewer stops while hidden and later returns to the foreground
    await act(async () => {
      visibility.mockReturnValue('hidden')
      document.dispatchEvent(new Event('visibilitychange'))
      await result.current.stop()
    })
    act(() => {
      visibility.mockReturnValue('visible')
      document.dispatchEvent(new Event('visibilitychange'))
    })

    // Then: Attachment removal triggers player cleanup while the shared publisher stays active
    expect(result.current.session).toBeNull()
    expect(result.current.playlistUrl).toBeNull()
    expect(forceStop).not.toHaveBeenCalled()
    expect(ensure).toHaveBeenCalledOnce()
    unmount()
  })

  it('does not reactivate WebRTC media through token refresh while backgrounded', async () => {
    // Given: A WebRTC attachment whose token refresh is due shortly
    freezePreviewClock()
    const visibility = vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('visible')
    vi.spyOn(apiClient, 'getCameraPreviewStatus').mockResolvedValue({
      camera_name: 'front', enabled: true, state: 'ready', viewer_count: 1,
      degraded_reason: null, last_error: null, idle_shutdown_at: null, httpStatus: 200,
    })
    const ensure = vi.spyOn(apiClient, 'ensureCameraPreviewActive').mockResolvedValue({
      camera_name: 'front', state: 'ready', viewer_count: 1, transport: 'webrtc',
      token: 'preview-token', token_expires_at: '2026-04-23T12:00:10.000Z', playlist_url: null,
      signaling_url: '/api/v1/preview/cameras/front/sessions', ice_servers: [],
      idle_timeout_s: 30, warning: null, httpStatus: 200,
    })
    const { result, unmount } = renderHook(() => useCameraPreview('front'), { wrapper: createWrapper() })
    await act(async () => { await result.current.start() })
    await waitFor(() => expect(result.current.session?.transport).toBe('webrtc'))
    vi.useFakeTimers()

    // When: The app remains hidden past the scheduled authorization refresh
    act(() => {
      visibility.mockReturnValue('hidden')
      document.dispatchEvent(new Event('visibilitychange'))
    })
    await act(async () => { await vi.advanceTimersByTimeAsync(60_000) })

    // Then: Camera activation remains idle until returning to the foreground
    expect(ensure).toHaveBeenCalledOnce()
    await act(async () => {
      visibility.mockReturnValue('visible')
      document.dispatchEvent(new Event('visibilitychange'))
    })
    expect(ensure).toHaveBeenCalledTimes(2)
    unmount()
  })

  it('preserves WebRTC viewing intent across a hidden reconnect and resumes on foreground', async () => {
    // Given: An attached WebRTC viewer with a bounded lease and real query reconnect handling
    vi.useFakeTimers()
    vi.setSystemTime(new Date(PREVIEW_TEST_NOW_MS))
    const visibility = vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('visible')
    const status = vi.spyOn(apiClient, 'getCameraPreviewStatus').mockResolvedValue({
      camera_name: 'front', enabled: true, state: 'ready', viewer_count: 1,
      degraded_reason: null, last_error: null, idle_shutdown_at: null, httpStatus: 200,
    })
    const ensure = vi.spyOn(apiClient, 'ensureCameraPreviewActive').mockResolvedValue({
      camera_name: 'front', state: 'ready', viewer_count: 1, transport: 'webrtc',
      token: 'preview-token', token_expires_at: null, lease_expires_at: '2026-04-23T12:01:00.000Z',
      playlist_url: null, signaling_url: '/api/v1/preview/cameras/front/sessions', ice_servers: [],
      idle_timeout_s: 30, warning: null, httpStatus: 200,
    })
    const { result, unmount } = renderHook(() => useCameraPreview('front'), { wrapper: createWrapper() })
    await act(async () => { await result.current.start(); await vi.advanceTimersByTimeAsync(0) })
    expect(result.current.session?.transport).toBe('webrtc')

    // When: Media idles out while hidden, then a network reconnect refetches status
    act(() => {
      visibility.mockReturnValue('hidden')
      document.dispatchEvent(new Event('visibilitychange'))
    })
    await act(async () => { await vi.advanceTimersByTimeAsync(120_000) })
    status.mockResolvedValue({
      camera_name: 'front', enabled: true, state: 'idle', viewer_count: 0,
      degraded_reason: null, last_error: null, idle_shutdown_at: null, httpStatus: 200,
    })
    const requestsBeforeReconnect = status.mock.calls.length
    await act(async () => {
      onlineManager.setOnline(false)
      onlineManager.setOnline(true)
      await vi.advanceTimersByTimeAsync(0)
    })
    const attachedAfterReconnect = result.current.session !== null
    expect(status.mock.calls.length).toBeGreaterThan(requestsBeforeReconnect)
    expect(ensure).toHaveBeenCalledOnce()
    ensure.mockResolvedValue({
      camera_name: 'front', state: 'ready', viewer_count: 0, transport: 'webrtc',
      token: 'fresh-token', token_expires_at: null, lease_expires_at: '2026-04-23T12:03:00.000Z',
      playlist_url: null, signaling_url: '/api/v1/preview/cameras/front/sessions', ice_servers: [],
      idle_timeout_s: 30, warning: null, httpStatus: 200,
    })
    status.mockResolvedValue({
      camera_name: 'front', enabled: true, state: 'ready', viewer_count: 0,
      degraded_reason: null, last_error: null, idle_shutdown_at: null, httpStatus: 200,
    })
    await act(async () => {
      visibility.mockReturnValue('visible')
      document.dispatchEvent(new Event('visibilitychange'))
      await vi.advanceTimersByTimeAsync(0)
    })

    // Then: Viewing intent survives without activating hidden media and foreground refreshes once
    expect(attachedAfterReconnect).toBe(true)
    expect(ensure).toHaveBeenCalledTimes(2)
    expect(result.current.session?.token).toBe('fresh-token')
    unmount()
  })

  it.each([
    { transport: 'hls' as const, disabled: false },
    { transport: 'webrtc' as const, disabled: true },
  ])('clears hidden $transport viewing intent when idle or disabled as appropriate', async ({ transport, disabled }) => {
    // Given: An attached viewer whose status can change while the page is hidden
    freezePreviewClock()
    const visibility = vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('visible')
    const status = vi.spyOn(apiClient, 'getCameraPreviewStatus').mockResolvedValue({
      camera_name: 'front', enabled: true, state: 'ready', viewer_count: 1,
      degraded_reason: null, last_error: null, idle_shutdown_at: null, httpStatus: 200,
    })
    const ensure = vi.spyOn(apiClient, 'ensureCameraPreviewActive').mockResolvedValue({
      camera_name: 'front', state: 'ready', viewer_count: 1, transport,
      token: 'preview-token', token_expires_at: '2026-04-23T12:01:00.000Z',
      lease_expires_at: transport === 'webrtc' ? '2026-04-23T12:01:00.000Z' : null,
      playlist_url: transport === 'hls' ? '/playlist.m3u8' : null,
      signaling_url: transport === 'webrtc' ? '/sessions' : null, ice_servers: [],
      idle_timeout_s: 30, warning: null, httpStatus: 200,
    })
    const { result, unmount } = renderHook(() => useCameraPreview('front'), { wrapper: createWrapper() })
    await act(async () => { await result.current.start() })

    // When: A hidden status refresh reports idle HLS or explicitly disabled WebRTC
    act(() => {
      visibility.mockReturnValue('hidden')
      document.dispatchEvent(new Event('visibilitychange'))
    })
    status.mockResolvedValue({
      camera_name: 'front', enabled: !disabled, state: 'idle', viewer_count: 0,
      degraded_reason: null, last_error: null, idle_shutdown_at: null, httpStatus: 200,
    })
    await act(async () => { await result.current.refreshStatus() })
    act(() => {
      visibility.mockReturnValue('visible')
      document.dispatchEvent(new Event('visibilitychange'))
    })

    // Then: The attachment clears and foregrounding does not restart it
    expect(result.current.session).toBeNull()
    expect(ensure).toHaveBeenCalledOnce()
    unmount()
  })

  it('refreshes tokenless WebRTC sessions from their lease deadline', async () => {
    // Given: An authentication-disabled server with a short viewer lease
    freezePreviewClock()
    const setTimeout = window.setTimeout.bind(window)
    let refresh: (() => void) | undefined
    vi.spyOn(window, 'setTimeout').mockImplementation((handler, timeout, ...args) => {
      if (timeout === 5_000 && typeof handler === 'function') {
        refresh = () => handler(...args)
      }
      return setTimeout(handler, timeout, ...args)
    })
    vi.spyOn(apiClient, 'getCameraPreviewStatus').mockResolvedValue({
      camera_name: 'front', enabled: true, state: 'ready', viewer_count: 1,
      degraded_reason: null, last_error: null, idle_shutdown_at: null, httpStatus: 200,
    })
    const ensure = vi.spyOn(apiClient, 'ensureCameraPreviewActive').mockResolvedValue({
      camera_name: 'front', state: 'ready', viewer_count: 1, transport: 'webrtc',
      token: null, token_expires_at: null, lease_expires_at: '2026-04-23T12:00:10.000Z',
      playlist_url: null, signaling_url: '/api/v1/preview/cameras/front/sessions', ice_servers: [],
      idle_timeout_s: 30, warning: null, httpStatus: 200,
    })
    const { result, unmount } = renderHook(() => useCameraPreview('front'), { wrapper: createWrapper() })
    await act(async () => { await result.current.start() })
    await waitFor(() => expect(result.current.session?.transport).toBe('webrtc'))
    expect(refresh).toBeDefined()

    // When: The refresh point precedes the lease deadline, despite there being no token
    await act(async () => { refresh!(); await Promise.resolve() })

    // Then: A fresh snapshot renews authorization without inventing a token
    expect(ensure).toHaveBeenCalledTimes(2)
    expect(result.current.session?.token).toBeNull()
    unmount()
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
      transport: 'hls',
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
        transport: 'hls',
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
        transport: 'hls',
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
        transport: 'hls',
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
        transport: 'hls',
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
        transport: 'hls',
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
        transport: 'hls',
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
        transport: 'hls',
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
        transport: 'hls',
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
        transport: 'hls',
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
})
