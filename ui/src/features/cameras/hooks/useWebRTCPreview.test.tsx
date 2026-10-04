// @vitest-environment happy-dom

import { act, renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { PropsWithChildren } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { apiClient, type PreviewSessionSnapshot } from '../../../api/client'
import { useWebRTCPreview } from './useWebRTCPreview'
import { useCameraPreview } from './useCameraPreview'

const SESSION: PreviewSessionSnapshot = {
  camera_name: 'front', state: 'ready', viewer_count: 0, token: 'token-1',
  token_expires_at: null, transport: 'webrtc', playlist_url: null,
  signaling_url: '/api/v1/preview/cameras/front/sessions', idle_timeout_s: 30,
  warning: null, ice_servers: [], httpStatus: 200,
}

class FakePeer extends EventTarget {
  static peers: FakePeer[] = []
  iceGatheringState = 'complete'
  localDescription = { type: 'offer', sdp: 'complete-offer' }
  readonly addTransceiver = vi.fn()
  readonly createOffer = vi.fn().mockResolvedValue(this.localDescription)
  readonly setLocalDescription = vi.fn().mockResolvedValue(undefined)
  readonly setRemoteDescription = vi.fn().mockResolvedValue(undefined)
  readonly close = vi.fn()
  constructor() { super(); FakePeer.peers.push(this) }
}

function videoRef() {
  const video = document.createElement('video')
  Object.defineProperty(video, 'srcObject', { configurable: true, writable: true, value: null })
  return { current: video }
}

describe('useWebRTCPreview', () => {
  beforeEach(() => {
    FakePeer.peers = []
    vi.stubGlobal('RTCPeerConnection', FakePeer)
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {})
    vi.spyOn(apiClient, 'createCameraPreviewPeer').mockImplementation(async (cameraName) => ({
      session_id: `${cameraName}-peer`, type: 'answer', sdp: 'answer', httpStatus: 201,
    }))
    vi.spyOn(apiClient, 'closeCameraPreviewPeer').mockResolvedValue({ accepted: true, httpStatus: 200 })
    vi.spyOn(apiClient, 'renewCameraPreviewPeer').mockResolvedValue({ accepted: true, httpStatus: 200 })
    vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('visible')
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  it('renews on token refresh and closes only the current viewer on stop and unmount', async () => {
    // Given: A mounted live viewer with a native peer connection
    const ref = videoRef()
    const { rerender, unmount } = renderHook(
      ({ session }: { session: PreviewSessionSnapshot | null }) => useWebRTCPreview(session, ref),
      { initialProps: { session: SESSION as PreviewSessionSnapshot | null } },
    )
    await waitFor(() => expect(apiClient.createCameraPreviewPeer).toHaveBeenCalledOnce())

    // When: A refreshed token arrives, followed by local stop
    rerender({ session: { ...SESSION, token: 'token-2' } })
    await waitFor(() => expect(apiClient.renewCameraPreviewPeer).toHaveBeenCalled())
    rerender({ session: null })
    unmount()

    // Then: Renewal keeps the peer and cleanup targets that viewer once
    expect(FakePeer.peers).toHaveLength(1)
    expect(FakePeer.peers[0]!.close).toHaveBeenCalledOnce()
    expect(apiClient.closeCameraPreviewPeer).toHaveBeenCalledExactlyOnceWith('front', 'front-peer', 'token-2')
  })

  it('closes on background and reconnects when visible without retaining the old peer', async () => {
    // Given: An active preview on a visible page
    const visibility = vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('visible')
    const ref = videoRef()
    const { unmount } = renderHook(() => useWebRTCPreview(SESSION, ref))
    await waitFor(() => expect(apiClient.createCameraPreviewPeer).toHaveBeenCalledOnce())

    // When: The app backgrounds and then returns to the foreground
    act(() => {
      visibility.mockReturnValue('hidden')
      document.dispatchEvent(new Event('visibilitychange'))
    })
    expect(FakePeer.peers[0]!.close).toHaveBeenCalledOnce()
    act(() => {
      visibility.mockReturnValue('visible')
      document.dispatchEvent(new Event('visibilitychange'))
    })
    await waitFor(() => expect(apiClient.createCameraPreviewPeer).toHaveBeenCalledTimes(2))

    // Then: A fresh viewer replaces the closed peer, and unmount closes it too
    expect(FakePeer.peers).toHaveLength(2)
    unmount()
    expect(FakePeer.peers[1]!.close).toHaveBeenCalledOnce()
  })

  it('uses the previous camera token for cleanup during a camera switch', async () => {
    // Given: A viewer attached to the front camera
    const ref = videoRef()
    const { rerender, unmount } = renderHook(({ session }) => useWebRTCPreview(session, ref),
      { initialProps: { session: SESSION } })
    await waitFor(() => expect(apiClient.createCameraPreviewPeer).toHaveBeenCalledOnce())

    // When: The component switches to another camera with its own token
    rerender({ session: { ...SESSION, camera_name: 'back', token: 'back-token', signaling_url: '/api/v1/preview/cameras/back/sessions' } })
    await waitFor(() => expect(apiClient.createCameraPreviewPeer).toHaveBeenCalledTimes(2))

    // Then: The old viewer is closed against the correct camera without cross-camera renewal
    expect(apiClient.closeCameraPreviewPeer).toHaveBeenCalledWith('front', 'front-peer', 'token-1')
    expect(apiClient.renewCameraPreviewPeer).not.toHaveBeenCalled()
    unmount()
    expect(apiClient.closeCameraPreviewPeer).toHaveBeenCalledWith('back', 'back-peer', 'back-token')
  })

  it('leaves HLS playback to the existing player', () => {
    // Given: A legacy HLS session and video element
    const ref = videoRef()

    // When: The shared preview hook receives the HLS transport
    const { result, unmount } = renderHook(() => useWebRTCPreview({ ...SESSION, transport: 'hls' }, ref))

    // Then: No browser peer or signaling request is created
    expect(result.current.state).toBe('closed')
    expect(FakePeer.peers).toHaveLength(0)
    expect(apiClient.createCameraPreviewPeer).not.toHaveBeenCalled()
    unmount()
  })

  it('reattaches with fresh authorization after delayed foreground activation and an idle status refetch', async () => {
    // Given: Both real preview hooks attached before a long background interval
    let clock = Date.parse('2026-10-03T12:00:00Z')
    vi.spyOn(Date, 'now').mockImplementation(() => clock)
    const visibility = vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('visible')
    const firstSession = { ...SESSION, lease_expires_at: new Date(clock + 60_000).toISOString() }
    const freshSession = { ...SESSION, token: 'fresh-token', lease_expires_at: new Date(clock + 180_000).toISOString() }
    let finishActivation: ((session: PreviewSessionSnapshot) => void) | undefined
    vi.spyOn(apiClient, 'ensureCameraPreviewActive')
      .mockResolvedValueOnce(firstSession)
      .mockImplementationOnce(() => new Promise((resolve) => { finishActivation = resolve }))
    const status = vi.spyOn(apiClient, 'getCameraPreviewStatus').mockResolvedValue({
      camera_name: 'front', enabled: true, state: 'ready', viewer_count: 1,
      degraded_reason: null, last_error: null, idle_shutdown_at: null, httpStatus: 200,
    })
    vi.mocked(apiClient.createCameraPreviewPeer).mockImplementation(async (_camera, token) => {
      if (clock >= Date.parse(firstSession.lease_expires_at) && token !== 'fresh-token') {
        throw new Error('expired authorization')
      }
      return { session_id: 'front-peer', type: 'answer', sdp: 'answer', httpStatus: 201 }
    })
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
    const wrapper = ({ children }: PropsWithChildren) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    const ref = videoRef()
    const { result, unmount } = renderHook(() => {
      const preview = useCameraPreview('front')
      const media = useWebRTCPreview(preview.session, ref)
      return { preview, media }
    }, { wrapper })
    await act(async () => { await result.current.preview.start() })
    await waitFor(() => expect(apiClient.createCameraPreviewPeer).toHaveBeenCalledOnce())
    vi.useFakeTimers()

    // When: Foregrounding occurs after expiry, with idle status racing a slow fresh activation
    act(() => {
      visibility.mockReturnValue('hidden')
      document.dispatchEvent(new Event('visibilitychange'))
    })
    clock += 120_000
    vi.setSystemTime(new Date(clock))
    status.mockResolvedValue({ camera_name: 'front', enabled: true, state: 'idle', viewer_count: 0,
      degraded_reason: null, last_error: null, idle_shutdown_at: null, httpStatus: 200 })
    act(() => {
      visibility.mockReturnValue('visible')
      document.dispatchEvent(new Event('visibilitychange'))
    })
    await act(async () => { await result.current.preview.refreshStatus() })
    await act(async () => { await vi.advanceTimersByTimeAsync(3_000) })
    expect(apiClient.createCameraPreviewPeer).toHaveBeenCalledOnce()
    expect(result.current.preview.session).not.toBeNull()
    status.mockResolvedValue({ camera_name: 'front', enabled: true, state: 'ready', viewer_count: 0,
      degraded_reason: null, last_error: null, idle_shutdown_at: null, httpStatus: 200 })
    await act(async () => {
      finishActivation!(freshSession)
      await vi.advanceTimersByTimeAsync(0)
    })

    // Then: The delayed descriptor attaches exactly one fresh peer without consuming stale retries
    expect(apiClient.createCameraPreviewPeer).toHaveBeenCalledTimes(2)
    expect(apiClient.createCameraPreviewPeer).toHaveBeenLastCalledWith(
      'front', 'fresh-token', expect.objectContaining({ type: 'offer' }), { signal: expect.any(AbortSignal) },
    )
    expect(FakePeer.peers).toHaveLength(2)
    expect(result.current.preview.session?.token).toBe('fresh-token')
    unmount()
  })
})
