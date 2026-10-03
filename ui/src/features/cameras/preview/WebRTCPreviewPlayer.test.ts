// @vitest-environment happy-dom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { apiClient, type PreviewSessionSnapshot } from '../../../api/client'
import { WebRTCPreviewPlayer } from './WebRTCPreviewPlayer'

const SESSION: PreviewSessionSnapshot = {
  camera_name: 'front', state: 'ready', viewer_count: 0, token: 'token-1',
  token_expires_at: '2026-10-03T12:01:00Z', transport: 'webrtc', playlist_url: null,
  signaling_url: '/api/v1/preview/cameras/front/sessions', idle_timeout_s: 30,
  warning: null, ice_servers: [{ urls: ['stun:example.com'] }], httpStatus: 200,
}

class FakePeer extends EventTarget {
  static peers: FakePeer[] = []
  static gatherImmediately = true
  connectionState: RTCPeerConnectionState = 'new'
  iceConnectionState: RTCIceConnectionState = 'new'
  iceGatheringState: RTCIceGatheringState = 'new'
  localDescription: RTCSessionDescriptionInit | null = null
  ontrack: ((event: { track: MediaStreamTrack }) => void) | null = null
  onconnectionstatechange: (() => void) | null = null
  oniceconnectionstatechange: (() => void) | null = null
  readonly addTransceiver = vi.fn()
  readonly close = vi.fn(() => { this.connectionState = 'closed' })
  readonly setRemoteDescription = vi.fn().mockResolvedValue(undefined)
  readonly createOffer = vi.fn().mockResolvedValue({ type: 'offer', sdp: 'offer-before-ice' })
  readonly config: RTCConfiguration

  constructor(config: RTCConfiguration) {
    super()
    this.config = config
    FakePeer.peers.push(this)
  }

  async setLocalDescription(description: RTCSessionDescriptionInit): Promise<void> {
    this.localDescription = description
    this.iceGatheringState = FakePeer.gatherImmediately ? 'complete' : 'gathering'
    if (FakePeer.gatherImmediately) {
      this.localDescription = { type: 'offer', sdp: 'offer-with-candidates' }
    }
  }

  completeGathering(): void {
    this.localDescription = { type: 'offer', sdp: 'offer-with-candidates' }
    this.iceGatheringState = 'complete'
    this.dispatchEvent(new Event('icegatheringstatechange'))
  }

  transition(state: RTCPeerConnectionState): void {
    this.connectionState = state
    this.onconnectionstatechange?.()
  }
}

class FakeStream {
  private readonly tracks: MediaStreamTrack[] = []
  addTrack(track: MediaStreamTrack): void { this.tracks.push(track) }
  getTracks(): MediaStreamTrack[] { return this.tracks }
}

function setup() {
  const video = document.createElement('video')
  Object.defineProperty(video, 'srcObject', { configurable: true, writable: true, value: null })
  const onState = vi.fn()
  const player = new WebRTCPreviewPlayer(SESSION, video, onState)
  player.start()
  return { video, player, onState }
}

describe('WebRTCPreviewPlayer', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-10-03T12:00:00Z'))
    FakePeer.peers = []
    FakePeer.gatherImmediately = true
    vi.stubGlobal('RTCPeerConnection', FakePeer)
    vi.stubGlobal('MediaStream', FakeStream)
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined)
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {})
    vi.spyOn(apiClient, 'createCameraPreviewPeer').mockResolvedValue({
      session_id: 'peer-1', type: 'answer', sdp: 'answer-with-candidates', httpStatus: 201,
    })
    vi.spyOn(apiClient, 'renewCameraPreviewPeer').mockResolvedValue({ accepted: true, httpStatus: 200 })
    vi.spyOn(apiClient, 'closeCameraPreviewPeer').mockResolvedValue({ accepted: true, httpStatus: 200 })
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  it('signals only after ICE gathering completes and plays received tracks muted', async () => {
    // Given: A browser peer gathering candidates for a camera-scoped session
    FakePeer.gatherImmediately = false
    const { player, video, onState } = setup()
    await vi.advanceTimersByTimeAsync(0)
    const peer = FakePeer.peers[0]!
    expect(apiClient.createCameraPreviewPeer).not.toHaveBeenCalled()

    // When: Candidate gathering finishes and media arrives
    peer.completeGathering()
    await vi.advanceTimersByTimeAsync(0)
    const track = { stop: vi.fn() } as unknown as MediaStreamTrack
    peer.ontrack?.({ track })
    peer.transition('connected')

    // Then: The complete offer is signaled, only receive tracks are requested, and playback is muted
    expect(apiClient.createCameraPreviewPeer).toHaveBeenCalledWith(
      'front', 'token-1', { type: 'offer', sdp: 'offer-with-candidates' }, { signal: expect.any(AbortSignal) },
    )
    expect(peer.addTransceiver.mock.calls).toEqual([
      ['video', { direction: 'recvonly' }], ['audio', { direction: 'recvonly' }],
    ])
    expect(peer.setRemoteDescription).toHaveBeenCalledWith({ type: 'answer', sdp: 'answer-with-candidates' })
    expect(video.srcObject).toBeTruthy()
    expect(video.muted).toBe(true)
    expect(onState).toHaveBeenLastCalledWith({ state: 'connected', error: null })
    player.close()
    expect(track.stop).toHaveBeenCalledOnce()
  })

  it('bounds failed gathering and retries before reporting a recoverable error', async () => {
    // Given: A browser that never finishes gathering candidates
    FakePeer.gatherImmediately = false
    const { player, onState } = setup()

    // When: All three bounded connection attempts exhaust their gathering deadlines
    await vi.advanceTimersByTimeAsync(32_000)

    // Then: Each peer closes, no incomplete offer is sent, and automatic retries stop
    expect(FakePeer.peers).toHaveLength(3)
    expect(FakePeer.peers.every((peer) => peer.close.mock.calls.length === 1)).toBe(true)
    expect(apiClient.createCameraPreviewPeer).not.toHaveBeenCalled()
    expect(onState).toHaveBeenLastCalledWith({ state: 'error', error: 'Live view connection failed. Restart preview.' })
    await vi.advanceTimersByTimeAsync(60_000)
    expect(FakePeer.peers).toHaveLength(3)
    player.close()
  })

  it('renews authorization on the existing peer without a new offer', async () => {
    // Given: An established viewer with a camera-scoped token
    const { player } = setup()
    await vi.advanceTimersByTimeAsync(0)
    FakePeer.peers[0]!.transition('connected')

    // When: The preview hook supplies a refreshed token
    player.updateSession({ ...SESSION, token: 'token-2', token_expires_at: '2026-10-03T12:02:00Z' })
    await vi.advanceTimersByTimeAsync(0)

    // Then: Only the existing server peer lease is renewed
    expect(apiClient.renewCameraPreviewPeer).toHaveBeenCalledWith('front', 'peer-1', 'token-2', { signal: expect.any(AbortSignal) })
    expect(apiClient.createCameraPreviewPeer).toHaveBeenCalledOnce()
    expect(FakePeer.peers).toHaveLength(1)
    player.close()
    expect(apiClient.closeCameraPreviewPeer).toHaveBeenCalledWith('front', 'peer-1', 'token-2')
  })

  it('cleans the viewer after a sustained disconnect and allows a bounded reconnect', async () => {
    // Given: A connected viewer
    const { player } = setup()
    await vi.advanceTimersByTimeAsync(0)
    const peer = FakePeer.peers[0]!
    peer.transition('connected')

    // When: The connection remains disconnected beyond the grace period
    peer.transition('disconnected')
    await vi.advanceTimersByTimeAsync(3_000)

    // Then: The old peer closes and a replacement connects through signaling
    expect(peer.close).toHaveBeenCalledOnce()
    expect(apiClient.closeCameraPreviewPeer).toHaveBeenCalledWith('front', 'peer-1', 'token-1')
    expect(FakePeer.peers).toHaveLength(2)
    player.close()
  })

  it('retains the existing peer after a transient disconnect', async () => {
    // Given: An established connection that briefly loses connectivity
    const { player } = setup()
    await vi.advanceTimersByTimeAsync(0)
    const peer = FakePeer.peers[0]!
    peer.transition('connected')

    // When: Connectivity returns within the grace period
    peer.transition('disconnected')
    await vi.advanceTimersByTimeAsync(500)
    peer.transition('connected')
    await vi.advanceTimersByTimeAsync(3_000)

    // Then: No replacement or cleanup is needed
    expect(FakePeer.peers).toHaveLength(1)
    expect(peer.close).not.toHaveBeenCalled()
    player.close()
  })

  it('closes a late answer after stop without attaching it or retrying', async () => {
    // Given: Signaling is pending while the browser has an open peer
    let answer: ((value: Awaited<ReturnType<typeof apiClient.createCameraPreviewPeer>>) => void) | undefined
    vi.mocked(apiClient.createCameraPreviewPeer).mockImplementation(() => new Promise((resolve) => { answer = resolve }))
    const { player, video } = setup()
    await vi.advanceTimersByTimeAsync(0)
    const peer = FakePeer.peers[0]!

    // When: The viewer stops before a delayed answer arrives
    player.close()
    answer!({ session_id: 'late-peer', type: 'answer', sdp: 'late-answer', httpStatus: 201 })
    await vi.advanceTimersByTimeAsync(60_000)

    // Then: Server and browser cleanup happen without applying the answer or restarting
    expect(peer.close).toHaveBeenCalledOnce()
    expect(peer.setRemoteDescription).not.toHaveBeenCalled()
    expect(apiClient.closeCameraPreviewPeer).toHaveBeenCalledWith('front', 'late-peer', 'token-1')
    expect(video.srcObject).toBeNull()
    expect(FakePeer.peers).toHaveLength(1)
  })

  it('renews a token that changed while the offer was in flight', async () => {
    // Given: A pending offer sent with the original token
    let answer: ((value: Awaited<ReturnType<typeof apiClient.createCameraPreviewPeer>>) => void) | undefined
    vi.mocked(apiClient.createCameraPreviewPeer).mockImplementation(() => new Promise((resolve) => { answer = resolve }))
    const { player } = setup()
    await vi.advanceTimersByTimeAsync(0)

    // When: Authorization refresh precedes the server answer
    player.updateSession({ ...SESSION, token: 'token-2' })
    answer!({ session_id: 'peer-1', type: 'answer', sdp: 'answer', httpStatus: 201 })
    await vi.advanceTimersByTimeAsync(0)

    // Then: The newly created peer receives the refreshed lease without another negotiation
    expect(apiClient.renewCameraPreviewPeer).toHaveBeenCalledWith('front', 'peer-1', 'token-2', { signal: expect.any(AbortSignal) })
    expect(apiClient.createCameraPreviewPeer).toHaveBeenCalledOnce()
    player.close()
  })

  it('connects and renews a bounded lease when server authentication is disabled', async () => {
    // Given: A browser attachment with no token and an explicit server lease deadline
    const video = document.createElement('video')
    Object.defineProperty(video, 'srcObject', { configurable: true, writable: true, value: null })
    const session = { ...SESSION, token: null, token_expires_at: null, lease_expires_at: '2026-10-03T12:01:00Z' }
    const player = new WebRTCPreviewPlayer(session, video, vi.fn())

    // When: The tokenless viewer connects and receives a refreshed lease snapshot
    player.start()
    await vi.advanceTimersByTimeAsync(0)
    FakePeer.peers[0]!.transition('connected')
    player.updateSession({ ...session, lease_expires_at: '2026-10-03T12:02:00Z' })
    await vi.advanceTimersByTimeAsync(0)
    player.close()

    // Then: Normal signaling, renewal, and viewer cleanup work without a fabricated token
    expect(apiClient.createCameraPreviewPeer).toHaveBeenCalledWith(
      'front', null, { type: 'offer', sdp: 'offer-with-candidates' }, { signal: expect.any(AbortSignal) },
    )
    expect(apiClient.renewCameraPreviewPeer).toHaveBeenCalledWith('front', 'peer-1', null, { signal: expect.any(AbortSignal) })
    expect(apiClient.closeCameraPreviewPeer).toHaveBeenCalledWith('front', 'peer-1', null)
    expect(FakePeer.peers).toHaveLength(1)
  })

  it('recovers an exhausted connection budget when fresh authorization arrives', async () => {
    // Given: Three failed attempts exhausted the original descriptor's retry budget
    vi.mocked(apiClient.createCameraPreviewPeer).mockRejectedValue(new Error('authorization rejected'))
    const { player, onState } = setup()
    await vi.advanceTimersByTimeAsync(3_000)
    expect(onState).toHaveBeenLastCalledWith({ state: 'error', error: 'Live view connection failed. Restart preview.' })

    // When: A fresh descriptor arrives after the failures
    vi.mocked(apiClient.createCameraPreviewPeer).mockResolvedValue({ session_id: 'fresh-peer', type: 'answer', sdp: 'answer', httpStatus: 201 })
    player.updateSession({ ...SESSION, token: 'fresh-token', lease_expires_at: '2026-10-03T12:02:00Z' })
    await vi.advanceTimersByTimeAsync(0)

    // Then: A single new attempt uses fresh authorization and can establish playback
    expect(FakePeer.peers).toHaveLength(4)
    expect(apiClient.createCameraPreviewPeer).toHaveBeenLastCalledWith(
      'front', 'fresh-token', { type: 'offer', sdp: 'offer-with-candidates' }, { signal: expect.any(AbortSignal) },
    )
    FakePeer.peers[3]!.transition('connected')
    expect(onState).toHaveBeenLastCalledWith({ state: 'connected', error: null })
    player.close()
  })
})
