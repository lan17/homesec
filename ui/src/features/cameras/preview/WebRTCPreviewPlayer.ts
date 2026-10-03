import { apiClient, type PreviewSessionSnapshot } from '../../../api/client'

const ICE_GATHER_TIMEOUT_MS = 10_000
const CONNECT_TIMEOUT_MS = 20_000
const DISCONNECT_GRACE_MS = 2_000
const RETRY_DELAY_MS = 1_000
const MAX_RETRIES = 2

export interface WebRTCPlayerState {
  state: 'connecting' | 'connected' | 'closed' | 'error'
  error: string | null
}

function waitForIceGathering(peer: RTCPeerConnection, signal: AbortSignal): Promise<void> {
  if (peer.iceGatheringState === 'complete') {
    return Promise.resolve()
  }
  return new Promise((resolve, reject) => {
    const finish = (error?: Error): void => {
      window.clearTimeout(timeout)
      peer.removeEventListener('icegatheringstatechange', onChange)
      signal.removeEventListener('abort', onAbort)
      if (error) {
        reject(error)
      } else {
        resolve()
      }
    }
    const onChange = (): void => {
      if (peer.iceGatheringState === 'complete') {
        finish()
      }
    }
    const onAbort = (): void => finish(new Error('Preview stopped'))
    const timeout = window.setTimeout(() => finish(new Error('ICE gathering timed out')), ICE_GATHER_TIMEOUT_MS)
    peer.addEventListener('icegatheringstatechange', onChange)
    signal.addEventListener('abort', onAbort, { once: true })
    if (signal.aborted) {
      onAbort()
    } else {
      onChange()
    }
  })
}

/** Browser media lifecycle only; signaling and authorization stay in the API client. */
export class WebRTCPreviewPlayer {
  private session: PreviewSessionSnapshot
  private readonly video: HTMLVideoElement
  private readonly onState: (state: WebRTCPlayerState) => void
  private peer: RTCPeerConnection | null = null
  private controller: AbortController | null = null
  private renewController: AbortController | null = null
  private stream: MediaStream | null = null
  private sessionId: string | null = null
  private disposed = false
  private retries = 0
  private connectTimeout: number | null = null
  private disconnectTimeout: number | null = null
  private retryTimeout: number | null = null

  constructor(
    session: PreviewSessionSnapshot,
    video: HTMLVideoElement,
    onState: (state: WebRTCPlayerState) => void,
  ) {
    this.session = session
    this.video = video
    this.onState = onState
  }

  start(): void {
    if (this.disposed || this.peer || this.retryTimeout !== null) {
      return
    }
    void this.connect()
  }

  updateSession(session: PreviewSessionSnapshot): void {
    if (session.camera_name !== this.session.camera_name || session.transport !== 'webrtc') {
      return
    }
    const previousToken = this.session.token
    const previousLease = this.session.lease_expires_at
    this.session = session
    const refreshed = session.token !== previousToken || session.lease_expires_at !== previousLease
    if (!this.disposed && refreshed && this.authorizationIsFresh()) {
      if (this.sessionId) {
        void this.renew(this.sessionId, session.token ?? null)
      } else if (!this.peer) {
        this.clearTimer('retryTimeout')
        this.retries = 0
        this.start()
      }
    }
  }

  close(): void {
    if (this.disposed) {
      return
    }
    this.disposed = true
    this.clearTimer('retryTimeout')
    this.cleanupPeer()
  }

  private clearTimer(name: 'connectTimeout' | 'disconnectTimeout' | 'retryTimeout'): void {
    const timer = this[name]
    if (timer !== null) {
      window.clearTimeout(timer)
      this[name] = null
    }
  }

  private cleanupPeer(): void {
    this.clearTimer('connectTimeout')
    this.clearTimer('disconnectTimeout')
    this.controller?.abort()
    this.controller = null
    this.renewController?.abort()
    this.renewController = null
    const peer = this.peer
    this.peer = null
    if (peer) {
      peer.ontrack = null
      peer.onconnectionstatechange = null
      peer.oniceconnectionstatechange = null
      peer.close()
    }
    this.stream?.getTracks().forEach((track) => track.stop())
    this.stream = null
    this.video.pause()
    this.video.srcObject = null
    const sessionId = this.sessionId
    this.sessionId = null
    if (sessionId) {
      this.deletePeer(sessionId)
    }
  }

  private deletePeer(sessionId: string): void {
    // Lease expiry bounds cleanup when the network or page is already gone.
    void apiClient.closeCameraPreviewPeer(this.session.camera_name, sessionId, this.session.token ?? null)
      .catch(() => {})
  }

  private fail(peer: RTCPeerConnection): void {
    if (this.disposed || this.peer !== peer) {
      return
    }
    this.cleanupPeer()
    if (this.retries < MAX_RETRIES) {
      this.retries += 1
      this.onState({ state: 'connecting', error: null })
      this.retryTimeout = window.setTimeout(() => {
        this.retryTimeout = null
        this.start()
      }, RETRY_DELAY_MS)
    } else {
      this.onState({ state: 'error', error: 'Live view connection failed. Restart preview.' })
    }
  }

  private async renew(sessionId: string, token: string | null): Promise<void> {
    this.renewController?.abort()
    const controller = new AbortController()
    this.renewController = controller
    const timeout = window.setTimeout(() => {
      if (this.renewController === controller && this.sessionId === sessionId && this.peer) {
        this.fail(this.peer)
      }
    }, CONNECT_TIMEOUT_MS)
    try {
      const result = await apiClient.renewCameraPreviewPeer(
        this.session.camera_name, sessionId, token, { signal: controller.signal },
      )
      if (!result.accepted && this.renewController === controller && this.sessionId === sessionId && this.peer) {
        this.fail(this.peer)
      }
    } catch {
      if (!controller.signal.aborted && this.renewController === controller && this.sessionId === sessionId && this.peer) {
        this.fail(this.peer)
      }
    } finally {
      window.clearTimeout(timeout)
      if (this.renewController === controller) {
        this.renewController = null
      }
    }
  }

  private async connect(): Promise<void> {
    if (typeof RTCPeerConnection === 'undefined') {
      this.onState({ state: 'error', error: 'This browser cannot connect to live view.' })
      return
    }
    this.onState({ state: 'connecting', error: null })
    if (!this.authorizationIsFresh()) {
      // Foreground activation refreshes the descriptor. Expired credentials
      // must not consume the bounded connection retry budget while it is pending.
      return
    }
    let peer: RTCPeerConnection
    try {
      peer = new RTCPeerConnection({
        iceServers: (this.session.ice_servers ?? []).map((server) => ({
          urls: server.urls,
          ...(server.username ? { username: server.username } : {}),
          ...(server.credential ? { credential: server.credential } : {}),
        })),
      })
    } catch {
      this.onState({ state: 'error', error: 'This browser cannot connect to live view.' })
      return
    }
    this.peer = peer
    const controller = new AbortController()
    this.controller = controller
    this.connectTimeout = window.setTimeout(() => this.fail(peer), CONNECT_TIMEOUT_MS)
    const current = (): boolean => !this.disposed && this.peer === peer
    const onConnectionChange = (): void => {
      if (!current()) {
        return
      }
      if (peer.connectionState === 'failed' || peer.iceConnectionState === 'failed') {
        this.fail(peer)
      } else if (peer.connectionState === 'disconnected' || peer.iceConnectionState === 'disconnected') {
        if (this.disconnectTimeout === null) {
          this.disconnectTimeout = window.setTimeout(() => this.fail(peer), DISCONNECT_GRACE_MS)
        }
      } else if (peer.connectionState === 'closed') {
        this.fail(peer)
      } else if (peer.connectionState === 'connected') {
        this.clearTimer('connectTimeout')
        this.clearTimer('disconnectTimeout')
        this.onState({ state: 'connected', error: null })
      }
    }
    peer.onconnectionstatechange = onConnectionChange
    peer.oniceconnectionstatechange = onConnectionChange
    peer.ontrack = (event) => {
      if (!current()) {
        event.track.stop()
        return
      }
      if (!this.stream) {
        this.stream = new MediaStream()
        this.video.srcObject = this.stream
      }
      this.stream.addTrack(event.track)
      this.video.muted = true
      this.video.defaultMuted = true
      void this.video.play().catch(() => {})
    }
    try {
      peer.addTransceiver('video', { direction: 'recvonly' })
      peer.addTransceiver('audio', { direction: 'recvonly' })
      await peer.setLocalDescription(await peer.createOffer())
      await waitForIceGathering(peer, controller.signal)
      if (!current()) {
        return
      }
      const sdp = peer.localDescription?.sdp
      if (!sdp) {
        throw new Error('Missing local description')
      }
      const token = this.session.token ?? null
      const leaseExpiresAt = this.session.lease_expires_at
      const answer = await apiClient.createCameraPreviewPeer(
        this.session.camera_name, token, { type: 'offer', sdp }, { signal: controller.signal },
      )
      if (!current()) {
        this.deletePeer(answer.session_id)
        return
      }
      this.sessionId = answer.session_id
      await peer.setRemoteDescription({ type: 'answer', sdp: answer.sdp })
      if (current() && ((this.session.token ?? null) !== token || this.session.lease_expires_at !== leaseExpiresAt)) {
        void this.renew(answer.session_id, this.session.token ?? null)
      }
    } catch {
      this.fail(peer)
    }
  }

  private authorizationIsFresh(): boolean {
    const expiresAt = this.session.lease_expires_at ?? this.session.token_expires_at
    return !expiresAt || Date.parse(expiresAt) > Date.now()
  }
}
