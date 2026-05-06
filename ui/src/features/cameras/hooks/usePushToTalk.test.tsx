// @vitest-environment happy-dom

import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { apiClient } from '../../../api/client'
import type { TalkSessionResponse, TalkStatusResponse } from '../../../api/generated/types'
import { usePushToTalk } from './usePushToTalk'

const idleStatus: TalkStatusResponse = {
  camera_name: 'front',
  enabled: true,
  policy_enabled: true,
  capability: 'supported',
  state: 'idle',
  active_session_id: null,
  supported_codecs: ['pcm_s16le'],
  offered_codecs: ['PCMU/8000'],
  selected_codec: 'pcm_s16le',
  last_error: null,
}

const talkSession: TalkSessionResponse = {
  camera_name: 'front',
  session_id: 'tk_123',
  state: 'starting',
  input: { codec: 'pcm_s16le', sample_rate: 16000, channels: 1, frame_ms: 20 },
  websocket_url: '/api/v1/talk/cameras/front/sessions/tk_123/stream?token=talk-token',
  stream_url: '/api/v1/talk/cameras/front/sessions/tk_123/stream?token=talk-token',
  token: 'talk-token',
  token_expires_at: null,
  max_session_s: 30,
  idle_timeout_s: 5,
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

function createMediaStream() {
  const track = { stop: vi.fn() }
  return {
    track,
    stream: {
      getTracks: () => [track],
    } as unknown as MediaStream,
  }
}

let sockets: FakeWebSocket[] = []
let lastScriptProcessor: FakeScriptProcessorNode | null = null

class FakeWebSocket {
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static readonly CLOSING = 2
  static readonly CLOSED = 3

  binaryType: BinaryType = 'blob'
  bufferedAmount = 0
  readyState = FakeWebSocket.CONNECTING
  sent: unknown[] = []
  onopen: ((event: Event) => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null

  readonly url: string

  constructor(url: string) {
    this.url = url
    sockets.push(this)
  }

  send(payload: unknown) {
    this.sent.push(payload)
  }

  open() {
    this.readyState = FakeWebSocket.OPEN
    this.onopen?.(new Event('open'))
  }

  message(data: unknown) {
    this.onmessage?.(new MessageEvent('message', { data }))
  }

  close(code = 1000, reason = '') {
    if (this.readyState === FakeWebSocket.CLOSED) {
      return
    }
    this.readyState = FakeWebSocket.CLOSED
    this.onclose?.(new CloseEvent('close', { code, reason }))
  }
}

class FakeAudioNode {
  connect = vi.fn()
  disconnect = vi.fn()
}

class FakeScriptProcessorNode extends FakeAudioNode {
  onaudioprocess: ((event: AudioProcessingEvent) => void) | null = null

  emit(samples: Float32Array) {
    this.onaudioprocess?.({
      inputBuffer: {
        getChannelData: () => samples,
      },
    } as unknown as AudioProcessingEvent)
  }
}

class FakeAudioContext {
  state: AudioContextState = 'running'
  sampleRate = 16_000
  destination = new FakeAudioNode() as unknown as AudioDestinationNode
  createMediaStreamSource = vi.fn(() => new FakeAudioNode() as unknown as MediaStreamAudioSourceNode)
  createGain = vi.fn(() => ({
    gain: { value: 1 },
    connect: vi.fn(),
    disconnect: vi.fn(),
  }) as unknown as GainNode)
  createScriptProcessor = vi.fn(() => {
    lastScriptProcessor = new FakeScriptProcessorNode()
    return lastScriptProcessor as unknown as ScriptProcessorNode
  })
  resume = vi.fn().mockResolvedValue(undefined)
  close = vi.fn().mockResolvedValue(undefined)

  constructor(options?: AudioContextOptions) {
    this.sampleRate = options?.sampleRate ?? 16_000
  }
}

function installBrowserFakes(getUserMedia: ReturnType<typeof vi.fn>) {
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: { getUserMedia },
  })
  Object.assign(globalThis, {
    WebSocket: FakeWebSocket,
    AudioContext: FakeAudioContext,
    CloseEvent: globalThis.CloseEvent ?? class extends Event {
      code: number
      reason: string
      constructor(type: string, init: { code?: number; reason?: string } = {}) {
        super(type)
        this.code = init.code ?? 1000
        this.reason = init.reason ?? ''
      }
    },
  })
  Reflect.deleteProperty(globalThis, 'AudioWorkletNode')
}

describe('usePushToTalk', () => {
  beforeEach(() => {
    sockets = []
    lastScriptProcessor = null
    vi.spyOn(apiClient, 'getCameraTalkStatus').mockResolvedValue({ ...idleStatus, httpStatus: 200 })
    vi.spyOn(apiClient, 'prepareCameraTalkSession').mockResolvedValue({ ...talkSession, httpStatus: 201 })
    vi.spyOn(apiClient, 'stopCameraTalkSession').mockResolvedValue({
      accepted: true,
      state: 'stopping',
      httpStatus: 202,
    })
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('starts, marks ready, and stops a talk session through the Phase 3 protocol', async () => {
    const { stream, track } = createMediaStream()
    installBrowserFakes(vi.fn().mockResolvedValue(stream))
    const { result } = renderHook(() => usePushToTalk('front'))
    await waitFor(() => expect(result.current.canStart).toBe(true))

    void act(() => {
      void result.current.start()
    })
    await waitFor(() => expect(sockets).toHaveLength(1))
    sockets[0].open()

    expect(sockets[0].sent[0]).toBe(JSON.stringify({
      type: 'start',
      codec: 'pcm_s16le',
      sample_rate: 16000,
      channels: 1,
      frame_ms: 20,
    }))

    sockets[0].message(JSON.stringify({ type: 'ready' }))
    await waitFor(() => expect(result.current.isStreaming).toBe(true))

    await act(async () => {
      await result.current.stop()
    })

    expect(sockets[0].sent).toContain(JSON.stringify({ type: 'stop' }))
    expect(apiClient.stopCameraTalkSession).toHaveBeenCalledWith('front', 'tk_123')
    expect(track.stop).toHaveBeenCalled()
    expect(result.current.isStreaming).toBe(false)
  })

  it('cancels release before getUserMedia resolves without preparing a session', async () => {
    const media = createMediaStream()
    const mediaDeferred = deferred<MediaStream>()
    installBrowserFakes(vi.fn().mockReturnValue(mediaDeferred.promise))
    const { result } = renderHook(() => usePushToTalk('front'))
    await waitFor(() => expect(result.current.canStart).toBe(true))

    void act(() => {
      void result.current.start()
    })
    await waitFor(() => expect(result.current.isStarting).toBe(true))
    await act(async () => {
      await result.current.stop()
    })
    mediaDeferred.resolve(media.stream)

    await waitFor(() => expect(media.track.stop).toHaveBeenCalled())
    expect(apiClient.prepareCameraTalkSession).not.toHaveBeenCalled()
    expect(apiClient.stopCameraTalkSession).not.toHaveBeenCalled()
    expect(result.current.isStreaming).toBe(false)
  })

  it('cancels release after prepare but before WebSocket ready and stops the prepared session', async () => {
    const { stream } = createMediaStream()
    installBrowserFakes(vi.fn().mockResolvedValue(stream))
    const { result } = renderHook(() => usePushToTalk('front'))
    await waitFor(() => expect(result.current.canStart).toBe(true))

    void act(() => {
      void result.current.start()
    })
    await waitFor(() => expect(sockets).toHaveLength(1))

    await act(async () => {
      await result.current.stop()
    })

    await waitFor(() => expect(apiClient.stopCameraTalkSession).toHaveBeenCalledTimes(1))
    expect(apiClient.stopCameraTalkSession).toHaveBeenCalledWith('front', 'tk_123')
    expect(sockets[0].readyState).toBe(FakeWebSocket.CLOSED)
    expect(result.current.isStreaming).toBe(false)
  })

  it('does not prepare a session when microphone permission is denied', async () => {
    installBrowserFakes(vi.fn().mockRejectedValue(new Error('denied')))
    const { result } = renderHook(() => usePushToTalk('front'))
    await waitFor(() => expect(result.current.canStart).toBe(true))

    await act(async () => {
      await result.current.start()
    })

    expect(apiClient.prepareCameraTalkSession).not.toHaveBeenCalled()
    expect(result.current.error).toBeInstanceOf(Error)
  })

  it('closes and stops the session when audio backpressure exceeds the threshold', async () => {
    const { stream } = createMediaStream()
    installBrowserFakes(vi.fn().mockResolvedValue(stream))
    const { result } = renderHook(() => usePushToTalk('front'))
    await waitFor(() => expect(result.current.canStart).toBe(true))

    void act(() => {
      void result.current.start()
    })
    await waitFor(() => expect(sockets).toHaveLength(1))
    sockets[0].open()
    sockets[0].message(JSON.stringify({ type: 'ready' }))
    await waitFor(() => expect(result.current.isStreaming).toBe(true))

    sockets[0].bufferedAmount = 20_000
    lastScriptProcessor?.emit(new Float32Array(320).fill(0.2))

    await waitFor(() => expect(sockets[0].readyState).toBe(FakeWebSocket.CLOSED))
    await waitFor(() => expect(apiClient.stopCameraTalkSession).toHaveBeenCalledWith('front', 'tk_123'))
    expect(result.current.error).toBeInstanceOf(Error)
  })
})
