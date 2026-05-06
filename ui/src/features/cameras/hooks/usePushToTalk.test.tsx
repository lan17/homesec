// @vitest-environment happy-dom

import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { apiClient } from '../../../api/client'
import type { TalkSessionResponse, TalkStatusResponse } from '../../../api/generated/types'
import { usePushToTalk } from './usePushToTalk'

type TalkStatusSnapshot = TalkStatusResponse & { httpStatus: number }

const idleStatus: TalkStatusResponse = {
  camera_name: 'front',
  enabled: true,
  policy_enabled: true,
  capability: 'supported',
  state: 'idle',
  active_session_id: null,
  supported_codecs: ['PCMU/8000'],
  offered_codecs: ['PCMU/8000'],
  selected_codec: 'PCMU/8000',
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
let fakeAudioWorklet: Pick<AudioWorklet, 'addModule'> | undefined
let audioWorkletAddModule: ReturnType<typeof vi.fn> | null = null
let lastAudioWorkletNode: FakeAudioWorkletNode | null = null
let lastGainNode: FakeGainNode | null = null

class FakeWebSocket {
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static readonly CLOSING = 2
  static readonly CLOSED = 3

  binaryType: BinaryType = 'blob'
  bufferedAmount = 0
  readyState = FakeWebSocket.CONNECTING
  sent: unknown[] = []
  deferCloseEvent = false
  lastClose: { code: number; reason: string } | null = null
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
    this.lastClose = { code, reason }
    if (!this.deferCloseEvent) {
      this.emitClose()
    }
  }

  emitClose(code = this.lastClose?.code ?? 1000, reason = this.lastClose?.reason ?? '') {
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

class FakeGainNode extends FakeAudioNode {
  gain = { value: 1 }
}

function recordAudioWorkletNode(node: FakeAudioWorkletNode): void {
  lastAudioWorkletNode = node
}

class FakeAudioWorkletNode extends FakeAudioNode {
  port = {
    onmessage: null as ((event: MessageEvent<ArrayBuffer>) => void) | null,
  }

  readonly name: string
  readonly options?: AudioWorkletNodeOptions

  constructor(_context: BaseAudioContext, name: string, options?: AudioWorkletNodeOptions) {
    super()
    this.name = name
    this.options = options
    recordAudioWorkletNode(this)
  }
}

class FakeAudioContext {
  state: AudioContextState = 'running'
  sampleRate = 16_000
  destination = new FakeAudioNode() as unknown as AudioDestinationNode
  audioWorklet = fakeAudioWorklet as AudioWorklet | undefined
  createMediaStreamSource = vi.fn(() => new FakeAudioNode() as unknown as MediaStreamAudioSourceNode)
  createGain = vi.fn(() => {
    lastGainNode = new FakeGainNode()
    return lastGainNode as unknown as GainNode
  })
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

function installBrowserFakes(
  getUserMedia: ReturnType<typeof vi.fn>,
  options: { audioWorklet?: boolean } = {},
) {
  fakeAudioWorklet = undefined
  audioWorkletAddModule = null
  if (options.audioWorklet) {
    audioWorkletAddModule = vi.fn().mockResolvedValue(undefined)
    fakeAudioWorklet = {
      addModule: audioWorkletAddModule as AudioWorklet['addModule'],
    }
  }
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
  if (options.audioWorklet) {
    Object.assign(globalThis, { AudioWorkletNode: FakeAudioWorkletNode })
  } else {
    Reflect.deleteProperty(globalThis, 'AudioWorkletNode')
  }
}

describe('usePushToTalk', () => {
  beforeEach(() => {
    sockets = []
    lastScriptProcessor = null
    fakeAudioWorklet = undefined
    audioWorkletAddModule = null
    lastAudioWorkletNode = null
    lastGainNode = null
    vi.spyOn(apiClient, 'getCameraTalkStatus').mockResolvedValue({ ...idleStatus, httpStatus: 200 })
    vi.spyOn(apiClient, 'prepareCameraTalkSession').mockResolvedValue({ ...talkSession, httpStatus: 201 })
    vi.spyOn(apiClient, 'stopCameraTalkSession').mockResolvedValue({
      accepted: true,
      state: 'stopping',
      httpStatus: 202,
    })
    Object.defineProperty(window, 'isSecureContext', {
      configurable: true,
      value: true,
    })
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('starts, marks ready, and stops a talk session through the Phase 3 protocol', async () => {
    // Given: Browser microphone capture and API talk preparation both succeed
    const { stream, track } = createMediaStream()
    installBrowserFakes(vi.fn().mockResolvedValue(stream))
    const { result } = renderHook(() => usePushToTalk('front'))
    await waitFor(() => expect(result.current.canStart).toBe(true))

    // When: Starting talk, receiving WebSocket ready, and then stopping
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
    expect(result.current.status?.selected_codec).toBe('PCMU/8000')

    await act(async () => {
      await result.current.stop()
    })

    // Then: The hook sends the protocol messages, stops the session, and releases media
    expect(sockets[0].sent).toContain(JSON.stringify({ type: 'stop' }))
    expect(apiClient.stopCameraTalkSession).toHaveBeenCalledWith('front', 'tk_123')
    expect(track.stop).toHaveBeenCalled()
    expect(result.current.isStreaming).toBe(false)
  })

  it('uses the prepared session input for websocket start and PCM frame sizing', async () => {
    // Given: The server prepares a talk session with a non-default input format
    const customSession: TalkSessionResponse = {
      ...talkSession,
      input: { codec: 'pcm_s16le', sample_rate: 8000, channels: 1, frame_ms: 20 },
    }
    vi.mocked(apiClient.prepareCameraTalkSession).mockResolvedValueOnce({
      ...customSession,
      httpStatus: 201,
    })
    const { stream } = createMediaStream()
    installBrowserFakes(vi.fn().mockResolvedValue(stream))
    const { result } = renderHook(() => usePushToTalk('front'))
    await waitFor(() => expect(result.current.canStart).toBe(true))

    // When: Starting talk and sending one browser audio callback after ready
    void act(() => {
      void result.current.start()
    })
    await waitFor(() => expect(sockets).toHaveLength(1))
    sockets[0].open()
    sockets[0].message(JSON.stringify({ type: 'ready' }))
    await waitFor(() => expect(result.current.isStreaming).toBe(true))
    lastScriptProcessor?.emit(new Float32Array(160).fill(0.2))

    // Then: The UI omits hard-coded input on prepare and uses the session input on stream
    expect(apiClient.prepareCameraTalkSession).toHaveBeenCalledWith(
      'front',
      {},
      expect.objectContaining({ signal: expect.any(Object) }),
    )
    expect(sockets[0].sent[0]).toBe(JSON.stringify({
      type: 'start',
      codec: 'pcm_s16le',
      sample_rate: 8000,
      channels: 1,
      frame_ms: 20,
    }))
    const audioFrame = sockets[0].sent.find((payload) => payload instanceof ArrayBuffer)
    expect(audioFrame).toBeInstanceOf(ArrayBuffer)
    expect((audioFrame as ArrayBuffer).byteLength).toBe(320)
  })

  it('ignores stale status responses after switching cameras', async () => {
    // Given: The first camera status request is still in flight when the hook switches cameras
    const frontStatus = deferred<TalkStatusSnapshot>()
    const officeStatus: TalkStatusSnapshot = {
      ...idleStatus,
      camera_name: 'office',
      enabled: false,
      policy_enabled: false,
      capability: 'disabled',
      state: 'disabled',
      httpStatus: 200,
    }
    vi.mocked(apiClient.getCameraTalkStatus).mockImplementation((cameraName) => {
      if (cameraName === 'front') {
        return frontStatus.promise
      }
      return Promise.resolve(officeStatus)
    })
    const { result, rerender } = renderHook(
      ({ cameraName }: { cameraName: string }) => usePushToTalk(cameraName),
      { initialProps: { cameraName: 'front' } },
    )
    await waitFor(() => expect(apiClient.getCameraTalkStatus).toHaveBeenCalledWith('front'))

    // When: The hook moves to another camera before the first status request resolves
    rerender({ cameraName: 'office' })
    await waitFor(() => expect(result.current.status?.camera_name).toBe('office'))
    await act(async () => {
      frontStatus.resolve({ ...idleStatus, camera_name: 'front', httpStatus: 200 })
    })

    // Then: The late response from the previous camera cannot overwrite current state
    expect(result.current.status?.camera_name).toBe('office')
    expect(result.current.status?.state).toBe('disabled')
    expect(result.current.canStart).toBe(false)
  })

  it('allows a new camera start after switching away from a hung microphone request', async () => {
    // Given: The first camera start is stuck waiting for microphone capture
    const frontMedia = deferred<MediaStream>()
    const officeMedia = createMediaStream()
    const getUserMedia = vi.fn()
      .mockReturnValueOnce(frontMedia.promise)
      .mockResolvedValueOnce(officeMedia.stream)
    const officeStatus: TalkStatusSnapshot = {
      ...idleStatus,
      camera_name: 'office',
      httpStatus: 200,
    }
    vi.mocked(apiClient.getCameraTalkStatus).mockImplementation((cameraName) => {
      if (cameraName === 'office') {
        return Promise.resolve(officeStatus)
      }
      return Promise.resolve({ ...idleStatus, httpStatus: 200 })
    })
    installBrowserFakes(getUserMedia)
    const { result, rerender } = renderHook(
      ({ cameraName }: { cameraName: string }) => usePushToTalk(cameraName),
      { initialProps: { cameraName: 'front' } },
    )
    await waitFor(() => expect(result.current.canStart).toBe(true))
    void act(() => {
      void result.current.start()
    })
    await waitFor(() => expect(getUserMedia).toHaveBeenCalledTimes(1))

    // When: The hook switches cameras and starts talk for the new camera
    rerender({ cameraName: 'office' })
    await waitFor(() => expect(result.current.canStart).toBe(true))
    void act(() => {
      void result.current.start()
    })

    // Then: The stale start lock from the first camera does not block the new start
    await waitFor(() => expect(getUserMedia).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(sockets).toHaveLength(1))
    expect(apiClient.prepareCameraTalkSession).toHaveBeenCalledWith(
      'office',
      {},
      expect.objectContaining({ signal: expect.any(Object) }),
    )
  })

  it('clears stale capability errors after a successful retry starts streaming', async () => {
    // Given: The current talk status is retryable but carries a stale probe error.
    vi.mocked(apiClient.getCameraTalkStatus).mockResolvedValueOnce({
      ...idleStatus,
      capability: 'error',
      state: 'error',
      last_error: 'previous probe failed',
      httpStatus: 200,
    })
    const { stream } = createMediaStream()
    installBrowserFakes(vi.fn().mockResolvedValue(stream))
    const { result } = renderHook(() => usePushToTalk('front'))
    await waitFor(() => expect(result.current.canStart).toBe(true))

    // When: Retrying talk and reaching the ready websocket state.
    void act(() => {
      void result.current.start()
    })
    await waitFor(() => expect(sockets).toHaveLength(1))
    sockets[0].open()
    sockets[0].message(JSON.stringify({ type: 'ready' }))
    await waitFor(() => expect(result.current.isStreaming).toBe(true))

    // Then: The optimistic active status no longer displays the stale probe error.
    expect(result.current.status?.state).toBe('active')
    expect(result.current.status?.capability).toBe('supported')
    expect(result.current.status?.selected_codec).toBe('PCMU/8000')
    expect(result.current.status?.last_error).toBeNull()
  })

  it('uses the camera codec from the ready message for optimistic active status', async () => {
    // Given: The server reports the actual selected camera-side codec in ready.
    const { stream } = createMediaStream()
    installBrowserFakes(vi.fn().mockResolvedValue(stream))
    const { result } = renderHook(() => usePushToTalk('front'))
    await waitFor(() => expect(result.current.canStart).toBe(true))

    // When: Starting talk and receiving a ready message with the selected camera codec.
    void act(() => {
      void result.current.start()
    })
    await waitFor(() => expect(sockets).toHaveLength(1))
    sockets[0].open()
    sockets[0].message(JSON.stringify({
      type: 'ready',
      camera_codec: 'PCMA/8000',
    }))
    await waitFor(() => expect(result.current.isStreaming).toBe(true))

    // Then: Optimistic UI state uses the camera-side codec, not the browser PCM codec.
    expect(result.current.status?.state).toBe('active')
    expect(result.current.status?.selected_codec).toBe('PCMA/8000')
    expect(result.current.status?.supported_codecs).toEqual(['PCMU/8000'])
    expect(result.current.status?.offered_codecs).toEqual(['PCMU/8000'])
    expect(result.current.status?.last_error).toBeNull()
  })

  it('cancels release before getUserMedia resolves without preparing a session', async () => {
    // Given: Microphone permission is still pending while talk can start
    const media = createMediaStream()
    const mediaDeferred = deferred<MediaStream>()
    installBrowserFakes(vi.fn().mockReturnValue(mediaDeferred.promise))
    const { result } = renderHook(() => usePushToTalk('front'))
    await waitFor(() => expect(result.current.canStart).toBe(true))

    // When: The user releases push-to-talk before getUserMedia resolves
    void act(() => {
      void result.current.start()
    })
    await waitFor(() => expect(result.current.isStarting).toBe(true))
    await act(async () => {
      await result.current.stop()
    })
    mediaDeferred.resolve(media.stream)

    // Then: The late media stream is closed and no backend talk session is reserved
    await waitFor(() => expect(media.track.stop).toHaveBeenCalled())
    expect(apiClient.prepareCameraTalkSession).not.toHaveBeenCalled()
    expect(apiClient.stopCameraTalkSession).not.toHaveBeenCalled()
    expect(result.current.isStreaming).toBe(false)
  })

  it('cancels release after prepare but before WebSocket ready and stops the prepared session', async () => {
    // Given: Starting talk reaches the reserved-session/WebSocket phase
    const { stream } = createMediaStream()
    installBrowserFakes(vi.fn().mockResolvedValue(stream))
    const { result } = renderHook(() => usePushToTalk('front'))
    await waitFor(() => expect(result.current.canStart).toBe(true))

    // When: The user releases before the WebSocket reports ready
    void act(() => {
      void result.current.start()
    })
    await waitFor(() => expect(sockets).toHaveLength(1))

    await act(async () => {
      await result.current.stop()
    })

    // Then: The prepared runtime session is stopped and the socket is closed
    await waitFor(() => expect(apiClient.stopCameraTalkSession).toHaveBeenCalledTimes(1))
    expect(apiClient.stopCameraTalkSession).toHaveBeenCalledWith('front', 'tk_123')
    expect(sockets[0].readyState).toBe(FakeWebSocket.CLOSED)
    expect(result.current.isStreaming).toBe(false)
  })

  it('does not prepare a session when microphone permission is denied', async () => {
    // Given: Browser microphone capture is denied
    installBrowserFakes(vi.fn().mockRejectedValue(new Error('denied')))
    const { result } = renderHook(() => usePushToTalk('front'))
    await waitFor(() => expect(result.current.canStart).toBe(true))

    // When: Starting push-to-talk
    await act(async () => {
      await result.current.start()
    })

    // Then: The hook reports the browser error without reserving a session
    expect(apiClient.prepareCameraTalkSession).not.toHaveBeenCalled()
    expect(result.current.error).toBeInstanceOf(Error)
  })

  it('reports insecure browser origins before requesting microphone capture', async () => {
    const getUserMedia = vi.fn()
    Object.defineProperty(window, 'isSecureContext', {
      configurable: true,
      value: false,
    })
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: undefined,
    })
    Object.assign(globalThis, { WebSocket: FakeWebSocket })
    const { result } = renderHook(() => usePushToTalk('front'))
    await waitFor(() => expect(result.current.canStart).toBe(true))

    // Given: The app is loaded from an insecure LAN origin where browsers hide getUserMedia
    // When: Starting push-to-talk
    await act(async () => {
      await result.current.start()
    })

    // Then: The hook reports the HTTPS/localhost requirement and never asks for the mic
    expect(result.current.error).toEqual(new Error('Microphone capture requires HTTPS or localhost.'))
    expect(getUserMedia).not.toHaveBeenCalled()
    expect(apiClient.prepareCameraTalkSession).not.toHaveBeenCalled()
  })

  it('closes and stops the session when audio backpressure exceeds the threshold', async () => {
    // Given: An active push-to-talk session with browser audio capture
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

    // When: The WebSocket buffered audio exceeds the backpressure threshold
    sockets[0].bufferedAmount = 20_000
    lastScriptProcessor?.emit(new Float32Array(320).fill(0.2))

    // Then: The hook closes the stream, stops the backend session, and surfaces an error
    await waitFor(() => expect(sockets[0].readyState).toBe(FakeWebSocket.CLOSED))
    await waitFor(() => expect(apiClient.stopCameraTalkSession).toHaveBeenCalledWith('front', 'tk_123'))
    expect(result.current.error).toBeInstanceOf(Error)
  })

  it('ignores stale socket close events after switching cameras', async () => {
    // Given: An active front camera talk socket whose close event will be delivered late
    const officeStatus: TalkStatusSnapshot = {
      ...idleStatus,
      camera_name: 'office',
      httpStatus: 200,
    }
    vi.mocked(apiClient.getCameraTalkStatus).mockImplementation((cameraName) => {
      if (cameraName === 'office') {
        return Promise.resolve(officeStatus)
      }
      return Promise.resolve({ ...idleStatus, httpStatus: 200 })
    })
    vi.mocked(apiClient.prepareCameraTalkSession).mockImplementation((cameraName) => {
      const sessionId = cameraName === 'office' ? 'tk_office' : 'tk_front'
      return Promise.resolve({
        ...talkSession,
        camera_name: cameraName,
        session_id: sessionId,
        websocket_url: `/api/v1/talk/cameras/${cameraName}/sessions/${sessionId}/stream`,
        stream_url: `/api/v1/talk/cameras/${cameraName}/sessions/${sessionId}/stream`,
        httpStatus: 201,
      })
    })
    const frontMedia = createMediaStream()
    const officeMedia = createMediaStream()
    installBrowserFakes(vi.fn()
      .mockResolvedValueOnce(frontMedia.stream)
      .mockResolvedValueOnce(officeMedia.stream))
    const { result, rerender } = renderHook(
      ({ cameraName }: { cameraName: string }) => usePushToTalk(cameraName),
      { initialProps: { cameraName: 'front' } },
    )
    await waitFor(() => expect(result.current.canStart).toBe(true))
    void act(() => {
      void result.current.start()
    })
    await waitFor(() => expect(sockets).toHaveLength(1))
    sockets[0].open()
    sockets[0].message(JSON.stringify({ type: 'ready' }))
    await waitFor(() => expect(result.current.isStreaming).toBe(true))
    const frontSocket = sockets[0]
    frontSocket.deferCloseEvent = true

    // When: The hook switches cameras and starts the replacement before the old close event
    rerender({ cameraName: 'office' })
    await waitFor(() => expect(result.current.canStart).toBe(true))
    void act(() => {
      void result.current.start()
    })
    await waitFor(() => expect(sockets).toHaveLength(2))
    const officeSocket = sockets[1]
    officeSocket.open()
    officeSocket.message(JSON.stringify({ type: 'ready' }))
    await waitFor(() => expect(result.current.isStreaming).toBe(true))
    await act(async () => {
      frontSocket.emitClose(1006, 'front socket closed late')
      await Promise.resolve()
    })

    // Then: The stale close event cannot close or clear the new camera stream
    expect(officeSocket.readyState).toBe(FakeWebSocket.OPEN)
    expect(result.current.status?.camera_name).toBe('office')
    expect(result.current.status?.state).toBe('active')
    expect(result.current.error).toBeNull()
  })

  it('keeps the AudioWorklet pipeline connected through a silent gain node', async () => {
    // Given: A browser with AudioWorklet support
    const { stream } = createMediaStream()
    installBrowserFakes(vi.fn().mockResolvedValue(stream), { audioWorklet: true })
    const { result } = renderHook(() => usePushToTalk('front'))
    await waitFor(() => expect(result.current.canStart).toBe(true))

    // When: Starting talk reaches the ready WebSocket state
    void act(() => {
      void result.current.start()
    })
    await waitFor(() => expect(sockets).toHaveLength(1))
    sockets[0].open()
    sockets[0].message(JSON.stringify({ type: 'ready' }))
    await waitFor(() => expect(result.current.isStreaming).toBe(true))

    // Then: The AudioWorklet graph is pulled by connecting it to a muted destination
    expect(audioWorkletAddModule).toHaveBeenCalledWith(expect.any(URL))
    expect(lastAudioWorkletNode?.options?.numberOfOutputs).toBe(1)
    expect(lastAudioWorkletNode?.connect).toHaveBeenCalledWith(lastGainNode)
    expect(lastGainNode?.gain.value).toBe(0)
    expect(lastGainNode?.connect).toHaveBeenCalled()
  })
})
