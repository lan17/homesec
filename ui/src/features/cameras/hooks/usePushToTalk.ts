import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { apiClient } from '../../../api/client'
import type {
  TalkInputFormat,
  TalkSessionResponse,
  TalkState,
  TalkStatusResponse,
} from '../../../api/generated/types'

const DEFAULT_TALK_INPUT: TalkInputFormat = {
  codec: 'pcm_s16le',
  sample_rate: 16_000,
  channels: 1,
  frame_ms: 20,
}

const TALK_WS_READY_TYPE = 'ready'
const TALK_WS_STOP_TYPE = 'stop'
const TALK_AUDIO_WORKLET_URL = new URL('../audio/talkPcmProcessor.js', import.meta.url)

type TalkReadyMessage = {
  type: typeof TALK_WS_READY_TYPE
}

type AudioPipeline = {
  context: AudioContext
  source: MediaStreamAudioSourceNode
  node: AudioNode
  silentGain?: GainNode
  stream: MediaStream
}

type ScriptProcessorNodeWithHandler = ScriptProcessorNode & {
  onaudioprocess: ((event: AudioProcessingEvent) => void) | null
}

export interface PushToTalkState {
  error: unknown
  isPending: boolean
  isStarting: boolean
  isStopping: boolean
  isStreaming: boolean
  session: TalkSessionResponse | null
  status: TalkStatusResponse | null
  canStart: boolean
  refreshStatus: () => Promise<void>
  start: () => Promise<void>
  stop: () => Promise<void>
}

function isReadyMessage(value: unknown): value is TalkReadyMessage {
  return Boolean(
    value && typeof value === 'object' && 'type' in value && value.type === TALK_WS_READY_TYPE,
  )
}

function resolveTalkWebSocketUrl(pathOrUrl: string): string {
  const trimmed = pathOrUrl.trim()
  if (trimmed.startsWith('ws://') || trimmed.startsWith('wss://')) {
    return trimmed
  }

  const resolved = trimmed.startsWith('http://') || trimmed.startsWith('https://')
    ? trimmed
    : apiClient.resolvePath(trimmed)
  const url = new URL(resolved, window.location.origin)
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  return url.toString()
}

function closeMediaStream(stream: MediaStream | null): void {
  stream?.getTracks().forEach((track) => {
    track.stop()
  })
}

function frameSampleCount(input: TalkInputFormat): number {
  return Math.round((input.sample_rate * input.frame_ms) / 1000)
}

function floatToPcm16(frame: Float32Array): ArrayBuffer {
  const pcm = new Int16Array(frame.length)
  for (let index = 0; index < frame.length; index += 1) {
    const clamped = Math.max(-1, Math.min(1, frame[index] ?? 0))
    pcm[index] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff
  }
  return pcm.buffer
}

async function startAudioPipeline(
  stream: MediaStream,
  input: TalkInputFormat,
  socket: WebSocket,
): Promise<AudioPipeline> {
  const context = new AudioContext({ sampleRate: input.sample_rate })
  const source = context.createMediaStreamSource(stream)
  const samplesPerFrame = frameSampleCount(input)

  const sendFrame = (frame: ArrayBuffer): void => {
    if (socket.readyState === WebSocket.OPEN) {
      socket.send(frame)
    }
  }

  if (context.audioWorklet && typeof AudioWorkletNode !== 'undefined') {
    await context.audioWorklet.addModule(TALK_AUDIO_WORKLET_URL)
    const worklet = new AudioWorkletNode(context, 'talk-pcm-processor', {
      numberOfInputs: 1,
      numberOfOutputs: 0,
      processorOptions: { frameSamples: samplesPerFrame },
    })
    worklet.port.onmessage = (event: MessageEvent<ArrayBuffer>) => {
      sendFrame(event.data)
    }
    source.connect(worklet)
    if (context.state === 'suspended') {
      await context.resume()
    }
    return { context, source, node: worklet, stream }
  }

  const scriptNode = context.createScriptProcessor(1024, input.channels, 1) as ScriptProcessorNodeWithHandler
  const silentGain = context.createGain()
  silentGain.gain.value = 0
  let samples: number[] = []
  scriptNode.onaudioprocess = (event) => {
    const channel = event.inputBuffer.getChannelData(0)
    samples = samples.concat(Array.from(channel))
    while (samples.length >= samplesPerFrame) {
      const frame = new Float32Array(samples.slice(0, samplesPerFrame))
      samples = samples.slice(samplesPerFrame)
      sendFrame(floatToPcm16(frame))
    }
  }
  source.connect(scriptNode)
  scriptNode.connect(silentGain)
  silentGain.connect(context.destination)
  if (context.state === 'suspended') {
    await context.resume()
  }
  return { context, source, node: scriptNode, silentGain, stream }
}

async function stopAudioPipeline(pipeline: AudioPipeline | null): Promise<void> {
  if (!pipeline) {
    return
  }
  if (typeof AudioWorkletNode !== 'undefined' && pipeline.node instanceof AudioWorkletNode) {
    pipeline.node.port.onmessage = null
  }
  if ('onaudioprocess' in pipeline.node) {
    ;(pipeline.node as ScriptProcessorNodeWithHandler).onaudioprocess = null
  }
  pipeline.source.disconnect()
  pipeline.node.disconnect()
  pipeline.silentGain?.disconnect()
  closeMediaStream(pipeline.stream)
  await pipeline.context.close().catch(() => {})
}

function statusAllowsStart(status: TalkStatusResponse | null): boolean {
  if (!status || !status.enabled) {
    return false
  }
  return status.state === 'idle' || status.state === 'error'
}

function nextStatusFromState(
  cameraName: string,
  state: TalkState,
  sessionId: string | null,
  previous: TalkStatusResponse | null,
): TalkStatusResponse {
  return {
    camera_name: previous?.camera_name ?? cameraName,
    enabled: previous?.enabled ?? true,
    state,
    active_session_id: sessionId,
    supported_codecs: previous?.supported_codecs ?? ['pcm_s16le'],
    selected_codec: previous?.selected_codec ?? 'pcm_s16le',
    last_error: null,
  }
}

export function usePushToTalk(cameraName: string): PushToTalkState {
  const [status, setStatus] = useState<TalkStatusResponse | null>(null)
  const [session, setSession] = useState<TalkSessionResponse | null>(null)
  const [error, setError] = useState<unknown>(null)
  const [isPending, setIsPending] = useState(true)
  const [isStarting, setIsStarting] = useState(false)
  const [isStopping, setIsStopping] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const socketRef = useRef<WebSocket | null>(null)
  const audioPipelineRef = useRef<AudioPipeline | null>(null)
  const pendingStreamRef = useRef<MediaStream | null>(null)
  const sessionRef = useRef<TalkSessionResponse | null>(null)
  const intentionalStopRef = useRef(false)
  const mountedRef = useRef(true)

  const cleanupSocketAndAudio = useCallback(async () => {
    const socket = socketRef.current
    socketRef.current = null
    if (socket && socket.readyState !== WebSocket.CLOSED && socket.readyState !== WebSocket.CLOSING) {
      socket.close(1000, 'Talk stopped')
    }
    closeMediaStream(pendingStreamRef.current)
    pendingStreamRef.current = null
    const pipeline = audioPipelineRef.current
    audioPipelineRef.current = null
    await stopAudioPipeline(pipeline)
  }, [])

  const refreshStatus = useCallback(async () => {
    setIsPending(true)
    try {
      const nextStatus = await apiClient.getCameraTalkStatus(cameraName)
      if (mountedRef.current) {
        setStatus(nextStatus)
        setError(null)
      }
    } catch (nextError) {
      if (mountedRef.current) {
        setError(nextError)
      }
    } finally {
      if (mountedRef.current) {
        setIsPending(false)
      }
    }
  }, [cameraName])

  const openTalkSocket = useCallback(
    (nextSession: TalkSessionResponse, stream: MediaStream): Promise<void> => {
      return new Promise((resolve, reject) => {
        const socket = new WebSocket(resolveTalkWebSocketUrl(nextSession.websocket_url))
        socket.binaryType = 'arraybuffer'
        socketRef.current = socket
        let settled = false

        const settleResolve = (): void => {
          settled = true
          resolve()
        }

        const settleReject = (nextError: unknown): void => {
          if (!settled) {
            settled = true
            reject(nextError)
          }
        }

        socket.onopen = () => {
          socket.send(
            JSON.stringify({
              type: 'start',
              codec: nextSession.input.codec,
              sample_rate: nextSession.input.sample_rate,
              channels: nextSession.input.channels,
              frame_ms: nextSession.input.frame_ms,
            }),
          )
        }

        socket.onmessage = (event: MessageEvent) => {
          let parsed: unknown = null
          if (typeof event.data === 'string') {
            try {
              parsed = JSON.parse(event.data)
            } catch {
              parsed = null
            }
          }
          if (!isReadyMessage(parsed)) {
            return
          }
          void startAudioPipeline(stream, nextSession.input, socket)
            .then((pipeline) => {
              pendingStreamRef.current = null
              audioPipelineRef.current = pipeline
              if (mountedRef.current) {
                setIsStreaming(true)
                setStatus((previous) =>
                  nextStatusFromState(cameraName, 'active', nextSession.session_id, previous),
                )
              }
              settleResolve()
            })
            .catch(settleReject)
        }

        socket.onerror = () => {
          settleReject(new Error('Talk WebSocket failed'))
        }

        socket.onclose = (event) => {
          void stopAudioPipeline(audioPipelineRef.current)
          audioPipelineRef.current = null
          closeMediaStream(pendingStreamRef.current)
          pendingStreamRef.current = null
          if (mountedRef.current) {
            setIsStreaming(false)
            setSession(null)
            sessionRef.current = null
            setStatus((previous) => nextStatusFromState(cameraName, 'idle', null, previous))
            if (!intentionalStopRef.current && event.code !== 1000) {
              setError(new Error(event.reason || 'Talk stream closed unexpectedly'))
            }
          }
          settleReject(new Error(event.reason || 'Talk stream closed'))
        }
      })
    },
    [cameraName],
  )

  const start = useCallback(async () => {
    if (isStarting || isStreaming || isStopping) {
      return
    }
    setIsStarting(true)
    setError(null)
    intentionalStopRef.current = false
    let mediaStream: MediaStream | null = null
    let preparedSession: TalkSessionResponse | null = null
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Microphone capture is not supported by this browser.')
      }
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: DEFAULT_TALK_INPUT.channels,
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: DEFAULT_TALK_INPUT.sample_rate,
        },
        video: false,
      })
      pendingStreamRef.current = mediaStream
      const nextSession = await apiClient.prepareCameraTalkSession(cameraName, {
        input: DEFAULT_TALK_INPUT,
      })
      preparedSession = nextSession
      setSession(nextSession)
      sessionRef.current = nextSession
      setStatus((previous) => nextStatusFromState(cameraName, 'starting', nextSession.session_id, previous))
      await openTalkSocket(nextSession, mediaStream)
    } catch (nextError) {
      closeMediaStream(mediaStream)
      pendingStreamRef.current = null
      await cleanupSocketAndAudio()
      if (preparedSession) {
        await apiClient.stopCameraTalkSession(cameraName, preparedSession.session_id).catch(() => {})
      }
      if (mountedRef.current) {
        setError(nextError)
        setSession(null)
        sessionRef.current = null
        setIsStreaming(false)
      }
    } finally {
      if (mountedRef.current) {
        setIsStarting(false)
      }
    }
  }, [cameraName, cleanupSocketAndAudio, isStarting, isStopping, isStreaming, openTalkSocket])

  const stop = useCallback(async () => {
    const activeSession = sessionRef.current
    if (isStopping || (!activeSession && !isStreaming && !isStarting)) {
      return
    }
    intentionalStopRef.current = true
    setIsStopping(true)
    setError(null)
    try {
      const socket = socketRef.current
      if (socket?.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: TALK_WS_STOP_TYPE }))
      }
      await cleanupSocketAndAudio()
      if (activeSession) {
        const stopResponse = await apiClient.stopCameraTalkSession(cameraName, activeSession.session_id)
        if (mountedRef.current) {
          setStatus((previous) => nextStatusFromState(cameraName, stopResponse.state, null, previous))
        }
      }
    } catch (nextError) {
      if (mountedRef.current) {
        setError(nextError)
      }
    } finally {
      if (mountedRef.current) {
        setSession(null)
        sessionRef.current = null
        setIsStreaming(false)
        setIsStarting(false)
        setIsStopping(false)
      }
      intentionalStopRef.current = false
    }
  }, [cameraName, cleanupSocketAndAudio, isStarting, isStopping, isStreaming])

  useEffect(() => {
    mountedRef.current = true
    void refreshStatus()
    return () => {
      mountedRef.current = false
      intentionalStopRef.current = true
      const activeSession = sessionRef.current
      void cleanupSocketAndAudio()
      if (activeSession) {
        void apiClient.stopCameraTalkSession(cameraName, activeSession.session_id).catch(() => {})
      }
    }
  }, [cameraName, cleanupSocketAndAudio, refreshStatus])

  const canStart = useMemo(
    () => !isPending && !isStarting && !isStopping && !isStreaming && statusAllowsStart(status),
    [isPending, isStarting, isStopping, isStreaming, status],
  )

  return {
    error,
    isPending,
    isStarting,
    isStopping,
    isStreaming,
    session,
    status,
    canStart,
    refreshStatus,
    start,
    stop,
  }
}
