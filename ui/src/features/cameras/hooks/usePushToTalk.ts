import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { apiClient } from '../../../api/client'
import {
  expectedPcmFrameBytes,
  frameSampleCount,
  PcmFrameEncoder,
} from '../audio/pcm'
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
const TALK_MAX_BUFFERED_AUDIO_MS = 500

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

async function startAudioPipeline(
  stream: MediaStream,
  input: TalkInputFormat,
  socket: WebSocket,
  onBackpressure: () => void,
): Promise<AudioPipeline> {
  const context = new AudioContext({ sampleRate: input.sample_rate })
  const source = context.createMediaStreamSource(stream)
  const samplesPerFrame = frameSampleCount(input.sample_rate, input.frame_ms)
  const maxBufferedBytes = expectedPcmFrameBytes(
    input.sample_rate,
    input.frame_ms,
    input.channels,
  ) * Math.max(1, Math.ceil(TALK_MAX_BUFFERED_AUDIO_MS / input.frame_ms))

  const sendFrame = (frame: ArrayBuffer): void => {
    if (socket.readyState !== WebSocket.OPEN) {
      return
    }
    if (socket.bufferedAmount > maxBufferedBytes) {
      onBackpressure()
      return
    }
    socket.send(frame)
  }

  if (context.audioWorklet && typeof AudioWorkletNode !== 'undefined') {
    await context.audioWorklet.addModule(TALK_AUDIO_WORKLET_URL)
    const worklet = new AudioWorkletNode(context, 'talk-pcm-processor', {
      numberOfInputs: 1,
      numberOfOutputs: 0,
      processorOptions: {
        frameSamples: samplesPerFrame,
        sourceSampleRate: context.sampleRate,
        targetSampleRate: input.sample_rate,
      },
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
  const encoder = new PcmFrameEncoder(context.sampleRate, input.sample_rate, samplesPerFrame)
  scriptNode.onaudioprocess = (event) => {
    const channel = event.inputBuffer.getChannelData(0)
    for (const frame of encoder.push(channel)) {
      sendFrame(frame)
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
    policy_enabled: previous?.policy_enabled ?? true,
    capability: 'supported',
    state,
    active_session_id: sessionId,
    supported_codecs: previous?.supported_codecs ?? ['pcm_s16le'],
    offered_codecs: previous?.offered_codecs ?? [],
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
  const startGenerationRef = useRef(0)
  const startInFlightRef = useRef(false)
  const startAbortRef = useRef<AbortController | null>(null)

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
    (
      nextSession: TalkSessionResponse,
      stream: MediaStream,
      isStartCancelled: () => boolean,
    ): Promise<void> => {
      return new Promise((resolve, reject) => {
        const socket = new WebSocket(resolveTalkWebSocketUrl(nextSession.websocket_url))
        socket.binaryType = 'arraybuffer'
        socketRef.current = socket
        let settled = false

        const settleResolve = (): void => {
          if (settled) {
            return
          }
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
          if (isStartCancelled()) {
            socket.close(1000, 'Talk start cancelled')
            settleReject(new Error('Talk start cancelled'))
            return
          }
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
          if (isStartCancelled()) {
            socket.close(1000, 'Talk start cancelled')
            settleReject(new Error('Talk start cancelled'))
            return
          }
          void startAudioPipeline(stream, nextSession.input, socket, () => {
            socket.close(1011, 'Talk backpressure')
            if (mountedRef.current) {
              setError(new Error('Talk audio fell behind; stopped the session.'))
            }
          })
            .then((pipeline) => {
              if (isStartCancelled()) {
                void stopAudioPipeline(pipeline)
                socket.close(1000, 'Talk start cancelled')
                settleReject(new Error('Talk start cancelled'))
                return
              }
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
          if (!intentionalStopRef.current && event.code !== 1000) {
            void apiClient.stopCameraTalkSession(cameraName, nextSession.session_id).catch(() => {})
          }
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
    if (startInFlightRef.current || isStreaming || isStopping) {
      return
    }
    const generation = startGenerationRef.current + 1
    startGenerationRef.current = generation
    startInFlightRef.current = true
    const abortController = new AbortController()
    startAbortRef.current = abortController
    const isStartCancelled = (): boolean => (
      startGenerationRef.current !== generation
      || abortController.signal.aborted
      || !mountedRef.current
    )

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
          autoGainControl: true,
          channelCount: DEFAULT_TALK_INPUT.channels,
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: DEFAULT_TALK_INPUT.sample_rate,
        },
        video: false,
      })
      if (isStartCancelled()) {
        closeMediaStream(mediaStream)
        return
      }
      pendingStreamRef.current = mediaStream
      const nextSession = await apiClient.prepareCameraTalkSession(
        cameraName,
        { input: DEFAULT_TALK_INPUT },
        { signal: abortController.signal },
      )
      preparedSession = nextSession
      if (isStartCancelled()) {
        closeMediaStream(mediaStream)
        pendingStreamRef.current = null
        await apiClient.stopCameraTalkSession(cameraName, nextSession.session_id).catch(() => {})
        return
      }
      setSession(nextSession)
      sessionRef.current = nextSession
      setStatus((previous) => nextStatusFromState(cameraName, 'starting', nextSession.session_id, previous))
      await openTalkSocket(nextSession, mediaStream, isStartCancelled)
      if (isStartCancelled()) {
        await cleanupSocketAndAudio()
        await apiClient.stopCameraTalkSession(cameraName, nextSession.session_id).catch(() => {})
      }
    } catch (nextError) {
      closeMediaStream(mediaStream)
      pendingStreamRef.current = null
      await cleanupSocketAndAudio()
      if (preparedSession && !intentionalStopRef.current) {
        await apiClient.stopCameraTalkSession(cameraName, preparedSession.session_id).catch(() => {})
      }
      if (mountedRef.current && !isStartCancelled()) {
        setError(nextError)
      }
      if (mountedRef.current) {
        setSession(null)
        sessionRef.current = null
        setIsStreaming(false)
      }
    } finally {
      startInFlightRef.current = false
      if (startAbortRef.current === abortController) {
        startAbortRef.current = null
      }
      if (mountedRef.current) {
        setIsStarting(false)
      }
    }
  }, [cameraName, cleanupSocketAndAudio, isStopping, isStreaming, openTalkSocket])

  const stop = useCallback(async () => {
    const activeSession = sessionRef.current
    const hasStartInFlight = startInFlightRef.current
    if (isStopping || (!activeSession && !isStreaming && !isStarting && !hasStartInFlight)) {
      return
    }
    startGenerationRef.current += 1
    startAbortRef.current?.abort()
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
        await apiClient.stopCameraTalkSession(cameraName, activeSession.session_id)
        if (mountedRef.current) {
          setStatus((previous) => nextStatusFromState(cameraName, 'idle', null, previous))
          void refreshStatus()
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
  }, [cameraName, cleanupSocketAndAudio, isStarting, isStopping, isStreaming, refreshStatus])

  useEffect(() => {
    mountedRef.current = true
    void refreshStatus()
    return () => {
      mountedRef.current = false
      startGenerationRef.current += 1
      startAbortRef.current?.abort()
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
