import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import {
  apiClient,
  type PreviewSessionSnapshot,
  type PreviewStatusSnapshot,
  type PreviewStopSnapshot,
} from '../../../api/client'
import { QUERY_KEYS } from '../../../api/hooks/queryKeys'
import { useNativeAppLifecycleState } from '../../../runtime/nativeAppLifecycle'

const PREVIEW_STATUS_REFRESH_MS = 5_000
const PREVIEW_TOKEN_REFRESH_LEEWAY_MS = 5_000
const PREVIEW_TOKEN_MIN_REFRESH_LEEWAY_MS = 250
const PREVIEW_TOKEN_REFRESH_RETRY_MS = 1_000
const PREVIEW_SESSION_ACTIVE_STATES = new Set(['starting', 'ready', 'degraded'])

export interface CameraPreviewState {
  status: PreviewStatusSnapshot | null
  session: PreviewSessionSnapshot | null
  playlistUrl: string | null
  warning: string | null
  error: Error | null
  isPending: boolean
  isStarting: boolean
  isStopping: boolean
  canStop: boolean
  start: () => Promise<void>
  stop: () => Promise<void>
  refreshStatus: () => Promise<PreviewStatusSnapshot | null>
}

interface StoredPreviewSession {
  snapshot: PreviewSessionSnapshot
  receivedAtMs: number
  statusRequestSeq: number
}

interface PreviewActivation {
  activationSeq: number
  pauseCountAtRequest: number
  snapshot: PreviewSessionSnapshot
}

interface PreviewActivationRequest {
  activationSeq: number
  pauseCountAtRequest: number
}

interface PreviewStopRequest {
  requestSeq: number
}

export function useCameraPreview(cameraName: string): CameraPreviewState {
  const queryClient = useQueryClient()
  const nativeLifecycle = useNativeAppLifecycleState()
  const nativeLifecycleRef = useRef(nativeLifecycle)
  const handledPauseCountRef = useRef(nativeLifecycle.pauseCount)
  const [sessionState, setSessionState] = useState<StoredPreviewSession | null>(null)
  const [knownTransport, setKnownTransport] = useState<Pick<PreviewSessionSnapshot, 'camera_name' | 'transport'> | null>(null)
  const knownTransportRef = useRef(knownTransport)
  const [startError, setStartError] = useState<Error | null>(null)
  const [refreshError, setRefreshError] = useState<Error | null>(null)
  const [stopError, setStopError] = useState<Error | null>(null)
  const sessionStateRef = useRef<StoredPreviewSession | null>(null)
  const statusRequestSeqRef = useRef(0)
  const sessionRequestSeqRef = useRef(0)
  const stopInFlightSeqRef = useRef<number | null>(null)
  const latestStopRequestSeqRef = useRef(0)
  const activeCameraRef = useRef(cameraName)
  const refreshesInFlightRef = useRef(0)

  useLayoutEffect(() => {
    nativeLifecycleRef.current = nativeLifecycle
  }, [nativeLifecycle])

  useLayoutEffect(() => {
    activeCameraRef.current = cameraName
    sessionRequestSeqRef.current += 1

    return () => {
      // Detached viewers cannot accept or clean up responses from their old connection.
      sessionRequestSeqRef.current += 1
    }
  }, [cameraName])

  const storeSession = useCallback((nextSession: PreviewSessionSnapshot) => {
    if (nextSession.camera_name !== activeCameraRef.current) {
      return
    }
    const nextState = {
      snapshot: nextSession,
      receivedAtMs: Date.now(),
      statusRequestSeq: statusRequestSeqRef.current,
    }
    setStopError(null)
    setStartError(null)
    sessionStateRef.current = nextState
    const nextTransport = { camera_name: nextSession.camera_name, transport: nextSession.transport }
    knownTransportRef.current = nextTransport
    setKnownTransport(nextTransport)
    setSessionState(nextState)
  }, [])

  const clearSession = useCallback(() => {
    sessionStateRef.current = null
    setSessionState(null)
  }, [])

  const beginSessionRequest = useCallback(() => {
    const requestSeq = sessionRequestSeqRef.current + 1
    sessionRequestSeqRef.current = requestSeq
    return requestSeq
  }, [])

  const beginStopRequest = useCallback(() => {
    const requestSeq = beginSessionRequest()
    stopInFlightSeqRef.current = requestSeq
    latestStopRequestSeqRef.current = requestSeq
    return requestSeq
  }, [beginSessionRequest])

  const beginCleanupBoundary = useCallback(() => {
    const requestSeq = beginSessionRequest()
    latestStopRequestSeqRef.current = requestSeq
    return requestSeq
  }, [beginSessionRequest])

  const finishStopRequest = useCallback((requestSeq: number) => {
    if (stopInFlightSeqRef.current === requestSeq) {
      stopInFlightSeqRef.current = null
    }
  }, [])

  const storeActivationIfCurrent = useCallback(async (activation: PreviewActivation) => {
    if (activation.snapshot.camera_name !== activeCameraRef.current) {
      return
    }
    const isLatestActivation = activation.activationSeq === sessionRequestSeqRef.current
    const wasSupersededByLatestStop =
      !isLatestActivation
      && activation.activationSeq < latestStopRequestSeqRef.current
      && sessionRequestSeqRef.current === latestStopRequestSeqRef.current
    const currentLifecycle = nativeLifecycleRef.current

    const stopLateActivation = async (): Promise<void> => {
      clearSession()
      const stopRequestSeq = beginStopRequest()
      try {
        await apiClient.stopCameraPreview(cameraName)
        setRefreshError(null)
        setStopError(null)
        await queryClient.invalidateQueries({ queryKey: QUERY_KEYS.cameraPreview(cameraName) })
      } catch (nextError) {
        setStopError(nextError as Error)
        await queryClient.invalidateQueries({ queryKey: QUERY_KEYS.cameraPreview(cameraName) })
        return
      } finally {
        finishStopRequest(stopRequestSeq)
      }
    }

    if (
      currentLifecycle.isBackgrounded
      || currentLifecycle.pauseCount !== activation.pauseCountAtRequest
    ) {
      if (isLatestActivation) {
        clearSession()
      }
      return
    }

    if (!isLatestActivation) {
      if (wasSupersededByLatestStop && activation.snapshot.transport !== 'webrtc') {
        await stopLateActivation()
      }
      return
    }

    storeSession(activation.snapshot)
    await queryClient.invalidateQueries({ queryKey: QUERY_KEYS.cameraPreview(cameraName) })
  }, [beginStopRequest, cameraName, clearSession, finishStopRequest, queryClient, storeSession])

  const startMutation = useMutation<PreviewActivation, Error, PreviewActivationRequest>({
    mutationFn: async ({ activationSeq, pauseCountAtRequest }) => {
      const snapshot = await apiClient.ensureCameraPreviewActive(cameraName)
      return { activationSeq, pauseCountAtRequest, snapshot }
    },
    onSuccess: async (activation) => {
      if (activation.activationSeq === sessionRequestSeqRef.current) {
        setStartError(null)
        setRefreshError(null)
      }
      await storeActivationIfCurrent(activation)
    },
    onError: (nextError, activation) => {
      if (activation.activationSeq === sessionRequestSeqRef.current && !nativeLifecycleRef.current.isBackgrounded) {
        setStartError(nextError)
      }
    },
  })

  const statusQuery = useQuery<PreviewStatusSnapshot>({
    queryKey: QUERY_KEYS.cameraPreview(cameraName),
    queryFn: async ({ signal }) => {
      statusRequestSeqRef.current += 1
      const requestSeq = statusRequestSeqRef.current
      const nextStatus = await apiClient.getCameraPreviewStatus(cameraName, { signal })
      const currentSession = sessionStateRef.current
      // Browser WebRTC peers detach while hidden and reattach when visible.
      const idleWhileBackgrounded = nextStatus.state === 'idle'
        && document.visibilityState === 'hidden'
        && currentSession?.snapshot.transport === 'webrtc'
      if (
        currentSession !== null
        && currentSession.snapshot.camera_name === cameraName
        && requestSeq > currentSession.statusRequestSeq
        && (nextStatus.enabled === false
          || (!PREVIEW_SESSION_ACTIVE_STATES.has(nextStatus.state)
            && !idleWhileBackgrounded
            && (nextStatus.state !== 'idle'
              || (!startMutation.isPending && refreshesInFlightRef.current === 0))))
      ) {
        beginCleanupBoundary()
        clearSession()
        setStartError(null)
        setRefreshError(null)
        setStopError(null)
      }
      return nextStatus
    },
    staleTime: PREVIEW_STATUS_REFRESH_MS,
    enabled: nativeLifecycle.isActive,
    refetchInterval: nativeLifecycle.isActive && sessionState ? PREVIEW_STATUS_REFRESH_MS : false,
  })

  const stopMutation = useMutation<PreviewStopSnapshot, Error, PreviewStopRequest>({
    mutationFn: () => apiClient.stopCameraPreview(cameraName),
    onSuccess: async (_snapshot, request) => {
      if (request.requestSeq !== sessionRequestSeqRef.current) {
        return
      }
      setStartError(null)
      setRefreshError(null)
      setStopError(null)
      clearSession()
      await queryClient.invalidateQueries({ queryKey: QUERY_KEYS.cameraPreview(cameraName) })
    },
    onSettled: (_snapshot, _error, request) => {
      finishStopRequest(request.requestSeq)
    },
  })
  const refetchStatus = statusQuery.refetch
  const stopPreview = stopMutation.mutateAsync
  const session = sessionState?.snapshot.camera_name === cameraName ? sessionState.snapshot : null
  const isWebRTC = knownTransport?.camera_name === cameraName && knownTransport.transport === 'webrtc'
  const authorizationExpiresAt = session?.transport === 'webrtc'
    ? session.lease_expires_at ?? session.token_expires_at
    : session?.token_expires_at

  const stop = useCallback(async () => {
    const requestSeq = beginStopRequest()
    clearSession()
    setStartError(null)
    setStopError(null)
    setRefreshError(null)
    const transport = knownTransportRef.current
    if (transport?.camera_name === cameraName && transport.transport === 'webrtc') {
      // Player cleanup closes this viewer without interrupting the publisher.
      finishStopRequest(requestSeq)
      return
    }
    try {
      await stopPreview({ requestSeq })
    } catch (nextError) {
      if (requestSeq === sessionRequestSeqRef.current) {
        setStopError(nextError as Error)
      }
      return
    }
  }, [beginStopRequest, cameraName, clearSession, finishStopRequest, stopPreview])

  const refreshSession = useCallback(async () => {
    if (nativeLifecycle.isBackgrounded || stopInFlightSeqRef.current !== null) {
      return
    }
    const activationSeq = beginSessionRequest()
    const pauseCountAtRequest = nativeLifecycleRef.current.pauseCount
    refreshesInFlightRef.current += 1
    try {
      const snapshot = await apiClient.ensureCameraPreviewActive(cameraName)
      if (activationSeq === sessionRequestSeqRef.current) {
        setRefreshError(null)
      }
      await storeActivationIfCurrent({ activationSeq, pauseCountAtRequest, snapshot })
    } catch (nextError) {
      if (activationSeq === sessionRequestSeqRef.current && !nativeLifecycleRef.current.isBackgrounded) {
        setRefreshError(nextError as Error)
      }
    } finally {
      refreshesInFlightRef.current -= 1
    }
  }, [beginSessionRequest, cameraName, nativeLifecycle.isBackgrounded, storeActivationIfCurrent])

  useEffect(() => {
    const pausedSinceLastCleanup = nativeLifecycle.pauseCount > handledPauseCountRef.current
    handledPauseCountRef.current = nativeLifecycle.pauseCount
    if (!nativeLifecycle.isBackgrounded && !pausedSinceLastCleanup) {
      return
    }

    // The publisher is shared. Detach this player and stop renewing its token;
    // existing viewer activity and idle expiry reclaim unused server resources.
    beginSessionRequest()
    clearSession()
  }, [beginSessionRequest, clearSession, nativeLifecycle.isBackgrounded, nativeLifecycle.pauseCount])

  useEffect(() => {
    if (!nativeLifecycle.isActive || nativeLifecycle.resumeCount === 0) {
      return
    }

    void refetchStatus()
  }, [nativeLifecycle.isActive, nativeLifecycle.resumeCount, refetchStatus])

  useEffect(() => {
    if (nativeLifecycle.isBackgrounded || authorizationExpiresAt == null || stopMutation.isPending) {
      return
    }

    const expiresAtMs = Date.parse(authorizationExpiresAt)
    if (Number.isNaN(expiresAtMs)) {
      return
    }

    const remainingMs = expiresAtMs - Date.now()
    const refreshDelayMs =
      refreshError === null
        ? Math.max(
            0,
            remainingMs
              - Math.max(
                  PREVIEW_TOKEN_MIN_REFRESH_LEEWAY_MS,
                  Math.min(PREVIEW_TOKEN_REFRESH_LEEWAY_MS, remainingMs / 2),
                ),
          )
        : remainingMs > 0
          ? Math.min(PREVIEW_TOKEN_REFRESH_RETRY_MS, remainingMs)
          : PREVIEW_TOKEN_REFRESH_RETRY_MS

    const isWebRTC = session?.transport === 'webrtc'
    let timeoutId: number | null = null
    if (!isWebRTC || document.visibilityState !== 'hidden') {
      timeoutId = window.setTimeout(() => {
        if (!isWebRTC || document.visibilityState !== 'hidden') {
          void refreshSession()
        }
      }, refreshDelayMs)
    }
    const onVisibility = (): void => {
      if (document.visibilityState === 'hidden') {
        beginSessionRequest()
        if (timeoutId !== null) {
          window.clearTimeout(timeoutId)
          timeoutId = null
        }
      } else {
        void refreshSession()
      }
    }
    if (isWebRTC) {
      document.addEventListener('visibilitychange', onVisibility)
    }

    return () => {
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId)
      }
      if (isWebRTC) {
        document.removeEventListener('visibilitychange', onVisibility)
      }
    }
  }, [
    nativeLifecycle.isBackgrounded,
    authorizationExpiresAt,
    session?.transport,
    beginSessionRequest,
    refreshError,
    refreshSession,
    stopMutation.isPending,
  ])

  const warning =
    session?.warning
    ?? statusQuery.data?.degraded_reason
    ?? statusQuery.data?.last_error
    ?? null

  const playlistUrl = session?.playlist_url ? apiClient.resolvePath(session.playlist_url) : null
  const error = (startError
    ?? stopError
    ?? refreshError
    ?? statusQuery.error
    ?? null) as Error | null

  return {
    status: statusQuery.data ?? null,
    session,
    playlistUrl,
    warning,
    error,
    isPending:
      (statusQuery.isPending && statusQuery.data === undefined)
      || startMutation.isPending
      || stopMutation.isPending,
    isStarting: startMutation.isPending,
    isStopping: stopMutation.isPending,
    canStop: session !== null || (!isWebRTC && (statusQuery.data?.state ?? 'idle') !== 'idle'),
    start: async () => {
      if (
        !nativeLifecycle.isActive
        || nativeLifecycle.isBackgrounded
        || stopInFlightSeqRef.current !== null
      ) {
        return
      }
      const activationSeq = beginSessionRequest()
      const pauseCountAtRequest = nativeLifecycleRef.current.pauseCount
      setStartError(null)
      try {
        await startMutation.mutateAsync({ activationSeq, pauseCountAtRequest })
      } catch {
        return
      }
    },
    stop,
    refreshStatus: async () => {
      if (nativeLifecycle.isBackgrounded) {
        return null
      }
      const result = await refetchStatus()
      return result.data ?? null
    },
  }
}
