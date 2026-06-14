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

interface PreviewStopRequest {
  requestSeq: number
}

export function useCameraPreview(cameraName: string): CameraPreviewState {
  const queryClient = useQueryClient()
  const nativeLifecycle = useNativeAppLifecycleState()
  const nativeLifecycleRef = useRef(nativeLifecycle)
  const [sessionState, setSessionState] = useState<StoredPreviewSession | null>(null)
  const [refreshError, setRefreshError] = useState<Error | null>(null)
  const [stopError, setStopError] = useState<Error | null>(null)
  const sessionStateRef = useRef<StoredPreviewSession | null>(null)
  const statusRequestSeqRef = useRef(0)
  const sessionRequestSeqRef = useRef(0)
  const stopInFlightSeqRef = useRef<number | null>(null)
  const latestStopRequestSeqRef = useRef(0)

  useLayoutEffect(() => {
    nativeLifecycleRef.current = nativeLifecycle
  }, [nativeLifecycle])

  const storeSession = useCallback((nextSession: PreviewSessionSnapshot) => {
    const nextState = {
      snapshot: nextSession,
      receivedAtMs: Date.now(),
      statusRequestSeq: statusRequestSeqRef.current,
    }
    setStopError(null)
    sessionStateRef.current = nextState
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

  const finishStopRequest = useCallback((requestSeq: number) => {
    if (stopInFlightSeqRef.current === requestSeq) {
      stopInFlightSeqRef.current = null
    }
  }, [])

  const storeActivationIfCurrent = useCallback(async (activation: PreviewActivation) => {
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

    if (currentLifecycle.isBackgrounded) {
      await stopLateActivation()
      return
    }

    if (currentLifecycle.pauseCount !== activation.pauseCountAtRequest) {
      if (isLatestActivation) {
        await stopLateActivation()
      }
      return
    }

    if (!isLatestActivation) {
      if (wasSupersededByLatestStop) {
        await stopLateActivation()
      }
      return
    }

    storeSession(activation.snapshot)
    await queryClient.invalidateQueries({ queryKey: QUERY_KEYS.cameraPreview(cameraName) })
  }, [beginStopRequest, cameraName, clearSession, finishStopRequest, queryClient, storeSession])

  const startMutation = useMutation<PreviewActivation, Error>({
    mutationFn: async () => {
      const activationSeq = beginSessionRequest()
      const pauseCountAtRequest = nativeLifecycleRef.current.pauseCount
      const snapshot = await apiClient.ensureCameraPreviewActive(cameraName)
      return { activationSeq, pauseCountAtRequest, snapshot }
    },
    onSuccess: async (activation) => {
      setRefreshError(null)
      await storeActivationIfCurrent(activation)
    },
  })

  const statusQuery = useQuery<PreviewStatusSnapshot>({
    queryKey: QUERY_KEYS.cameraPreview(cameraName),
    queryFn: async ({ signal }) => {
      statusRequestSeqRef.current += 1
      const requestSeq = statusRequestSeqRef.current
      const nextStatus = await apiClient.getCameraPreviewStatus(cameraName, { signal })
      const currentSession = sessionStateRef.current
      if (
        currentSession !== null
        && requestSeq > currentSession.statusRequestSeq
        && (nextStatus.enabled === false
          || (!PREVIEW_SESSION_ACTIVE_STATES.has(nextStatus.state) && !startMutation.isPending))
      ) {
        clearSession()
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
  const session = sessionState?.snapshot ?? null

  const stop = useCallback(async () => {
    const requestSeq = beginStopRequest()
    clearSession()
    setStopError(null)
    try {
      await stopPreview({ requestSeq })
    } catch (nextError) {
      if (requestSeq === sessionRequestSeqRef.current) {
        setStopError(nextError as Error)
      }
      return
    }
  }, [beginStopRequest, clearSession, stopPreview])

  const refreshSession = useCallback(async () => {
    if (nativeLifecycle.isBackgrounded || stopInFlightSeqRef.current !== null) {
      return
    }
    try {
      const activationSeq = beginSessionRequest()
      const pauseCountAtRequest = nativeLifecycleRef.current.pauseCount
      const snapshot = await apiClient.ensureCameraPreviewActive(cameraName)
      setRefreshError(null)
      await storeActivationIfCurrent({ activationSeq, pauseCountAtRequest, snapshot })
    } catch (nextError) {
      setRefreshError(nextError as Error)
    }
  }, [beginSessionRequest, cameraName, nativeLifecycle.isBackgrounded, storeActivationIfCurrent])

  useEffect(() => {
    if (!nativeLifecycle.isBackgrounded || sessionStateRef.current === null) {
      return
    }

    void stop()
  }, [nativeLifecycle.isBackgrounded, stop])

  useEffect(() => {
    if (!nativeLifecycle.isActive || nativeLifecycle.resumeCount === 0) {
      return
    }

    void refetchStatus()
  }, [nativeLifecycle.isActive, nativeLifecycle.resumeCount, refetchStatus])

  useEffect(() => {
    if (nativeLifecycle.isBackgrounded || session?.token_expires_at == null || stopMutation.isPending) {
      return
    }

    const expiresAtMs = Date.parse(session.token_expires_at)
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

    const timeoutId = window.setTimeout(() => {
      void refreshSession()
    }, refreshDelayMs)

    return () => {
      window.clearTimeout(timeoutId)
    }
  }, [
    nativeLifecycle.isBackgrounded,
    session?.token_expires_at,
    refreshError,
    refreshSession,
    stopMutation.isPending,
  ])

  const warning =
    session?.warning
    ?? statusQuery.data?.degraded_reason
    ?? statusQuery.data?.last_error
    ?? null

  const playlistUrl = session ? apiClient.resolvePath(session.playlist_url) : null
  const error = (startMutation.error
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
    start: async () => {
      if (
        !nativeLifecycle.isActive
        || nativeLifecycle.isBackgrounded
        || stopInFlightSeqRef.current !== null
      ) {
        return
      }
      try {
        await startMutation.mutateAsync()
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
