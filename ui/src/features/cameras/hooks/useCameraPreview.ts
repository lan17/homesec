import { useCallback, useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import {
  apiClient,
  type PreviewSessionSnapshot,
  type PreviewStatusSnapshot,
  type PreviewStopSnapshot,
} from '../../../api/client'
import { QUERY_KEYS } from '../../../api/hooks/queryKeys'

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

export function useCameraPreview(cameraName: string): CameraPreviewState {
  const queryClient = useQueryClient()
  const [sessionState, setSessionState] = useState<StoredPreviewSession | null>(null)
  const [refreshError, setRefreshError] = useState<Error | null>(null)
  const sessionStateRef = useRef<StoredPreviewSession | null>(null)
  const statusRequestSeqRef = useRef(0)

  const storeSession = useCallback((nextSession: PreviewSessionSnapshot) => {
    const nextState = {
      snapshot: nextSession,
      receivedAtMs: Date.now(),
      statusRequestSeq: statusRequestSeqRef.current,
    }
    sessionStateRef.current = nextState
    setSessionState(nextState)
  }, [])

  const clearSession = useCallback(() => {
    sessionStateRef.current = null
    setSessionState(null)
  }, [])

  const startMutation = useMutation<PreviewSessionSnapshot, Error>({
    mutationFn: () => apiClient.ensureCameraPreviewActive(cameraName),
    onSuccess: async (nextSession) => {
      setRefreshError(null)
      storeSession(nextSession)
      await queryClient.invalidateQueries({ queryKey: QUERY_KEYS.cameraPreview(cameraName) })
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
      }
      return nextStatus
    },
    staleTime: PREVIEW_STATUS_REFRESH_MS,
    refetchInterval: sessionState ? PREVIEW_STATUS_REFRESH_MS : false,
  })

  const stopMutation = useMutation<PreviewStopSnapshot, Error>({
    mutationFn: () => apiClient.stopCameraPreview(cameraName),
    onSuccess: async () => {
      setRefreshError(null)
      clearSession()
      await queryClient.invalidateQueries({ queryKey: QUERY_KEYS.cameraPreview(cameraName) })
    },
  })
  const session = sessionState?.snapshot ?? null

  const refreshSession = useCallback(async () => {
    try {
      const nextSession = await apiClient.ensureCameraPreviewActive(cameraName)
      setRefreshError(null)
      storeSession(nextSession)
      await queryClient.invalidateQueries({ queryKey: QUERY_KEYS.cameraPreview(cameraName) })
    } catch (nextError) {
      setRefreshError(nextError as Error)
    }
  }, [cameraName, queryClient, storeSession])

  useEffect(() => {
    if (session?.token_expires_at == null || stopMutation.isPending) {
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
  }, [session?.token_expires_at, refreshError, refreshSession, stopMutation.isPending])

  const warning =
    session?.warning
    ?? statusQuery.data?.degraded_reason
    ?? statusQuery.data?.last_error
    ?? null

  const playlistUrl = session ? apiClient.resolvePath(session.playlist_url) : null
  const error = (startMutation.error
    ?? stopMutation.error
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
      try {
        await startMutation.mutateAsync()
      } catch {
        return
      }
    },
    stop: async () => {
      try {
        await stopMutation.mutateAsync()
      } catch {
        return
      }
    },
    refreshStatus: async () => {
      const result = await statusQuery.refetch()
      return result.data ?? null
    },
  }
}
