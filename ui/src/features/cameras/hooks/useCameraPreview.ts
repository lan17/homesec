import { useCallback, useEffect, useState } from 'react'
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
}

export function useCameraPreview(cameraName: string): CameraPreviewState {
  const queryClient = useQueryClient()
  const [sessionState, setSessionState] = useState<StoredPreviewSession | null>(null)
  const [refreshError, setRefreshError] = useState<Error | null>(null)
  const session = sessionState?.snapshot ?? null

  const statusQuery = useQuery<PreviewStatusSnapshot>({
    queryKey: QUERY_KEYS.cameraPreview(cameraName),
    queryFn: ({ signal }) => apiClient.getCameraPreviewStatus(cameraName, { signal }),
    staleTime: PREVIEW_STATUS_REFRESH_MS,
    refetchInterval: session ? PREVIEW_STATUS_REFRESH_MS : false,
  })

  const startMutation = useMutation<PreviewSessionSnapshot, Error>({
    mutationFn: () => apiClient.ensureCameraPreviewActive(cameraName),
    onSuccess: async (nextSession) => {
      setRefreshError(null)
      setSessionState({
        snapshot: nextSession,
        receivedAtMs: Date.now(),
      })
      await queryClient.invalidateQueries({ queryKey: QUERY_KEYS.cameraPreview(cameraName) })
    },
  })

  const stopMutation = useMutation<PreviewStopSnapshot, Error>({
    mutationFn: () => apiClient.stopCameraPreview(cameraName),
    onSuccess: async () => {
      setRefreshError(null)
      setSessionState(null)
      await queryClient.invalidateQueries({ queryKey: QUERY_KEYS.cameraPreview(cameraName) })
    },
  })

  const statusIsNewerThanSession =
    sessionState !== null && statusQuery.dataUpdatedAt >= sessionState.receivedAtMs
  const statusState = statusQuery.data?.state
  const newerStatusInvalidatesSession =
    statusIsNewerThanSession
    && (statusQuery.data?.enabled === false
      || (statusState !== undefined
        && !PREVIEW_SESSION_ACTIVE_STATES.has(statusState)
        && !startMutation.isPending))
  const activeSession =
    newerStatusInvalidatesSession
      ? null
      : session

  const refreshSession = useCallback(async () => {
    try {
      const nextSession = await apiClient.ensureCameraPreviewActive(cameraName)
      setRefreshError(null)
      setSessionState({
        snapshot: nextSession,
        receivedAtMs: Date.now(),
      })
      await queryClient.invalidateQueries({ queryKey: QUERY_KEYS.cameraPreview(cameraName) })
    } catch (nextError) {
      setRefreshError(nextError as Error)
    }
  }, [cameraName, queryClient])

  useEffect(() => {
    if (activeSession?.token_expires_at == null || stopMutation.isPending) {
      return
    }

    const expiresAtMs = Date.parse(activeSession.token_expires_at)
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
        : Math.max(0, Math.min(PREVIEW_TOKEN_REFRESH_RETRY_MS, remainingMs))

    const timeoutId = window.setTimeout(() => {
      void refreshSession()
    }, refreshDelayMs)

    return () => {
      window.clearTimeout(timeoutId)
    }
  }, [activeSession?.token_expires_at, refreshError, refreshSession, stopMutation.isPending])

  const warning =
    activeSession?.warning
    ?? statusQuery.data?.degraded_reason
    ?? statusQuery.data?.last_error
    ?? null

  const playlistUrl = activeSession ? apiClient.resolvePath(activeSession.playlist_url) : null
  const error = (startMutation.error
    ?? stopMutation.error
    ?? refreshError
    ?? statusQuery.error
    ?? null) as Error | null

  return {
    status: statusQuery.data ?? null,
    session: activeSession,
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
