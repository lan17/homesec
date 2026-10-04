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

export function useCameraPreview(cameraName: string): CameraPreviewState {
  const queryClient = useQueryClient()
  const [sessionState, setSessionState] = useState<StoredPreviewSession | null>(null)
  const [knownTransport, setKnownTransport] = useState<Pick<PreviewSessionSnapshot, 'camera_name' | 'transport'> | null>(null)
  const knownTransportRef = useRef(knownTransport)
  const [refreshError, setRefreshError] = useState<Error | null>(null)
  const sessionStateRef = useRef<StoredPreviewSession | null>(null)
  const statusRequestSeqRef = useRef(0)
  const activeCameraRef = useRef(cameraName)
  const sessionRevisionRef = useRef(0)
  const refreshesInFlightRef = useRef(0)

  useEffect(() => {
    activeCameraRef.current = cameraName
    sessionRevisionRef.current += 1
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
    sessionStateRef.current = nextState
    const nextTransport = { camera_name: nextSession.camera_name, transport: nextSession.transport }
    knownTransportRef.current = nextTransport
    setKnownTransport(nextTransport)
    setSessionState(nextState)
  }, [])

  const clearSession = useCallback(() => {
    sessionRevisionRef.current += 1
    sessionStateRef.current = null
    setSessionState(null)
  }, [])

  const startMutation = useMutation<PreviewSessionSnapshot | null, Error>({
    mutationFn: async () => {
      const revision = sessionRevisionRef.current
      const nextSession = await apiClient.ensureCameraPreviewActive(cameraName)
      return revision === sessionRevisionRef.current && cameraName === activeCameraRef.current
        ? nextSession
        : null
    },
    onSuccess: async (nextSession) => {
      if (!nextSession) {
        return
      }
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
      // Background WebRTC peers detach deliberately. An idle publisher does
      // not cancel the user's intent to refresh and reattach when visible.
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
            && !startMutation.isPending
            && refreshesInFlightRef.current === 0))
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
  const session = sessionState?.snapshot.camera_name === cameraName ? sessionState.snapshot : null
  // The publisher transport stays known after this viewer detaches. Scope that
  // identity to the camera so another camera retains its legacy stop behavior.
  const isWebRTC = knownTransport?.camera_name === cameraName && knownTransport.transport === 'webrtc'
  const authorizationExpiresAt = session?.transport === 'webrtc'
    ? session.lease_expires_at ?? session.token_expires_at
    : session?.token_expires_at

  const refreshSession = useCallback(async () => {
    const revision = sessionRevisionRef.current
    refreshesInFlightRef.current += 1
    try {
      const nextSession = await apiClient.ensureCameraPreviewActive(cameraName)
      if (revision !== sessionRevisionRef.current || cameraName !== activeCameraRef.current) {
        return
      }
      setRefreshError(null)
      storeSession(nextSession)
      await queryClient.invalidateQueries({ queryKey: QUERY_KEYS.cameraPreview(cameraName) })
    } catch (nextError) {
      if (revision === sessionRevisionRef.current && cameraName === activeCameraRef.current) {
        setRefreshError(nextError as Error)
      }
    } finally {
      refreshesInFlightRef.current -= 1
    }
  }, [cameraName, queryClient, storeSession])

  useEffect(() => {
    if (authorizationExpiresAt == null || stopMutation.isPending) {
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
        sessionRevisionRef.current += 1
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
  }, [authorizationExpiresAt, session?.transport, refreshError, refreshSession, stopMutation.isPending])

  const warning =
    session?.warning
    ?? statusQuery.data?.degraded_reason
    ?? statusQuery.data?.last_error
    ?? null

  const playlistUrl = session?.playlist_url ? apiClient.resolvePath(session.playlist_url) : null
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
    canStop: session !== null || (!isWebRTC && (statusQuery.data?.state ?? 'idle') !== 'idle'),
    start: async () => {
      try {
        await startMutation.mutateAsync()
      } catch {
        return
      }
    },
    stop: async () => {
      sessionRevisionRef.current += 1
      const transport = knownTransportRef.current
      if (transport?.camera_name === cameraName && transport.transport === 'webrtc') {
        // WebRTC player cleanup closes this viewer; other viewers keep watching.
        clearSession()
        setRefreshError(null)
        return
      }
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
