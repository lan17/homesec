import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import {
  apiClient,
  type PreviewSessionSnapshot,
  type PreviewStatusSnapshot,
  type PreviewStopSnapshot,
} from '../../../api/client'
import { QUERY_KEYS } from '../../../api/hooks/queryKeys'

const PREVIEW_STATUS_REFRESH_MS = 5_000

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

export function useCameraPreview(cameraName: string): CameraPreviewState {
  const queryClient = useQueryClient()
  const [session, setSession] = useState<PreviewSessionSnapshot | null>(null)

  const statusQuery = useQuery<PreviewStatusSnapshot>({
    queryKey: QUERY_KEYS.cameraPreview(cameraName),
    queryFn: ({ signal }) => apiClient.getCameraPreviewStatus(cameraName, { signal }),
    staleTime: PREVIEW_STATUS_REFRESH_MS,
    refetchInterval: session ? PREVIEW_STATUS_REFRESH_MS : false,
  })

  const startMutation = useMutation<PreviewSessionSnapshot, Error>({
    mutationFn: () => apiClient.ensureCameraPreviewActive(cameraName),
    onSuccess: async (nextSession) => {
      setSession(nextSession)
      await queryClient.invalidateQueries({ queryKey: QUERY_KEYS.cameraPreview(cameraName) })
    },
  })

  const stopMutation = useMutation<PreviewStopSnapshot, Error>({
    mutationFn: () => apiClient.stopCameraPreview(cameraName),
    onSuccess: async () => {
      setSession(null)
      await queryClient.invalidateQueries({ queryKey: QUERY_KEYS.cameraPreview(cameraName) })
    },
  })

  const activeSession =
    statusQuery.data?.enabled === false
    || (statusQuery.data?.state === 'idle' && !startMutation.isPending)
      ? null
      : session

  const warning =
    activeSession?.warning
    ?? statusQuery.data?.degraded_reason
    ?? statusQuery.data?.last_error
    ?? null

  const playlistUrl = activeSession ? apiClient.resolvePath(activeSession.playlist_url) : null
  const error = (startMutation.error ?? stopMutation.error ?? statusQuery.error ?? null) as Error | null

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
      await startMutation.mutateAsync()
    },
    stop: async () => {
      await stopMutation.mutateAsync()
    },
    refreshStatus: async () => {
      const result = await statusQuery.refetch()
      return result.data ?? null
    },
  }
}
