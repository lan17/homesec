import { useQuery } from '@tanstack/react-query'

import { apiClient } from '../client'
import type { CameraListResponse } from '../generated/types'
import { QUERY_KEYS } from './queryKeys'

const CAMERAS_REFRESH_MS = 30_000

export function useCamerasQuery() {
  return useQuery<CameraListResponse>({
    queryKey: QUERY_KEYS.cameras,
    queryFn: ({ signal }) => apiClient.getCameras({ signal }),
    staleTime: CAMERAS_REFRESH_MS,
    refetchInterval: CAMERAS_REFRESH_MS,
  })
}
