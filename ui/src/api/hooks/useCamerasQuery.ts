import { useQuery } from '@tanstack/react-query'

import { apiClient } from '../client'
import type { CameraListResponse } from '../generated/types'

const CAMERAS_REFRESH_MS = 30_000

export function useCamerasQuery() {
  return useQuery<CameraListResponse>({
    queryKey: ['cameras'],
    queryFn: ({ signal }) => apiClient.getCameras({ signal }),
    staleTime: CAMERAS_REFRESH_MS,
    refetchInterval: CAMERAS_REFRESH_MS,
  })
}
