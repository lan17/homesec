import { apiClient } from '../../api/client'
import type {
  DiscoverRequest,
  DiscoveredCameraResponse,
  ProbeRequest,
  ProbeResponse,
} from '../../api/generated/types'

export async function discoverOnvifCameras(
  payload: DiscoverRequest = { timeout_s: 8.0, attempts: 2, ttl: 4 },
): Promise<DiscoveredCameraResponse[]> {
  return apiClient.discoverOnvifCameras(payload)
}

export async function probeOnvifCamera(payload: ProbeRequest): Promise<ProbeResponse> {
  return apiClient.probeOnvifCamera(payload)
}
