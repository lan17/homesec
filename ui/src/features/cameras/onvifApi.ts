import { apiClient } from '../../api/client'
import type {
  DiscoverRequest,
  DiscoveredCameraResponse,
  ProbeRequest,
  ProbeResponse,
} from '../../api/generated/types'
import { ONVIF_DEFAULT_DISCOVER_REQUEST } from './onvifDefaults'

export async function discoverOnvifCameras(
  payload: DiscoverRequest = ONVIF_DEFAULT_DISCOVER_REQUEST,
): Promise<DiscoveredCameraResponse[]> {
  return apiClient.discoverOnvifCameras(payload)
}

export async function probeOnvifCamera(payload: ProbeRequest): Promise<ProbeResponse> {
  return apiClient.probeOnvifCamera(payload)
}
