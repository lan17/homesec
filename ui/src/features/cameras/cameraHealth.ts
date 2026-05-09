import type { CameraResponse } from '../../api/generated/types'
import type { StatusBadgeTone } from '../../components/ui/StatusBadge'

export function cameraHealthLabel(camera: CameraResponse): 'Online' | 'Disabled' | 'Offline' {
  if (!camera.enabled) {
    return 'Disabled'
  }
  return camera.healthy ? 'Online' : 'Offline'
}

export function cameraHealthTone(camera: CameraResponse): StatusBadgeTone {
  if (!camera.enabled) {
    return 'unknown'
  }
  return camera.healthy ? 'healthy' : 'unhealthy'
}

export function countOfflineCameras(cameras: readonly CameraResponse[] | undefined): number {
  return cameras?.filter((camera) => camera.enabled && !camera.healthy).length ?? 0
}

export function cameraIssueSummary(cameras: readonly CameraResponse[] | undefined): string | null {
  const offlineCount = countOfflineCameras(cameras)
  if (offlineCount === 0) {
    return null
  }
  return offlineCount === 1 ? '1 camera offline' : `${offlineCount} cameras offline`
}

export function formatLastSeen(value: number | null, nowMs = Date.now()): string {
  if (!value) {
    return 'Status unavailable'
  }

  const heartbeatMs = value * 1000
  const elapsedMs = Math.max(0, nowMs - heartbeatMs)
  const elapsedMinutes = Math.floor(elapsedMs / 60_000)

  if (elapsedMinutes < 1) {
    return 'just now'
  }
  if (elapsedMinutes < 60) {
    return elapsedMinutes === 1 ? '1 minute ago' : `${elapsedMinutes} minutes ago`
  }

  const elapsedHours = Math.floor(elapsedMinutes / 60)
  if (elapsedHours < 24) {
    return elapsedHours === 1 ? '1 hour ago' : `${elapsedHours} hours ago`
  }

  return new Date(heartbeatMs).toLocaleString()
}
