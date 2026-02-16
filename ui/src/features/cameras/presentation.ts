import { isAPIError } from '../../api/client'
import type { RuntimeState } from '../../api/generated/types'

interface RuntimeStatusLike {
  state: RuntimeState
  reload_in_progress: boolean
}

export function describeCameraError(error: unknown): string {
  if (isAPIError(error)) {
    return error.errorCode ? `${error.message} (${error.errorCode})` : error.message
  }
  if (error instanceof Error) {
    return error.message
  }
  return 'Unknown error'
}

export function runtimeStatusTone(status: RuntimeStatusLike): 'degraded' | 'healthy' | 'unhealthy' {
  if (status.state === 'failed') {
    return 'unhealthy'
  }
  if (status.reload_in_progress || status.state === 'reloading') {
    return 'degraded'
  }
  return 'healthy'
}

export function formatRuntimeTimestamp(value: string | null): string {
  if (!value) {
    return 'n/a'
  }
  const parsed = new Date(value)
  if (Number.isNaN(parsed.valueOf())) {
    return value
  }
  return parsed.toLocaleString()
}
