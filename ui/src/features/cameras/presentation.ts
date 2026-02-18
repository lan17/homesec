import type { RuntimeState } from '../../api/generated/types'
import { describeUnknownError } from '../shared/errorPresentation'

interface RuntimeStatusLike {
  state: RuntimeState
  reload_in_progress: boolean
}

export function describeCameraError(error: unknown): string {
  return describeUnknownError(error)
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
