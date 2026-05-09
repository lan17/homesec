import { afterEach, describe, expect, it, vi } from 'vitest'

import type { CameraResponse } from '../../api/generated/types'
import {
  cameraHealthLabel,
  cameraHealthTone,
  cameraIssueSummary,
  countOfflineCameras,
  formatLastSeen,
} from './cameraHealth'

function makeCamera(overrides: Partial<CameraResponse> = {}): CameraResponse {
  return {
    name: 'front_door',
    enabled: true,
    healthy: true,
    last_heartbeat: 1_770_000_000,
    source_backend: 'rtsp',
    source_config: {},
    ...overrides,
  }
}

describe('cameraHealth presentation helpers', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it('derives simple homeowner labels from existing camera fields only', () => {
    // Given: Cameras with the existing enabled and healthy fields
    const online = makeCamera()
    const disabled = makeCamera({ enabled: false, healthy: false })
    const offline = makeCamera({ enabled: true, healthy: false })

    // When: Labels and badge tones are derived
    const labels = [
      cameraHealthLabel(online),
      cameraHealthLabel(disabled),
      cameraHealthLabel(offline),
    ]

    // Then: The UI uses only simple M1 health states
    expect(labels).toEqual(['Online', 'Disabled', 'Offline'])
    expect(cameraHealthTone(online)).toBe('healthy')
    expect(cameraHealthTone(disabled)).toBe('unknown')
    expect(cameraHealthTone(offline)).toBe('unhealthy')
  })

  it('counts only enabled unhealthy cameras as offline issues', () => {
    // Given: A mixed camera list with disabled and offline cameras
    const cameras = [
      makeCamera({ name: 'front' }),
      makeCamera({ name: 'garage', enabled: false, healthy: false }),
      makeCamera({ name: 'driveway', enabled: true, healthy: false }),
    ]

    // When: The compact shell issue summary is computed
    const offlineCount = countOfflineCameras(cameras)
    const summary = cameraIssueSummary(cameras)

    // Then: Disabled cameras are not treated as active offline problems
    expect(offlineCount).toBe(1)
    expect(summary).toBe('1 camera offline')
  })

  it('formats last-seen text without adding backend reason codes', () => {
    // Given: A stable current time and existing last_heartbeat values
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2026-02-14T12:10:00.000Z'))
    const heartbeat = Date.parse('2026-02-14T12:05:00.000Z') / 1000

    // When: Last-seen display text is formatted
    const seen = formatLastSeen(heartbeat)
    const unavailable = formatLastSeen(null)

    // Then: The UI shows conservative time/status text only
    expect(seen).toBe('5 minutes ago')
    expect(unavailable).toBe('Status unavailable')

  })
})
