import { describe, expect, it } from 'vitest'

import { APIError } from '../../api/client'
import { formatRuntimeTimestamp, runtimeStatusTone, describeCameraError } from './presentation'

describe('describeCameraError', () => {
  it('renders canonical API errors with error_code suffix', () => {
    // Given: A typed APIError carrying a canonical error code
    const error = new APIError('Camera not found', 404, null, 'CAMERA_NOT_FOUND')

    // When: Formatting the error for page display
    const message = describeCameraError(error)

    // Then: Message should include the canonical code for operator context
    expect(message).toBe('Camera not found (CAMERA_NOT_FOUND)')
  })
})

describe('runtimeStatusTone', () => {
  it('returns unhealthy when runtime state is failed', () => {
    // Given: A runtime status with failed state
    const status = {
      state: 'failed',
      generation: 7,
      reload_in_progress: false,
      active_config_version: null,
      last_reload_at: null,
      last_reload_error: 'boot failure',
    } as const

    // When: Mapping runtime status to badge tone
    const tone = runtimeStatusTone(status)

    // Then: UI should mark failed state as unhealthy
    expect(tone).toBe('unhealthy')
  })

  it('returns degraded while reload is in progress', () => {
    // Given: A runtime status actively reloading
    const status = {
      state: 'reloading',
      generation: 3,
      reload_in_progress: true,
      active_config_version: 'abc123',
      last_reload_at: null,
      last_reload_error: null,
    } as const

    // When: Mapping runtime status to badge tone
    const tone = runtimeStatusTone(status)

    // Then: UI should communicate degraded transitional state
    expect(tone).toBe('degraded')
  })

  it('returns healthy for idle runtime even when last reload had an error', () => {
    // Given: A runtime status that recovered to idle after a prior failed reload
    const status = {
      state: 'idle',
      generation: 8,
      reload_in_progress: false,
      active_config_version: 'def456',
      last_reload_at: '2026-02-16T02:10:00.000Z',
      last_reload_error: 'previous failure',
    } as const

    // When: Mapping runtime status to badge tone
    const tone = runtimeStatusTone(status)

    // Then: Tone should reflect current state rather than historical error text
    expect(tone).toBe('healthy')
  })
})

describe('formatRuntimeTimestamp', () => {
  it('formats ISO runtime timestamps for operator readability', () => {
    // Given: A runtime timestamp in ISO-8601 format
    const value = '2026-02-16T01:23:45.000Z'

    // When: Formatting timestamp for display
    const formatted = formatRuntimeTimestamp(value)

    // Then: Output should be non-empty and not the fallback marker
    expect(formatted.length).toBeGreaterThan(0)
    expect(formatted).not.toBe('n/a')
  })

  it('returns fallback for absent runtime timestamp values', () => {
    // Given: No runtime timestamp
    const value = null

    // When: Formatting timestamp for display
    const formatted = formatRuntimeTimestamp(value)

    // Then: Fallback should keep UI deterministic
    expect(formatted).toBe('n/a')
  })
})
