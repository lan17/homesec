import { describe, expect, it } from 'vitest'

import { buildDirectMediaPath, computeTokenRefreshDelayMs, toDateOrNull } from './useClipMediaUrl'

describe('buildDirectMediaPath', () => {
  it('encodes clip IDs for media endpoint paths', () => {
    // Given: A clip ID containing characters that need URL encoding
    const clipId = 'garage/clip 1'

    // When: Building direct media path
    const path = buildDirectMediaPath(clipId)

    // Then: Path should be correctly encoded for route safety
    expect(path).toBe('/api/v1/clips/garage%2Fclip%201/media')
  })
})

describe('toDateOrNull', () => {
  it('returns Date for valid timestamp strings', () => {
    // Given: A valid ISO timestamp
    const iso = '2026-02-15T20:00:00.000Z'

    // When: Parsing date string
    const value = toDateOrNull(iso)

    // Then: Date object should be returned
    expect(value).not.toBeNull()
    expect(value?.toISOString()).toBe(iso)
  })

  it('returns null for invalid timestamps', () => {
    // Given: A malformed timestamp value
    const invalid = 'not-a-time'

    // When: Parsing malformed timestamp
    const value = toDateOrNull(invalid)

    // Then: Parser should return null instead of throwing
    expect(value).toBeNull()
  })
})

describe('computeTokenRefreshDelayMs', () => {
  it('computes preemptive refresh delay from expiry', () => {
    // Given: Expiry in 65 seconds and 60-second refresh lead
    const now = Date.parse('2026-02-15T20:00:00.000Z')
    const expiresAt = '2026-02-15T20:01:05.000Z'

    // When: Computing refresh delay
    const delay = computeTokenRefreshDelayMs(expiresAt, now)

    // Then: Refresh should be scheduled 5 seconds from now
    expect(delay).toBe(5_000)
  })

  it('returns null for unparseable expiry values', () => {
    // Given: Invalid expiry string
    const expiresAt = 'invalid-expiry'

    // When: Computing refresh delay
    const delay = computeTokenRefreshDelayMs(expiresAt, Date.now())

    // Then: Delay should be null to skip scheduling
    expect(delay).toBeNull()
  })
})
