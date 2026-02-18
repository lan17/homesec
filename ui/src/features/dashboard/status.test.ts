import { describe, expect, it } from 'vitest'

import { formatLastUpdated, healthTone } from './status'

describe('healthTone', () => {
  it('maps known backend statuses to matching UI tones', () => {
    // Given: A backend health status string
    // When: Mapping to UI tone
    // Then: Known statuses should be preserved
    expect(healthTone('healthy')).toBe('healthy')
    expect(healthTone('degraded')).toBe('degraded')
    expect(healthTone('unhealthy')).toBe('unhealthy')
  })

  it('maps unknown statuses to unknown tone', () => {
    // Given: A backend status outside the expected set
    // When: Mapping to UI tone
    // Then: Tone should be unknown
    expect(healthTone('partial')).toBe('unknown')
  })
})

describe('formatLastUpdated', () => {
  it('returns placeholder text for empty timestamps', () => {
    // Given: No successful query update timestamp
    // When: Formatting for dashboard display
    // Then: A placeholder string should be returned
    expect(formatLastUpdated(0)).toBe('Not yet updated')
  })
})
