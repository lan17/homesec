import { describe, expect, it } from 'vitest'

import { formatHealthStatusLabel, formatLastUpdated, formatSystemValue, formatUptime, healthTone } from './status'

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
    // When: Formatting for system display
    // Then: A placeholder string should be returned
    expect(formatLastUpdated(0)).toBe('Not yet updated')
  })
})

describe('system presentation helpers', () => {
  it('formats health, runtime, and uptime values for product copy', () => {
    // Given: Existing health and stats API values
    // When: Formatting them for System cards
    // Then: Visible labels should avoid raw casing and unit suffixes
    expect(formatHealthStatusLabel('healthy')).toBe('Healthy')
    expect(formatHealthStatusLabel('other')).toBe('Status unavailable')
    expect(formatSystemValue('pipeline_running')).toBe('Pipeline Running')
    expect(formatUptime(120)).toBe('2 min')
  })
})
