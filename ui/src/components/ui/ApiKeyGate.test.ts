import { describe, expect, it } from 'vitest'

import { toApiKeyGateActionErrorMessage } from './apiKeyGateErrors'

describe('toApiKeyGateActionErrorMessage', () => {
  it('returns thrown Error message when available', () => {
    // Given: A submit failure with explicit error text
    const error = new Error('Unauthorized')

    // When: Mapping the failure to user-facing copy
    const message = toApiKeyGateActionErrorMessage(error)

    // Then: The original message is preserved
    expect(message).toBe('Unauthorized')
  })

  it('returns fallback message for unknown failures', () => {
    // Given: A submit failure without an Error instance
    const error = { code: 'UNKNOWN' }

    // When: Mapping the failure to user-facing copy
    const message = toApiKeyGateActionErrorMessage(error)

    // Then: The stable fallback message is used
    expect(message).toBe('Unable to apply API key. Please try again.')
  })
})
