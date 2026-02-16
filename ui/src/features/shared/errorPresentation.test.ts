import { describe, expect, it } from 'vitest'

import { APIError } from '../../api/client'
import { describeAPIError, describeUnknownError } from './errorPresentation'

describe('describeAPIError', () => {
  it('appends canonical error code when available', () => {
    // Given: APIError containing canonical error code
    const error = new APIError('Unauthorized', 401, null, 'UNAUTHORIZED')

    // When: Formatting API error text
    const message = describeAPIError(error)

    // Then: Message includes canonical error code suffix
    expect(message).toBe('Unauthorized (UNAUTHORIZED)')
  })
})

describe('describeUnknownError', () => {
  it('formats APIError instances via shared API formatter', () => {
    // Given: APIError without canonical code
    const error = new APIError('Bad request', 400, null, null)

    // When: Formatting unknown error value
    const message = describeUnknownError(error)

    // Then: APIError message is preserved
    expect(message).toBe('Bad request')
  })

  it('returns default message for non-Error values', () => {
    // Given: Unknown non-Error failure object
    const error = { status: 'broken' }

    // When: Formatting unknown error value
    const message = describeUnknownError(error)

    // Then: Stable default message is returned
    expect(message).toBe('Unknown error')
  })
})
