import { describe, expect, it } from 'vitest'

import { APIError, extractAPIErrorEnvelope, isAPIError, isUnauthorizedAPIError } from './errors'

describe('extractAPIErrorEnvelope', () => {
  it('reads canonical detail/error_code payloads', () => {
    // Given: A canonical API error envelope payload
    const payload = { detail: 'Unauthorized', error_code: 'UNAUTHORIZED' }

    // When: Extracting the API error envelope
    const envelope = extractAPIErrorEnvelope(payload)

    // Then: Detail and error code are preserved
    expect(envelope).toEqual({ detail: 'Unauthorized', errorCode: 'UNAUTHORIZED' })
  })

  it('falls back to message when detail is absent', () => {
    // Given: A payload using message instead of detail
    const payload = { message: 'Bad request', error_code: 'BAD_REQUEST' }

    // When: Extracting the API error envelope
    const envelope = extractAPIErrorEnvelope(payload)

    // Then: Message is promoted to detail and error code is preserved
    expect(envelope).toEqual({ detail: 'Bad request', errorCode: 'BAD_REQUEST' })
  })

  it('returns default envelope when payload is not recognized', () => {
    // Given: A payload that does not include detail or message
    const payload = { foo: 'bar' }

    // When: Extracting the API error envelope
    const envelope = extractAPIErrorEnvelope(payload)

    // Then: A stable default envelope is returned
    expect(envelope).toEqual({ detail: 'API request failed', errorCode: null })
  })
})

describe('APIError guards', () => {
  it('identifies APIError and unauthorized APIError instances', () => {
    // Given: APIError instances with different HTTP status codes
    const unauthorizedError = new APIError('Unauthorized', 401, null, 'UNAUTHORIZED')
    const genericError = new APIError('Bad request', 400, null, 'BAD_REQUEST')

    // When / Then: Type guards classify by error type and status
    expect(isAPIError(unauthorizedError)).toBe(true)
    expect(isUnauthorizedAPIError(unauthorizedError)).toBe(true)
    expect(isUnauthorizedAPIError(genericError)).toBe(false)
  })
})
