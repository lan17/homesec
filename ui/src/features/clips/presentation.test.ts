import { describe, expect, it } from 'vitest'

import { APIError } from '../../api/client'
import { describeClipError, renderDetectedObjects } from './presentation'

describe('describeClipError', () => {
  it('formats APIError with canonical error code', () => {
    // Given: An API error with canonical error_code metadata
    const error = new APIError('Invalid cursor', 400, { detail: 'Invalid cursor' }, 'CLIPS_BAD_CURSOR')

    // When: Formatting error for clips page rendering
    const message = describeClipError(error)

    // Then: Error code should be appended for clear operator context
    expect(message).toBe('Invalid cursor (CLIPS_BAD_CURSOR)')
  })
})

describe('renderDetectedObjects', () => {
  it('returns fallback text when detected objects are absent', () => {
    // Given: A clip payload without detected_objects field
    const clip = {
      id: 'clip-1',
      camera: 'front_door',
      status: 'done',
      created_at: '2026-02-14T00:00:00.000Z',
      alerted: false,
    }

    // When: Rendering detected objects label
    const value = renderDetectedObjects(clip)

    // Then: UI should show explicit empty-state text
    expect(value).toBe('None')
  })
})
