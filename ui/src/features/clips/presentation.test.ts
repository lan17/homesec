import { describe, expect, it } from 'vitest'

import { APIError } from '../../api/client'
import {
  describeClipError,
  renderDetectedObjects,
  resolveClipExternalLink,
  resolveClipViewLink,
} from './presentation'

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

describe('resolveClipExternalLink', () => {
  it('prefers explicit view_url over storage_uri when both are present', () => {
    // Given: A clip with both a view URL and a storage URI
    const clip = {
      id: 'clip-1',
      camera: 'front_door',
      status: 'done',
      created_at: '2026-02-14T00:00:00.000Z',
      alerted: false,
      view_url: 'https://dropbox.example/view/clip-1',
      storage_uri: 'https://storage.example/clip-1.mp4',
    }

    // When: Resolving external storage link
    const value = resolveClipExternalLink(clip)

    // Then: View URL should be chosen as the canonical playback target
    expect(value).toBe('https://dropbox.example/view/clip-1')
  })

  it('returns null for non-http storage handles', () => {
    // Given: A clip with backend storage handle but no public URL
    const clip = {
      id: 'clip-1',
      camera: 'front_door',
      status: 'done',
      created_at: '2026-02-14T00:00:00.000Z',
      alerted: false,
      storage_uri: 'dropbox:/clips/clip-1.mp4',
      view_url: null,
    }

    // When: Resolving external storage link
    const value = resolveClipExternalLink(clip)

    // Then: UI should treat this as non-playable link without backend view URL
    expect(value).toBeNull()
  })
})

describe('resolveClipViewLink', () => {
  it('returns null for non-http view_url schemes', () => {
    // Given: A clip payload with a non-http view_url
    const clip = {
      id: 'clip-unsafe',
      camera: 'front_door',
      status: 'done',
      created_at: '2026-02-14T00:00:00.000Z',
      alerted: false,
      view_url: 'javascript:alert(1)',
      storage_uri: null,
    }

    // When: Resolving the direct view URL link
    const value = resolveClipViewLink(clip)

    // Then: Unsafe schemes are rejected
    expect(value).toBeNull()
  })
})
