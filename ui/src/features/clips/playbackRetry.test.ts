import { describe, expect, it } from 'vitest'

import {
  nextAttemptsAfterMediaSourceChange,
  shouldRefreshPlaybackSource,
} from './playbackRetry'

describe('shouldRefreshPlaybackSource', () => {
  it('allows one token-based retry when media url is present', () => {
    // Given: Tokenized playback with a valid media source and zero retries used
    const mediaUrl = '/api/v1/clips/clip-1/media?token=abc'
    const usesToken = true
    const attempts = 0

    // When: Evaluating whether refresh should be attempted
    const allowed = shouldRefreshPlaybackSource(mediaUrl, usesToken, attempts)

    // Then: Refresh is allowed for the first retry
    expect(allowed).toBe(true)
  })

  it('rejects refresh once max retry count is reached', () => {
    // Given: Tokenized playback with one retry already consumed
    const mediaUrl = '/api/v1/clips/clip-1/media?token=abc'
    const usesToken = true
    const attempts = 1

    // When: Evaluating whether refresh should be attempted
    const allowed = shouldRefreshPlaybackSource(mediaUrl, usesToken, attempts)

    // Then: No further retries are allowed
    expect(allowed).toBe(false)
  })

  it('rejects refresh for non-token playback', () => {
    // Given: Direct playback URL without token mode
    const mediaUrl = '/api/v1/clips/clip-1/media'
    const usesToken = false
    const attempts = 0

    // When: Evaluating whether refresh should be attempted
    const allowed = shouldRefreshPlaybackSource(mediaUrl, usesToken, attempts)

    // Then: Refresh is skipped because retry logic is token-only
    expect(allowed).toBe(false)
  })
})

describe('nextAttemptsAfterMediaSourceChange', () => {
  it('resets retry attempts when media source changes', () => {
    // Given: Existing retry count and a new media URL after refresh
    const previousMediaUrl = '/api/v1/clips/clip-1/media?token=old'
    const currentMediaUrl = '/api/v1/clips/clip-1/media?token=new'
    const attempts = 1

    // When: Computing attempts for the new source
    const nextAttempts = nextAttemptsAfterMediaSourceChange(
      previousMediaUrl,
      currentMediaUrl,
      attempts,
    )

    // Then: Retry count resets for the new media source
    expect(nextAttempts).toBe(0)
  })

  it('retains retry attempts when media source is unchanged', () => {
    // Given: Existing retry count and unchanged media URL
    const previousMediaUrl = '/api/v1/clips/clip-1/media?token=same'
    const currentMediaUrl = '/api/v1/clips/clip-1/media?token=same'
    const attempts = 1

    // When: Computing attempts for the same source
    const nextAttempts = nextAttemptsAfterMediaSourceChange(
      previousMediaUrl,
      currentMediaUrl,
      attempts,
    )

    // Then: Retry count is preserved
    expect(nextAttempts).toBe(1)
  })
})
