import { describe, expect, it } from 'vitest'

import {
  CAMERA_BACKEND_OPTIONS,
  defaultSourceConfigForBackend,
  parseSourceConfigJson,
} from './forms'

describe('defaultSourceConfigForBackend', () => {
  it('returns valid JSON templates for every backend option', () => {
    // Given: Supported camera source backend options
    const backends = CAMERA_BACKEND_OPTIONS

    // When: Resolving default source config template for each backend
    const parsedTemplates = backends.map((backend) =>
      parseSourceConfigJson(defaultSourceConfigForBackend(backend)),
    )

    // Then: Every backend should produce a parseable JSON object template
    for (const parsed of parsedTemplates) {
      expect(parsed.ok).toBe(true)
    }
  })
})

describe('parseSourceConfigJson', () => {
  it('rejects empty config text', () => {
    // Given: An empty source config field value
    const raw = '   '

    // When: Parsing the source config JSON
    const parsed = parseSourceConfigJson(raw)

    // Then: Parser should return a user-facing validation error
    expect(parsed).toEqual({
      ok: false,
      message: 'Source config JSON is required.',
    })
  })

  it('rejects malformed JSON payloads', () => {
    // Given: A malformed JSON string
    const raw = '{"watch_dir":'

    // When: Parsing the source config JSON
    const parsed = parseSourceConfigJson(raw)

    // Then: Parser should return syntax validation feedback
    expect(parsed).toEqual({
      ok: false,
      message: 'Source config must be valid JSON.',
    })
  })

  it('rejects JSON arrays because source config must be an object', () => {
    // Given: A JSON array payload
    const raw = '["unexpected"]'

    // When: Parsing the source config JSON
    const parsed = parseSourceConfigJson(raw)

    // Then: Parser should reject non-object root values
    expect(parsed).toEqual({
      ok: false,
      message: 'Source config must be a JSON object.',
    })
  })

  it('parses JSON object payloads used for camera create requests', () => {
    // Given: A valid source config JSON object
    const raw = '{"watch_dir":"./recordings","poll_interval":1}'

    // When: Parsing the source config JSON
    const parsed = parseSourceConfigJson(raw)

    // Then: Parser should return a typed object payload
    expect(parsed).toEqual({
      ok: true,
      value: {
        watch_dir: './recordings',
        poll_interval: 1,
      },
    })
  })
})
