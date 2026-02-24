import { describe, expect, it } from 'vitest'

import {
  CAMERA_BACKEND_OPTIONS,
  defaultSourceConfigPatchForCamera,
  defaultSourceConfigForBackend,
  parseSourceConfigPatchJson,
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

describe('parseSourceConfigPatchJson', () => {
  it('rejects patch payloads that include redacted placeholders', () => {
    // Given: A source config patch payload containing UI redaction placeholder text
    const raw = '{"rtsp_url":"rtsp://***redacted***@camera.local/stream"}'

    // When: Parsing source config patch JSON
    const parsed = parseSourceConfigPatchJson(raw)

    // Then: Patch parser rejects placeholder values and prompts for replacement strategy
    expect(parsed).toEqual({
      ok: false,
      message:
        'Source config patch cannot include redacted placeholders. Omit unchanged secret fields or provide replacement values.',
    })
  })

  it('accepts patch payloads with null clears and replacement values', () => {
    // Given: A valid patch payload that clears one key and replaces another
    const raw = '{"detect_rtsp_url":null,"rtsp_url":"rtsp://user:pass@camera.local/live"}'

    // When: Parsing source config patch JSON
    const parsed = parseSourceConfigPatchJson(raw)

    // Then: Parser preserves patch semantics for null-clears and replacements
    expect(parsed).toEqual({
      ok: true,
      value: {
        detect_rtsp_url: null,
        rtsp_url: 'rtsp://user:pass@camera.local/live',
      },
    })
  })
})

describe('defaultSourceConfigPatchForCamera', () => {
  it('drops redacted values from camera source config defaults', () => {
    // Given: A camera source_config object containing redacted and non-secret fields
    const sourceConfig = {
      rtsp_url: 'rtsp://***redacted***@camera.local/stream',
      output_dir: './recordings',
      nested: {
        password: '***redacted***',
        timeout_s: 10,
      },
    }

    // When: Building default patch JSON for editor initialization
    const result = defaultSourceConfigPatchForCamera(sourceConfig)

    // Then: Returned template keeps non-secret values and omits redacted placeholders
    expect(JSON.parse(result)).toEqual({
      output_dir: './recordings',
      nested: {
        timeout_s: 10,
      },
    })
  })
})
