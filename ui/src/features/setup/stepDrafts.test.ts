import { describe, expect, it } from 'vitest'

import {
  parseCameraStepDraft,
  parseDetectionStepDraft,
  parseStorageStepDraft,
} from './stepDrafts'

describe('stepDrafts', () => {
  it('parses camera draft object from wizard step payload', () => {
    // Given: Wizard camera step data with nested camera payload
    const value = {
      camera: {
        name: 'front_door',
        enabled: true,
        source_backend: 'rtsp',
        source_config: { rtsp_url: 'rtsp://camera.local/stream' },
      },
    }

    // When: Parsing camera step draft
    const parsed = parseCameraStepDraft(value)

    // Then: Camera payload is returned for wizard hydration
    expect(parsed).toMatchObject({
      name: 'front_door',
      source_backend: 'rtsp',
    })
  })

  it('parses storage draft only for supported storage backends', () => {
    // Given: Valid and invalid storage draft payloads
    const valid = { storage: { backend: 'dropbox', config: { root: '/homesec' } } }
    const invalid = { storage: { backend: 's3', config: { bucket: 'clips' } } }

    // When: Parsing storage drafts
    const parsedValid = parseStorageStepDraft(valid)
    const parsedInvalid = parseStorageStepDraft(invalid)

    // Then: Supported backends are accepted and unknown backends are rejected
    expect(parsedValid).toEqual({
      backend: 'dropbox',
      config: { root: '/homesec' },
    })
    expect(parsedInvalid).toBeNull()
  })

  it('parses detection draft only when filter/vlm shape is valid', () => {
    // Given: Valid and malformed detection draft payloads
    const valid = {
      detection: {
        filter: {
          backend: 'yolo',
          config: { classes: ['person'], min_confidence: 0.5 },
        },
        vlm: {
          backend: 'openai',
          run_mode: 'trigger_only',
          trigger_classes: ['person'],
          config: { api_key_env: 'OPENAI_API_KEY', model: 'gpt-4o' },
        },
      },
    }
    const invalid = {
      detection: {
        filter: {
          backend: 'yolo',
          config: { classes: ['person'], min_confidence: 0.5 },
        },
        vlm: {
          backend: 'openai',
          run_mode: 'invalid',
          trigger_classes: ['person'],
          config: {},
        },
      },
    }

    // When: Parsing detection step drafts
    const parsedValid = parseDetectionStepDraft(valid)
    const parsedInvalid = parseDetectionStepDraft(invalid)

    // Then: Valid payload is accepted and malformed run_mode is rejected
    expect(parsedValid).toMatchObject({
      filter: { backend: 'yolo' },
      vlm: { backend: 'openai', run_mode: 'trigger_only' },
    })
    expect(parsedInvalid).toBeNull()
  })
})
