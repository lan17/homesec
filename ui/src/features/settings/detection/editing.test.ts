import { describe, expect, it } from 'vitest'

import { buildDetectionPatch, readDetectionSettings, withVlmEnabled } from './editing'

function savedConfig() {
  return {
    filter: { backend: 'yolo', config: { classes: ['person', 'horse'], min_confidence: 0.05, model_path: 'custom.pt', sample_fps: 7, min_hits: 3, min_box_h_ratio: 0.3, max_workers: 2 } },
    vlm: { backend: 'openai', run_mode: 'always', trigger_classes: ['horse'], preprocessing: { max_frames: 6, quality: 73, max_size: 700 }, config: { api_key_env: 'MY_KEY', model: 'configured-model', base_url: 'https://ai.test/v1', token_param: 'max_tokens', max_tokens: 321, temperature: 0.7, request_timeout: 43 } },
  }
}

describe('existing detection configuration editing', () => {
  it('preserves server-supported confidence below the setup slider minimum', () => {
    // Given: A valid saved confidence of 0.05
    const original = readDetectionSettings(savedConfig())
    // When: The value is read and confidence is explicitly set to zero
    const edited = { ...original, filter: { ...original.filter, form: { ...original.filter.form!, config: { ...original.filter.form!.config, min_confidence: 0 } } } }
    // Then: No clamping occurs and only the changed threshold is patched
    expect(original.filter.form?.config.min_confidence).toBe(0.05)
    expect(buildDetectionPatch(original, edited)).toEqual({ filter: { config: { min_confidence: 0 } } })
  })

  it('edits the model without replacing run mode, preprocessing, or advanced controls', () => {
    // Given: AI analysis uses always mode and advanced token/preprocessing settings
    const config = savedConfig()
    const original = readDetectionSettings(config)
    // When: Only the model changes
    const edited = { ...original, vlm: { ...original.vlm, form: { ...original.vlm.form!, config: { ...original.vlm.form!.config, model: 'new-model' } } } }
    // Then: The patch contains no mode, trigger, preprocessing, or token-field replacement
    expect(buildDetectionPatch(original, edited)).toEqual({ vlm: { config: { model: 'new-model' } } })
    expect(config.vlm.preprocessing).toEqual({ max_frames: 6, quality: 73, max_size: 700 })
    expect(config.vlm.config.max_tokens).toBe(321)
  })

  it('explicitly disables AI analysis while preserving the saved configuration', () => {
    // Given: AI analysis is enabled in always mode
    const original = readDetectionSettings(savedConfig())
    // When: The operator disables analysis
    const edited = { ...original, vlm: { ...original.vlm, form: withVlmEnabled(original.vlm.form!, false) } }
    // Then: The patch changes only run_mode to never
    expect(buildDetectionPatch(original, edited)).toEqual({ vlm: { run_mode: 'never' } })
    expect(edited.vlm.form?.config).toEqual(original.vlm.form?.config)
    expect(edited.vlm.form?.trigger_classes).toEqual(['horse'])
  })

  it('rejects unsupported class changes, empty classes, and out-of-range confidence', () => {
    // Given: A valid built-in YOLO configuration
    const original = readDetectionSettings(savedConfig())
    const changed = (config: { classes?: string[]; min_confidence?: number }) => ({ ...original, filter: { ...original.filter, form: { ...original.filter.form!, config: { ...original.filter.form!.config, ...config } } } })
    // When/Then: Invalid guided edits are rejected before submission
    expect(() => buildDetectionPatch(original, changed({ classes: ['car'] }))).toThrow('does not support: car')
    expect(() => buildDetectionPatch(original, changed({ classes: [] }))).toThrow('at least one')
    expect(() => buildDetectionPatch(original, changed({ min_confidence: 1.01 }))).toThrow('between 0 and 1')
    expect(buildDetectionPatch(original, changed({ classes: ['bird', 'person'] }))).toEqual({ filter: { config: { classes: ['bird', 'person'] } } })
  })

  it('preserves existing unsupported classes when editing an unrelated field', () => {
    // Given: An existing configuration contains a class outside the built-in mapping
    const config = savedConfig()
    config.filter.config.classes = ['person', 'car']
    const original = readDetectionSettings(config)
    // When: Only confidence changes
    const edited = { ...original, filter: { ...original.filter, form: { ...original.filter.form!, config: { ...original.filter.form!.config, min_confidence: 0.8 } } } }
    // Then: The class list is not silently rewritten
    expect(buildDetectionPatch(original, edited)).toEqual({ filter: { config: { min_confidence: 0.8 } } })
  })

  it('keeps custom detection backends read-only and validates untrusted field shapes', () => {
    // Given: Existing third-party detection plugins
    const config = { filter: { backend: 'custom-filter', config: { preserve: true } }, vlm: { backend: 'custom-ai', config: {} } }
    // When: The existing-install editor reads them
    const original = readDetectionSettings(config)
    // Then: No guided forms or replacement patches are created, and malformed supported payloads fail
    expect(original.filter.form).toBeNull()
    expect(original.vlm.form).toBeNull()
    expect(buildDetectionPatch(original, original)).toEqual({})
    expect(() => readDetectionSettings({ ...savedConfig(), filter: { backend: 'yolo', config: { classes: [123] } } })).toThrow('list of strings')
    expect(() => readDetectionSettings({ ...savedConfig(), filter: { backend: 'yolo', config: { min_confidence: null } } })).toThrow('between 0 and 1')
  })
})
