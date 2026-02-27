import { describe, expect, it } from 'vitest'

import {
  buildDetectionStepData,
  defaultDetectionFormState,
  detectionFormStateFromStepData,
  validateDetectionFormState,
  type DetectionFormState,
} from './types'

describe('detection types helpers', () => {
  it('hydrates defaults when step data is missing', () => {
    // Given: No persisted wizard detection data
    const stepData = null

    // When: Hydrating form state from step data
    const state = detectionFormStateFromStepData(stepData)

    // Then: Defaults are applied for filter and disabled VLM
    expect(state).toEqual(defaultDetectionFormState())
    expect(state.vlm_enabled).toBe(false)
  })

  it('normalizes malformed step payloads and coerces enabled VLM run_mode', () => {
    // Given: Detection payload with malformed classes/confidence and VLM run_mode=never
    const stepData = {
      filter: {
        backend: 'yolo',
        config: {
          classes: ['', 'Person', 'person'],
          min_confidence: 2.7,
        },
      },
      vlm: {
        backend: 'openai',
        run_mode: 'never' as const,
        trigger_classes: [' Person ', '', 'person'],
        config: {
          api_key_env: '',
          model: '',
          base_url: '',
        },
      },
    }

    // When: Hydrating detection form state
    const state = detectionFormStateFromStepData(stepData)

    // Then: Values are normalized and VLM run_mode coerces to trigger_only when enabled
    expect(state.filter.config.classes).toEqual(['person'])
    expect(state.filter.config.min_confidence).toBe(1.0)
    expect(state.vlm_enabled).toBe(true)
    expect(state.vlm.run_mode).toBe('trigger_only')
    expect(state.vlm.config.api_key_env).toBe('OPENAI_API_KEY')
    expect(state.vlm.config.model).toBe('gpt-4o')
    expect(state.vlm.config.base_url).toBe('https://api.openai.com/v1')
  })

  it('returns validation errors for invalid filter and VLM state', () => {
    // Given: Invalid detection form state with empty classes and missing VLM fields
    const state: DetectionFormState = {
      filter: {
        backend: 'yolo',
        config: {
          classes: [],
          min_confidence: 0.5,
        },
      },
      vlm_enabled: true,
      vlm: {
        backend: 'openai',
        run_mode: 'trigger_only',
        trigger_classes: [],
        config: {
          api_key_env: '',
          model: '',
          base_url: '',
        },
      },
    }

    // When: Validating detection form state
    const error = validateDetectionFormState(state)

    // Then: Validation stops at first failing rule
    expect(error).toBe('At least one detection class is required.')
  })

  it('builds finalized step payload with normalized classes and trimmed VLM config', () => {
    // Given: Detection form with mixed-case classes and whitespace in VLM config
    const state: DetectionFormState = {
      filter: {
        backend: 'yolo',
        config: {
          classes: [' person ', 'CAR', 'person'],
          min_confidence: 0.55,
        },
      },
      vlm_enabled: true,
      vlm: {
        backend: 'openai',
        run_mode: 'always',
        trigger_classes: [' person ', '', 'car'],
        config: {
          api_key_env: ' OPENAI_KEY ',
          model: ' gpt-4.1-mini ',
          base_url: ' https://api.openai.com/v1 ',
        },
      },
    }

    // When: Building detection step data for finalize payload
    const payload = buildDetectionStepData(state)

    // Then: Output is normalized and trimmed for backend contract
    expect(payload).toEqual({
      filter: {
        backend: 'yolo',
        config: {
          classes: ['person', 'car'],
          min_confidence: 0.55,
        },
      },
      vlm: {
        backend: 'openai',
        run_mode: 'always',
        trigger_classes: ['person', 'car'],
        config: {
          api_key_env: 'OPENAI_KEY',
          model: 'gpt-4.1-mini',
          base_url: 'https://api.openai.com/v1',
        },
      },
    })
  })

  it('emits null VLM payload when VLM is disabled', () => {
    // Given: Detection form where VLM remains disabled
    const state = defaultDetectionFormState()

    // When: Building detection step data
    const payload = buildDetectionStepData(state)

    // Then: Finalized payload keeps VLM as null
    expect(payload.vlm).toBeNull()
  })
})
