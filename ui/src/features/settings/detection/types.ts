import type { TestConnectionRequest } from '../../../api/generated/types'

export type VlmRunMode = 'trigger_only' | 'always' | 'never'

export interface FilterFormState {
  backend: 'yolo'
  config: {
    classes: string[]
    min_confidence: number
  }
}

export interface VlmFormState {
  backend: 'openai'
  run_mode: VlmRunMode
  trigger_classes: string[]
  config: {
    api_key_env: string
    model: string
    base_url: string
  }
}

export interface DetectionFormState {
  filter: FilterFormState
  vlm_enabled: boolean
  vlm: VlmFormState
}

export interface DetectionStepData {
  filter: {
    backend: string
    config: Record<string, unknown>
  }
  vlm: {
    backend: string
    config: Record<string, unknown>
    run_mode: VlmRunMode
    trigger_classes: string[]
  } | null
}

const DEFAULT_FILTER_CLASSES = ['person', 'car', 'dog', 'cat'] as const
const DEFAULT_FILTER_MIN_CONFIDENCE = 0.5
const DEFAULT_VLM_MODEL = 'gpt-4o'
const DEFAULT_VLM_BASE_URL = 'https://api.openai.com/v1'
const DEFAULT_VLM_API_KEY_ENV = 'OPENAI_API_KEY'

function normalizeClassList(values: readonly string[]): string[] {
  const seen = new Set<string>()
  const normalized: string[] = []

  for (const value of values) {
    const cleaned = value.trim().toLowerCase()
    if (cleaned.length === 0 || seen.has(cleaned)) {
      continue
    }
    seen.add(cleaned)
    normalized.push(cleaned)
  }

  return normalized
}

function normalizeFilterState(value: DetectionStepData['filter'] | undefined): FilterFormState {
  const classesValue = value?.config.classes
  const classes = Array.isArray(classesValue)
    ? normalizeClassList(
        classesValue.filter((item): item is string => typeof item === 'string'),
      )
    : [...DEFAULT_FILTER_CLASSES]

  const minConfidenceValue = value?.config.min_confidence
  const minConfidence =
    typeof minConfidenceValue === 'number' && Number.isFinite(minConfidenceValue)
      ? Math.max(0.1, Math.min(1.0, minConfidenceValue))
      : DEFAULT_FILTER_MIN_CONFIDENCE

  return {
    backend: 'yolo',
    config: {
      classes: classes.length > 0 ? classes : [...DEFAULT_FILTER_CLASSES],
      min_confidence: minConfidence,
    },
  }
}

function normalizeVlmState(
  value: Exclude<DetectionStepData['vlm'], null> | null | undefined,
): VlmFormState {
  const runMode = value?.run_mode
  const normalizedRunMode: VlmRunMode =
    runMode === 'always' || runMode === 'never' || runMode === 'trigger_only'
      ? runMode
      : 'trigger_only'

  const triggerClassesValue = value?.trigger_classes
  const triggerClasses = Array.isArray(triggerClassesValue)
    ? normalizeClassList(
        triggerClassesValue.filter((item): item is string => typeof item === 'string'),
      )
    : ['person']

  const config = value?.config ?? {}
  return {
    backend: 'openai',
    run_mode: normalizedRunMode,
    trigger_classes: triggerClasses.length > 0 ? triggerClasses : ['person'],
    config: {
      api_key_env:
        typeof config.api_key_env === 'string' && config.api_key_env.trim().length > 0
          ? config.api_key_env
          : DEFAULT_VLM_API_KEY_ENV,
      model:
        typeof config.model === 'string' && config.model.trim().length > 0
          ? config.model
          : DEFAULT_VLM_MODEL,
      base_url:
        typeof config.base_url === 'string' && config.base_url.trim().length > 0
          ? config.base_url
          : DEFAULT_VLM_BASE_URL,
    },
  }
}

export function defaultDetectionFormState(): DetectionFormState {
  return {
    filter: normalizeFilterState(undefined),
    vlm_enabled: false,
    vlm: normalizeVlmState(undefined),
  }
}

export function detectionFormStateFromStepData(
  value: DetectionStepData | null | undefined,
): DetectionFormState {
  const base = defaultDetectionFormState()
  if (!value) {
    return base
  }
  const filter = normalizeFilterState(value.filter)
  const vlmEnabled = value.vlm !== null
  const vlm = normalizeVlmState(value.vlm)
  if (vlmEnabled && vlm.run_mode === 'never') {
    return {
      filter,
      vlm_enabled: true,
      vlm: { ...vlm, run_mode: 'trigger_only' },
    }
  }
  return {
    filter,
    vlm_enabled: vlmEnabled,
    vlm,
  }
}

export function validateDetectionFormState(value: DetectionFormState): string | null {
  const classes = normalizeClassList(value.filter.config.classes)
  if (classes.length === 0) {
    return 'At least one detection class is required.'
  }

  const confidence = value.filter.config.min_confidence
  if (!Number.isFinite(confidence) || confidence < 0.1 || confidence > 1.0) {
    return 'Confidence threshold must be between 0.1 and 1.0.'
  }

  if (!value.vlm_enabled) {
    return null
  }

  if (value.vlm.config.api_key_env.trim().length === 0) {
    return 'VLM API key env var is required.'
  }
  if (value.vlm.config.model.trim().length === 0) {
    return 'VLM model is required.'
  }
  if (value.vlm.config.base_url.trim().length === 0) {
    return 'VLM base URL is required.'
  }
  if (value.vlm.run_mode === 'trigger_only' && normalizeClassList(value.vlm.trigger_classes).length === 0) {
    return 'At least one VLM trigger class is required for trigger-only mode.'
  }

  return null
}

export function buildDetectionStepData(value: DetectionFormState): DetectionStepData {
  const filterClasses = normalizeClassList(value.filter.config.classes)
  const filterConfig = {
    classes: filterClasses.length > 0 ? filterClasses : [...DEFAULT_FILTER_CLASSES],
    min_confidence: value.filter.config.min_confidence,
  }

  if (!value.vlm_enabled) {
    return {
      filter: {
        backend: value.filter.backend,
        config: filterConfig,
      },
      vlm: null,
    }
  }

  const triggerClasses = normalizeClassList(value.vlm.trigger_classes)

  return {
    filter: {
      backend: value.filter.backend,
      config: filterConfig,
    },
    vlm: {
      backend: value.vlm.backend,
      run_mode: value.vlm.run_mode,
      trigger_classes: triggerClasses.length > 0 ? triggerClasses : ['person'],
      config: {
        api_key_env: value.vlm.config.api_key_env.trim(),
        model: value.vlm.config.model.trim(),
        base_url: value.vlm.config.base_url.trim(),
      },
    },
  }
}

export function buildAnalyzerTestRequest(value: VlmFormState): TestConnectionRequest {
  return {
    type: 'analyzer',
    backend: value.backend,
    config: {
      api_key_env: value.config.api_key_env.trim(),
      model: value.config.model.trim(),
      base_url: value.config.base_url.trim(),
    },
  }
}
