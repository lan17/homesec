import type { ConfigPatch } from '../../../api/client'
import {
  diffConfig,
  expectConfigObject,
  expectConfigString,
  expectConfigStringList,
  isEnvReference,
} from '../configEditing'
import type { FilterFormState, VlmFormState, VlmRunMode } from './types'

// The built-in YOLO backend resolves class IDs through HUMAN_ANIMAL_CLASSES.
export const SUPPORTED_YOLO_CLASSES = [
  'person', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe',
] as const

export interface DetectionSettingsState {
  filter: { backend: string; form: FilterFormState | null }
  vlm: { backend: string; form: VlmFormState | null }
}

function runMode(value: unknown): VlmRunMode {
  if (value === 'trigger_only' || value === 'always' || value === 'never') {
    return value
  }
  throw new Error('vlm.run_mode is invalid.')
}

export function readDetectionSettings(config: Record<string, unknown>): DetectionSettingsState {
  const filter = expectConfigObject(config.filter, 'filter')
  const filterBackend = expectConfigString(filter.backend, 'filter.backend')
  const filterConfig = expectConfigObject(filter.config, 'filter.config')
  let filterForm: FilterFormState | null = null
  if (filterBackend === 'yolo') {
    const confidence = filterConfig.min_confidence === undefined ? 0.5 : filterConfig.min_confidence
    if (typeof confidence !== 'number' || !Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
      throw new Error('filter.config.min_confidence must be between 0 and 1.')
    }
    filterForm = {
      backend: 'yolo',
      config: {
        classes: expectConfigStringList(filterConfig.classes === undefined ? ['person'] : filterConfig.classes, 'filter.config.classes'),
        min_confidence: confidence,
      },
    }
  }
  const vlm = expectConfigObject(config.vlm, 'vlm')
  const vlmBackend = expectConfigString(vlm.backend, 'vlm.backend')
  const vlmConfig = expectConfigObject(vlm.config, 'vlm.config')
  let vlmForm: VlmFormState | null = null
  if (vlmBackend === 'openai') {
    vlmForm = {
      backend: 'openai',
      run_mode: runMode(vlm.run_mode === undefined ? 'trigger_only' : vlm.run_mode),
      trigger_classes: expectConfigStringList(vlm.trigger_classes === undefined ? ['person'] : vlm.trigger_classes, 'vlm.trigger_classes'),
      config: {
        api_key_env: expectConfigString(vlmConfig.api_key_env === undefined ? '' : vlmConfig.api_key_env, 'vlm.config.api_key_env'),
        model: expectConfigString(vlmConfig.model === undefined ? '' : vlmConfig.model, 'vlm.config.model'),
        base_url: expectConfigString(vlmConfig.base_url === undefined ? 'https://api.openai.com/v1' : vlmConfig.base_url, 'vlm.config.base_url'),
      },
    }
  }
  return { filter: { backend: filterBackend, form: filterForm }, vlm: { backend: vlmBackend, form: vlmForm } }
}

export function withVlmEnabled(value: VlmFormState, enabled: boolean): VlmFormState {
  return { ...value, run_mode: enabled ? (value.run_mode === 'never' ? 'trigger_only' : value.run_mode) : 'never' }
}

export function buildDetectionPatch(
  original: DetectionSettingsState,
  edited: DetectionSettingsState,
): Pick<ConfigPatch, 'filter' | 'vlm'> {
  if (edited.filter.backend !== original.filter.backend || edited.vlm.backend !== original.vlm.backend) {
    throw new Error('Detection backends cannot be changed here.')
  }
  const patch: Pick<ConfigPatch, 'filter' | 'vlm'> = {}
  const originalFilter = original.filter.form
  const editedFilter = edited.filter.form
  if (originalFilter && editedFilter) {
    const configPatch = diffConfig(originalFilter.config, editedFilter.config)
    if (configPatch.classes !== undefined) {
      if (editedFilter.config.classes.length === 0) {
        throw new Error('Choose at least one detection class.')
      }
      const unsupported = editedFilter.config.classes.filter((name) => !SUPPORTED_YOLO_CLASSES.some((supported) => supported === name))
      if (unsupported.length > 0) {
        throw new Error(`The YOLO backend does not support: ${unsupported.join(', ')}.`)
      }
    }
    if (configPatch.min_confidence !== undefined) {
      const confidence = editedFilter.config.min_confidence
      if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
        throw new Error('Confidence must be between 0 and 1.')
      }
    }
    if (Object.keys(configPatch).length > 0) {
      patch.filter = { config: configPatch }
    }
  }
  const originalVlm = original.vlm.form
  const editedVlm = edited.vlm.form
  if (originalVlm && editedVlm) {
    const vlmPatch = diffConfig(
      { config: originalVlm.config, run_mode: originalVlm.run_mode, trigger_classes: originalVlm.trigger_classes },
      { config: editedVlm.config, run_mode: editedVlm.run_mode, trigger_classes: editedVlm.trigger_classes },
    )
    if (Object.keys(vlmPatch).length > 0) {
      if (editedVlm.run_mode !== 'never') {
        if (!editedVlm.config.model.trim()) {
          throw new Error('An AI model is required.')
        }
        if (!isEnvReference(editedVlm.config.api_key_env)) {
          throw new Error('AI credentials must reference an environment variable name.')
        }
        if (!editedVlm.config.base_url.trim()) {
          throw new Error('An AI API endpoint is required.')
        }
      }
      patch.vlm = {
        ...(vlmPatch.config !== undefined ? { config: expectConfigObject(vlmPatch.config, 'vlm patch.config') } : {}),
        ...(vlmPatch.run_mode !== undefined ? { run_mode: runMode(vlmPatch.run_mode) } : {}),
        ...(vlmPatch.trigger_classes !== undefined ? { trigger_classes: expectConfigStringList(vlmPatch.trigger_classes, 'vlm patch.trigger_classes') } : {}),
      }
    }
  }
  return patch
}
