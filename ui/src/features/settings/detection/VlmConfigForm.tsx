import { useMemo, useState } from 'react'

import type { TestConnectionResponse } from '../../../api/generated/types'
import { TestConnectionButton } from '../../shared/TestConnectionButton'
import { buildAnalyzerTestRequest, type VlmFormState } from './types'

interface VlmConfigFormProps {
  enabled: boolean
  value: VlmFormState
  filterClasses: readonly string[]
  onToggle: (enabled: boolean) => void
  onChange: (value: VlmFormState) => void
}

export function VlmConfigForm({
  enabled,
  value,
  filterClasses,
  onToggle,
  onChange,
}: VlmConfigFormProps) {
  const [testResult, setTestResult] = useState<TestConnectionResponse | null>(null)
  const triggerClassOptions = useMemo(() => {
    const options = new Set<string>()
    for (const className of filterClasses) {
      options.add(className)
    }
    for (const className of value.trigger_classes) {
      options.add(className)
    }
    return [...options]
  }, [filterClasses, value.trigger_classes])

  const testRequest = useMemo(() => buildAnalyzerTestRequest(value), [value])

  function resetTestResult(): void {
    setTestResult(null)
  }

  function updateValue(nextValue: VlmFormState): void {
    onChange(nextValue)
    resetTestResult()
  }

  return (
    <section className="inline-form">
      <label className="field-label form-checkbox-field" htmlFor="detection-vlm-enabled">
        <input
          id="detection-vlm-enabled"
          type="checkbox"
          checked={enabled}
          onChange={(event) => {
            const nextEnabled = event.target.checked
            onToggle(nextEnabled)
            if (nextEnabled && value.run_mode === 'never') {
              updateValue({
                ...value,
                run_mode: 'trigger_only',
              })
            }
          }}
        />
        Enable AI scene analysis (VLM)
      </label>

      {enabled ? (
        <div className="detection-vlm">
          <label className="field-label" htmlFor="detection-vlm-base-url">
            API endpoint
            <input
              id="detection-vlm-base-url"
              className="input"
              type="text"
              value={value.config.base_url}
              onChange={(event) => {
                updateValue({
                  ...value,
                  config: {
                    ...value.config,
                    base_url: event.target.value,
                  },
                })
              }}
            />
          </label>

          <label className="field-label" htmlFor="detection-vlm-model">
            Model
            <input
              id="detection-vlm-model"
              className="input"
              type="text"
              value={value.config.model}
              onChange={(event) => {
                updateValue({
                  ...value,
                  config: {
                    ...value.config,
                    model: event.target.value,
                  },
                })
              }}
            />
          </label>

          <label className="field-label" htmlFor="detection-vlm-api-key-env">
            API key env var
            <input
              id="detection-vlm-api-key-env"
              className="input"
              type="text"
              value={value.config.api_key_env}
              onChange={(event) => {
                updateValue({
                  ...value,
                  config: {
                    ...value.config,
                    api_key_env: event.target.value,
                  },
                })
              }}
            />
          </label>

          <label className="field-label" htmlFor="detection-vlm-run-mode">
            Run mode
            <select
              id="detection-vlm-run-mode"
              className="input"
              value={value.run_mode}
              onChange={(event) => {
                const nextRunMode = event.target.value
                if (
                  nextRunMode === 'trigger_only'
                  || nextRunMode === 'always'
                  || nextRunMode === 'never'
                ) {
                  updateValue({
                    ...value,
                    run_mode: nextRunMode,
                  })
                }
              }}
            >
              <option value="trigger_only">Trigger only</option>
              <option value="always">Always</option>
              <option value="never">Never</option>
            </select>
          </label>

          <fieldset className="detection-vlm__trigger-classes">
            <legend>Trigger classes</legend>
            <div className="detection-vlm__trigger-grid">
              {triggerClassOptions.map((className) => (
                <label key={className} className="form-checkbox-field">
                  <input
                    type="checkbox"
                    checked={value.trigger_classes.includes(className)}
                    onChange={(event) => {
                      const checked = event.target.checked
                      const nextTriggerClasses = checked
                        ? [...value.trigger_classes, className]
                        : value.trigger_classes.filter((item) => item !== className)
                      updateValue({
                        ...value,
                        trigger_classes: nextTriggerClasses,
                      })
                    }}
                  />
                  {className}
                </label>
              ))}
            </div>
          </fieldset>

          <TestConnectionButton
            request={testRequest}
            result={testResult}
            onResult={setTestResult}
            idleLabel="Test analysis"
            retryLabel="Retry analysis test"
            pendingLabel="Testing analysis..."
            description="Validate analyzer connectivity before continuing."
          />
        </div>
      ) : null}
    </section>
  )
}
