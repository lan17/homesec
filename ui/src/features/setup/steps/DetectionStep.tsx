import { useState } from 'react'

import { Button } from '../../../components/ui/Button'
import { FilterConfigForm } from '../../settings/detection/FilterConfigForm'
import {
  buildDetectionStepData,
  detectionFormStateFromStepData,
  type DetectionStepData,
  validateDetectionFormState,
} from '../../settings/detection/types'
import { VlmConfigForm } from '../../settings/detection/VlmConfigForm'

interface DetectionStepProps {
  initialData: DetectionStepData | null
  onComplete: () => void
  onUpdateData: (data: DetectionStepData) => void
  onSkip: () => void
}

export function DetectionStep({
  initialData,
  onComplete,
  onUpdateData,
  onSkip,
}: DetectionStepProps) {
  const [value, setValue] = useState(() => detectionFormStateFromStepData(initialData))
  const [validationError, setValidationError] = useState<string | null>(null)

  function handleSaveAndContinue(): void {
    const maybeError = validateDetectionFormState(value)
    if (maybeError) {
      setValidationError(maybeError)
      return
    }
    setValidationError(null)
    onUpdateData(buildDetectionStepData(value))
    onComplete()
  }

  return (
    <section className="wizard-step-card">
      <FilterConfigForm
        value={value.filter}
        onChange={(nextFilter) => {
          setValue((current) => {
            const nextClasses = nextFilter.config.classes
            return {
              ...current,
              filter: nextFilter,
              vlm: {
                ...current.vlm,
                trigger_classes: current.vlm.trigger_classes.filter((item) =>
                  nextClasses.includes(item),
                ),
              },
            }
          })
          setValidationError(null)
        }}
      />

      <VlmConfigForm
        enabled={value.vlm_enabled}
        value={value.vlm}
        filterClasses={value.filter.config.classes}
        onToggle={(enabled) => {
          setValue((current) => {
            if (!enabled) {
              return {
                ...current,
                vlm_enabled: false,
              }
            }
            return {
              ...current,
              vlm_enabled: true,
              vlm: {
                ...current.vlm,
                run_mode:
                  current.vlm.run_mode === 'never' ? 'trigger_only' : current.vlm.run_mode,
              },
            }
          })
          setValidationError(null)
        }}
        onChange={(nextVlm) => {
          setValue((current) => ({
            ...current,
            vlm: nextVlm,
          }))
          setValidationError(null)
        }}
      />

      {validationError ? <p className="error-text">{validationError}</p> : null}

      <div className="inline-form__actions">
        <Button variant="ghost" onClick={onSkip}>
          Skip detection step
        </Button>
        <Button onClick={handleSaveAndContinue}>Save and continue</Button>
      </div>
    </section>
  )
}
