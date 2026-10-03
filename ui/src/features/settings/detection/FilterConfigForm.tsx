import { useState } from 'react'

import { Button } from '../../../components/ui/Button'
import type { FilterFormState } from './types'

interface FilterConfigFormProps {
  value: FilterFormState
  onChange: (value: FilterFormState) => void
  supportedClasses?: readonly string[]
  confidenceMin?: number
}

function normalizeClassName(value: string): string {
  return value.trim().toLowerCase()
}

export function FilterConfigForm({ value, onChange, supportedClasses, confidenceMin = 0.1 }: FilterConfigFormProps) {
  const [newClassName, setNewClassName] = useState('')

  function addClass(): void {
    const nextClass = normalizeClassName(newClassName)
    if (nextClass.length === 0 || value.config.classes.includes(nextClass)
      || (supportedClasses && !supportedClasses.includes(nextClass))) {
      return
    }
    onChange({
      ...value,
      config: {
        ...value.config,
        classes: [...value.config.classes, nextClass],
      },
    })
    setNewClassName('')
  }

  function removeClass(name: string): void {
    onChange({
      ...value,
      config: {
        ...value.config,
        classes: value.config.classes.filter((item) => item !== name),
      },
    })
  }

  return (
    <section className="inline-form">
      <h3 className="detection-step__title">Object detection (YOLO)</h3>

      <div className="detection-classes">
        <p className="field-label">Detection classes</p>
        <div className="detection-classes__chips">
          {value.config.classes.map((className) => (
            <button
              key={className}
              type="button"
              className="detection-classes__chip"
              aria-label={`Remove class ${className}`}
              onClick={() => {
                removeClass(className)
              }}
            >
              {className} ×
            </button>
          ))}
        </div>
        <div className="detection-classes__input-row">
          {supportedClasses ? (
            <select className="input" aria-label="Detection class to add" value={newClassName}
              onChange={(event) => { setNewClassName(event.target.value) }}>
              <option value="">Choose a supported class</option>
              {supportedClasses.filter((name) => !value.config.classes.includes(name)).map((name) => <option key={name} value={name}>{name}</option>)}
            </select>
          ) : <input
            id="detection-class-input"
            className="input"
            type="text"
            aria-label="Detection classes"
            value={newClassName}
            placeholder="Add class (for example: person)"
            onChange={(event) => {
              setNewClassName(event.target.value)
            }}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                event.preventDefault()
                addClass()
              }
            }}
          />}
          <Button type="button" variant="ghost" onClick={addClass}>
            Add class
          </Button>
        </div>
      </div>

      <label className="field-label" htmlFor="detection-min-confidence">
        Confidence threshold: {value.config.min_confidence.toFixed(2)}
        <input
          id="detection-min-confidence"
          className="detection-step__range"
          type="range"
          min={confidenceMin}
          max={1.0}
          step={0.01}
          value={value.config.min_confidence}
          onChange={(event) => {
            onChange({
              ...value,
              config: {
                ...value.config,
                min_confidence: Number.parseFloat(event.target.value),
              },
            })
          }}
        />
      </label>
    </section>
  )
}
