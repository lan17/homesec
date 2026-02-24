import { useState, type FormEvent } from 'react'

import { Button } from './Button'
import { toApiKeyGateActionErrorMessage } from './apiKeyGateErrors'

interface ApiKeyGateProps {
  busy: boolean
  onSubmit: (apiKey: string) => Promise<void> | void
  onClear: () => Promise<void> | void
}

export function ApiKeyGate({ busy, onSubmit, onClear }: ApiKeyGateProps) {
  const [value, setValue] = useState('')
  const [validationError, setValidationError] = useState<string | null>(null)
  const [submitError, setSubmitError] = useState<string | null>(null)

  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault()
    const apiKey = value.trim()
    if (!apiKey) {
      setValidationError('API key is required.')
      return
    }

    setValidationError(null)
    setSubmitError(null)
    try {
      await onSubmit(apiKey)
      setValue('')
    } catch (error) {
      setSubmitError(toApiKeyGateActionErrorMessage(error))
    }
  }

  async function handleClear(): Promise<void> {
    setValidationError(null)
    setSubmitError(null)
    try {
      await onClear()
    } catch (error) {
      setSubmitError(toApiKeyGateActionErrorMessage(error))
    }
  }

  return (
    <form className="inline-form" onSubmit={handleSubmit}>
      <p className="muted">This endpoint requires API authentication.</p>
      <label className="field-label" htmlFor="api-key-input">
        API key
      </label>
      <input
        id="api-key-input"
        className="input"
        type="password"
        autoComplete="off"
        value={value}
        placeholder="Paste API key"
        onChange={(event) => setValue(event.target.value)}
        disabled={busy}
      />
      {validationError ? <p className="error-text">{validationError}</p> : null}
      {submitError ? <p className="error-text">{submitError}</p> : null}
      <div className="inline-form__actions">
        <Button type="submit" disabled={busy}>
          {busy ? 'Applying...' : 'Apply key'}
        </Button>
        <Button type="button" variant="ghost" disabled={busy} onClick={handleClear}>
          Clear key
        </Button>
      </div>
    </form>
  )
}
