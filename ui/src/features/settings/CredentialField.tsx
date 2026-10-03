import { useState } from 'react'

import type { ConfigResponse } from '../../api/generated/types'
import { Button } from '../../components/ui/Button'

export type CredentialDraft = Record<string, string | null>

export interface CredentialFields {
  prefix: string
  statuses: ConfigResponse['credentials']
  editable: boolean
  draft: CredentialDraft
  onChange: (path: string, value: string | null | undefined) => void
}

interface CredentialFieldProps {
  id: string
  label: string
  envKey: string
  envValue: string
  onEnvChange: (value: string) => void
  credentials: CredentialFields
}

export function CredentialField({ id, label, envKey, envValue, onEnvChange, credentials }: CredentialFieldProps) {
  const [replacing, setReplacing] = useState(false)
  const path = `${credentials.prefix}.${envKey}`
  const status = credentials.statuses[path]
  const draft = credentials.draft[path]
  const configured = status?.configured ?? false
  const editing = replacing || !configured || typeof draft === 'string'

  return (
    <div className="inline-form">
      <p className="field-label">{label}</p>
      <p className="subtle">{draft === null ? 'Will stop being used after Save and Apply.'
        : typeof draft === 'string' ? 'Replacement is ready to save.'
          : configured ? `Configured${status?.source === 'environment' ? ' from the environment' : ''}. Saved values are hidden.` : 'Not configured.'}</p>
      {credentials.editable ? (
        <>
          {editing && draft !== null ? (
            <label className="field-label" htmlFor={id}>
              {configured ? `Replace ${label}` : label}
              <input id={id} className="input" type="password" autoComplete="new-password"
                value={typeof draft === 'string' ? draft : ''}
                placeholder={configured ? 'Enter a replacement' : 'Enter a value'}
                onChange={(event) => { credentials.onChange(path, event.target.value || undefined) }} />
            </label>
          ) : null}
          <div className="inline-form__actions">
            {configured && !editing && draft !== null ? (
              <Button variant="ghost" onClick={() => { setReplacing(true) }}>Replace {label}</Button>
            ) : null}
            {(configured || typeof draft === 'string') && draft !== null ? (
              <Button variant="ghost" onClick={() => { credentials.onChange(path, null); setReplacing(false) }}>Clear {label}</Button>
            ) : null}
            {draft !== undefined || replacing ? (
              <Button variant="ghost" onClick={() => { credentials.onChange(path, undefined); setReplacing(false) }}>Keep saved {label}</Button>
            ) : null}
          </div>
          <p className="subtle">Leave blank to keep the saved credential. Save stores credentials on this HomeSec host. Apply restarts HomeSec to use them.</p>
        </>
      ) : <p className="subtle">Enable API authentication on the HomeSec host to enter credentials here.</p>}
      <details>
        <summary>Advanced: environment variable</summary>
        <label className="field-label" htmlFor={`${id}-env`}>
          {label} env var
          <input id={`${id}-env`} className="input" type="text" value={envValue}
            onChange={(event) => {
              credentials.onChange(path, undefined)
              setReplacing(false)
              onEnvChange(event.target.value)
            }} />
        </label>
        <p className="subtle">Changing this reference uses the named environment variable instead of a saved credential.</p>
      </details>
    </div>
  )
}
