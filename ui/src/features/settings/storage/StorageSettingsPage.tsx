import { useState } from 'react'
import { Link } from 'react-router-dom'

import type { ConfigPatch, TestConnectionResponse } from '../../../api/generated/types'
import { Button } from '../../../components/ui/Button'
import { Card } from '../../../components/ui/Card'
import { TestConnectionButton } from '../../shared/TestConnectionButton'
import { describeUnknownError } from '../../shared/errorPresentation'
import { ConfigApplyPanel } from '../ConfigApplyPanel'
import type { CredentialDraft, CredentialFields } from '../CredentialField'
import { credentialsBlockProbe, updateCredentialDraft } from '../credentialEditing'
import { useConfigSettings } from '../useConfigSettings'
import { buildStorageSettingsPatch, storageSettingsDraft, type StorageSettingsDraft } from './editing'
import { StorageConfigForm } from './StorageConfigForm'
import { buildStorageTestRequest } from './types'

const PATH_FIELDS = [
  ['clips_dir', 'Clip working directory'],
  ['backups_dir', 'Backup directory'],
  ['artifacts_dir', 'Analysis artifact directory'],
] as const

export function StorageSettingsPage() {
  const settings = useConfigSettings()
  const [draft, setDraft] = useState<StorageSettingsDraft | null>(null)
  const [credentialDraft, setCredentialDraft] = useState<CredentialDraft>({})
  const [validationError, setValidationError] = useState<string | null>(null)
  const [testResult, setTestResult] = useState<TestConnectionResponse | null>(null)
  const config = settings.configQuery.data
  let current = draft
  let hydrationError: string | null = null
  if (!current && config) {
    try {
      current = storageSettingsDraft(config)
    } catch (error) {
      hydrationError = describeUnknownError(error)
    }
  }
  const busy = settings.savePending || settings.applyPending
  const changed = current !== null
    && (JSON.stringify(current.value) !== JSON.stringify(current.original)
      || JSON.stringify(current.paths) !== JSON.stringify(current.originalPaths)
      || Object.keys(credentialDraft).length > 0)
  const hasRedactedConfig = current !== null
    && JSON.stringify(current.value.config).includes('***redacted***')
  const credentials: CredentialFields = {
    prefix: 'storage.config', statuses: config?.credentials ?? {}, editable: config?.credentials_editable ?? false,
    draft: credentialDraft,
    onChange: (path, value) => {
      if (current) { setDraft(current) }
      setCredentialDraft((previous) => updateCredentialDraft(previous, path, value))
      setTestResult(null)
      setValidationError(null)
    },
  }
  const credentialProbeBlocked = credentialsBlockProbe(credentials, config?.apply_required ?? 'none')

  async function save(): Promise<void> {
    if (!current) {
      return
    }
    setValidationError(null)
    let patch: ConfigPatch
    try {
      patch = buildStorageSettingsPatch(current)
      if (patch.storage && Object.keys(patch.storage).length === 0) { delete patch.storage }
      if (Object.keys(credentialDraft).length > 0) { patch.credentials = credentialDraft }
    } catch (error) {
      setValidationError(describeUnknownError(error))
      return
    }
    try {
      await settings.saveConfig(patch)
      setDraft(null)
      setCredentialDraft({})
      setTestResult(null)
    } catch {
      // The shared activation panel displays save errors; retain this draft.
      return
    }
  }

  return (
    <section className="page fade-in-up">
      <header className="page__header">
        <div><h1 className="page__title">Storage settings</h1>
          <p className="page__lead">Edit the configured storage backend. Save changes, then apply them.</p>
        </div>
        <Link to="/settings" className="button button--ghost">Back to Settings</Link>
      </header>
      {settings.configQuery.isPending ? <p>Loading saved settings…</p> : null}
      {hydrationError ? <p role="alert" className="error-text">{hydrationError}</p> : null}
      {config && !current && !hydrationError ? (
        <Card title="Storage configuration">
          <p>This storage backend is not supported by the editor. Its saved configuration is preserved. Edit it in YAML.</p>
        </Card>
      ) : null}
      {current ? <Card title="Storage configuration">
        <fieldset disabled={busy} className="inline-form">
          <StorageConfigForm value={current.value} allowBackendChange={false} credentials={credentials}
            onChange={(value) => {
              if (current) { setDraft({ ...current, value }) }
              setTestResult(null)
              setValidationError(null)
            }} />
          <details>
            <summary>Local working paths</summary>
            {PATH_FIELDS.map(([key, label]) => (
              <label key={key} htmlFor={`storage-${key}`} className="field-label">
                {label}
                <input id={`storage-${key}`} className="input" type="text"
                  value={typeof current.paths[key] === 'string' ? current.paths[key] : ''}
                  onChange={(event) => {
                    if (current) { setDraft({ ...current, paths: { ...current.paths, [key]: event.target.value } }) }
                    setValidationError(null)
                  }} />
              </label>
            ))}
          </details>
          {credentialProbeBlocked ? <p className="subtle">Save and apply credential changes before checking the storage connection.</p> : hasRedactedConfig ? (
            <p className="subtle">Stored credentials are preserved when saving. Connectivity checks are unavailable while the configuration contains redacted values.</p>
          ) : <TestConnectionButton request={buildStorageTestRequest(current.value)}
            result={testResult} onResult={setTestResult} idleLabel="Check storage connection"
            description="Checks connectivity and path access; this does not upload a sample clip." />}
          <div className="inline-form__actions">
            <Button disabled={!changed || busy} onClick={() => { void save() }}>
              {settings.savePending ? 'Saving…' : 'Save storage settings'}
            </Button>
            <Button variant="ghost" disabled={!draft || busy} onClick={() => {
              setDraft(null); setCredentialDraft({}); setValidationError(null); setTestResult(null)
            }}>Discard draft</Button>
          </div>
        </fieldset>
        {validationError ? <p className="error-text" role="alert">{validationError}</p> : null}
      </Card> : null}
      <ConfigApplyPanel settings={settings} />
    </section>
  )
}
