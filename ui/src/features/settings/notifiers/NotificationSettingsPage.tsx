import { useMemo, useState } from 'react'

import type { ConfigPatch } from '../../../api/client'
import type { TestConnectionResponse } from '../../../api/generated/types'
import { Button } from '../../../components/ui/Button'
import { Card } from '../../../components/ui/Card'
import { TestConnectionButton } from '../../shared/TestConnectionButton'
import { describeUnknownError } from '../../shared/errorPresentation'
import { ConfigApplyPanel } from '../ConfigApplyPanel'
import type { CredentialDraft, CredentialFields } from '../CredentialField'
import { credentialsBlockProbe, updateCredentialDraft } from '../credentialEditing'
import { AlertPolicyForm } from '../alerts/AlertPolicyForm'
import { useConfigSettings } from '../useConfigSettings'
import { NOTIFIER_BACKENDS } from './backends'
import {
  buildNotificationPatch,
  isEditableNotifier,
  readNotificationSettings,
  type ConfiguredNotifier,
  type NotificationSettingsState,
} from './editing'
import type { RiskLevel } from './types'

interface NotificationDraft {
  version: string
  original: NotificationSettingsState
  edited: NotificationSettingsState
}

export function NotificationSettingsPage() {
  const settings = useConfigSettings()
  const [draft, setDraft] = useState<NotificationDraft | null>(null)
  const [credentialDraft, setCredentialDraft] = useState<CredentialDraft>({})
  const [formMessage, setFormMessage] = useState<string | null>(null)
  const [formError, setFormError] = useState<string | null>(null)
  const [connectionResults, setConnectionResults] = useState<Record<number, TestConnectionResponse | null>>({})
  const baseline = useMemo(() => {
    const snapshot = settings.configQuery.data
    if (!snapshot) {
      return { value: null, error: null }
    }
    try {
      return { value: readNotificationSettings(snapshot.config), error: null }
    } catch (error) {
      return { value: null, error: describeUnknownError(error) }
    }
  }, [settings.configQuery.data])
  const value = draft?.edited ?? baseline.value
  const busy = settings.savePending || settings.applyPending

  function updateValue(next: NotificationSettingsState): void {
    const snapshot = settings.configQuery.data
    const original = baseline.value
    if (!original || !snapshot) {
      return
    }
    setDraft((previous) => ({
      version: previous?.version ?? snapshot.saved_config_version,
      original: previous?.original ?? original,
      edited: next,
    }))
    setFormMessage(null)
    setFormError(null)
  }

  function updateNotifier(index: number, change: Partial<ConfiguredNotifier>): void {
    if (!value) {
      return
    }
    updateValue({
      ...value,
      notifiers: value.notifiers.map((entry) => entry.index === index ? { ...entry, ...change } : entry),
    })
    setConnectionResults((previous) => ({ ...previous, [index]: null }))
  }

  function updateRisk(minRiskLevel: RiskLevel): void {
    if (value) {
      updateValue({ ...value, alertPolicy: { ...value.alertPolicy, minRiskLevel } })
    }
  }

  async function save(): Promise<void> {
    if (!draft) {
      setFormMessage('No notification changes to save.')
      return
    }
    setFormError(null)
    let patch: Pick<ConfigPatch, 'notifiers' | 'alert_policy' | 'credentials'>
    try {
      patch = buildNotificationPatch(draft.original, draft.edited)
      if (Object.keys(credentialDraft).length > 0) { patch.credentials = credentialDraft }
    } catch (error) {
      setFormError(describeUnknownError(error))
      return
    }
    if (Object.keys(patch).length === 0) {
      setFormMessage('No notification changes to save.')
      return
    }
    try {
      await settings.saveConfig({ ...patch, expected_config_version: draft.version })
      setDraft(null)
      setCredentialDraft({})
      setConnectionResults({})
      setFormMessage('Notification settings saved.')
    } catch {
      // The shared activation panel displays save errors; retain this draft.
      return
    }
  }

  return (
    <section className="page fade-in-up">
      <header className="page__header">
        <div>
          <h1 className="page__title">Notifications</h1>
          <p className="page__lead">Edit configured alert destinations and the default risk threshold.</p>
        </div>
      </header>
      <ConfigApplyPanel settings={settings} />
      {baseline.error ? <p className="error-text" role="alert">{baseline.error}</p> : null}
      {!value && !baseline.error ? <p className="subtle">Loading notification settings…</p> : null}
      {value ? (
        <>
          {value.notifiers.length === 0 ? (
            <Card title="Alert destinations"><p className="muted">No notifier instances are configured. Add destinations in the YAML configuration.</p></Card>
          ) : null}
          {value.notifiers.map((entry) => {
            const backend = isEditableNotifier(entry.backend) ? NOTIFIER_BACKENDS[entry.backend] : null
            const BackendForm = backend?.component
            const snapshot = settings.configQuery.data
            const credentials: CredentialFields = {
              prefix: `notifiers.${entry.index}.config`, statuses: snapshot?.credentials ?? {}, editable: snapshot?.credentials_editable ?? false,
              draft: credentialDraft,
              onChange: (path, next) => {
                updateValue(value)
                setCredentialDraft((previous) => updateCredentialDraft(previous, path, next))
                setConnectionResults((previous) => ({ ...previous, [entry.index]: null }))
              },
            }
            return (
              <Card key={entry.index} title={`${backend?.label ?? entry.backend} · notifier ${entry.index + 1}`}>
                {BackendForm ? (
                  <fieldset className="inline-form" disabled={busy}>
                    <label className="field-label form-checkbox-field" htmlFor={`notifier-${entry.index}-enabled`}>
                      <input id={`notifier-${entry.index}-enabled`} type="checkbox" checked={entry.enabled}
                        onChange={(event) => { updateNotifier(entry.index, { enabled: event.target.checked }) }} />
                      Enable notifier {entry.index + 1}
                    </label>
                    {entry.enabled ? (
                      <>
                        <BackendForm config={entry.config} idPrefix={`notifier-${entry.index}`} credentials={credentials}
                          onChange={(config) => { updateNotifier(entry.index, { config }) }} />
                        {credentialsBlockProbe(credentials, snapshot?.apply_required ?? 'none')
                          ? <p className="subtle">Save and apply credential changes before checking this connection.</p>
                          : <TestConnectionButton request={{ type: 'notifier', backend: entry.backend, config: entry.config }}
                          result={connectionResults[entry.index] ?? null}
                          onResult={(result) => { setConnectionResults((previous) => ({ ...previous, [entry.index]: result })) }}
                          idleLabel="Check connection" retryLabel="Check connection again" pendingLabel="Checking connection…"
                          description="Check connectivity using these settings. This does not send a sample alert." />}
                      </>
                    ) : null}
                  </fieldset>
                ) : (
                  <p className="muted">This integration is read-only here. It is {entry.enabled ? 'enabled' : 'disabled'}; edit its settings in YAML.</p>
                )}
              </Card>
            )
          })}
          <Card title="Alert sensitivity">
            {value.alertPolicy.minRiskLevel !== null ? (
              <fieldset className="inline-form" disabled={busy}>
                <AlertPolicyForm value={{ minRiskLevel: value.alertPolicy.minRiskLevel }} onChange={(next) => { updateRisk(next.minRiskLevel) }} />
                {!value.alertPolicy.enabled ? <p className="subtle">The alert policy is disabled in saved settings.</p> : null}
                <p className="subtle">Per-camera overrides and other alert rules remain configured in YAML.</p>
              </fieldset>
            ) : <p className="muted">The {value.alertPolicy.backend} alert policy is read-only here. Edit it in YAML.</p>}
          </Card>
          <div className="inline-form__actions">
            <Button disabled={busy} onClick={() => { void save() }}>{settings.savePending ? 'Saving…' : 'Save notification settings'}</Button>
            {draft ? (
              <Button variant="ghost" disabled={busy} onClick={() => {
                setDraft(null)
                setCredentialDraft({})
                setConnectionResults({})
                setFormError(null)
                setFormMessage(null)
              }}>Discard unsaved changes</Button>
            ) : null}
          </div>
          {formMessage ? <p role="status">{formMessage}</p> : null}
          {formError ? <p className="error-text" role="alert">{formError}</p> : null}
        </>
      ) : null}
    </section>
  )
}
