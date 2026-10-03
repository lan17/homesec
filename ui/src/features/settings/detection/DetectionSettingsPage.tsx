import { useMemo, useState } from 'react'

import type { ConfigPatch } from '../../../api/client'
import { Button } from '../../../components/ui/Button'
import { Card } from '../../../components/ui/Card'
import { describeUnknownError } from '../../shared/errorPresentation'
import { ConfigApplyPanel } from '../ConfigApplyPanel'
import type { CredentialDraft, CredentialFields } from '../CredentialField'
import { credentialsBlockProbe, updateCredentialDraft } from '../credentialEditing'
import { useConfigSettings } from '../useConfigSettings'
import { FilterConfigForm } from './FilterConfigForm'
import { VlmConfigForm } from './VlmConfigForm'
import {
  SUPPORTED_YOLO_CLASSES,
  buildDetectionPatch,
  readDetectionSettings,
  withVlmEnabled,
  type DetectionSettingsState,
} from './editing'
import type { FilterFormState, VlmFormState } from './types'

interface DetectionDraft {
  version: string
  original: DetectionSettingsState
  edited: DetectionSettingsState
}

export function DetectionSettingsPage() {
  const settings = useConfigSettings()
  const [draft, setDraft] = useState<DetectionDraft | null>(null)
  const [credentialDraft, setCredentialDraft] = useState<CredentialDraft>({})
  const [formMessage, setFormMessage] = useState<string | null>(null)
  const [formError, setFormError] = useState<string | null>(null)
  const baseline = useMemo(() => {
    const snapshot = settings.configQuery.data
    if (!snapshot) {
      return { value: null, error: null }
    }
    try {
      return { value: readDetectionSettings(snapshot.config), error: null }
    } catch (error) {
      return { value: null, error: describeUnknownError(error) }
    }
  }, [settings.configQuery.data])
  const value = draft?.edited ?? baseline.value
  const busy = settings.savePending || settings.applyPending
  const snapshot = settings.configQuery.data
  const credentials: CredentialFields = {
    prefix: 'vlm.config', statuses: snapshot?.credentials ?? {}, editable: snapshot?.credentials_editable ?? false,
    draft: credentialDraft,
    onChange: (path, next) => {
      if (value) { updateValue(value) }
      setCredentialDraft((previous) => updateCredentialDraft(previous, path, next))
    },
  }

  function updateValue(next: DetectionSettingsState): void {
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

  function updateFilter(form: FilterFormState): void {
    if (value) {
      updateValue({ ...value, filter: { ...value.filter, form } })
    }
  }

  function updateVlm(form: VlmFormState): void {
    if (value) {
      updateValue({ ...value, vlm: { ...value.vlm, form } })
    }
  }

  async function save(): Promise<void> {
    if (!draft) {
      setFormMessage('No detection changes to save.')
      return
    }
    setFormError(null)
    let patch: Pick<ConfigPatch, 'filter' | 'vlm' | 'credentials'>
    try {
      patch = buildDetectionPatch(draft.original, draft.edited)
      if (Object.keys(credentialDraft).length > 0) { patch.credentials = credentialDraft }
    } catch (error) {
      setFormError(describeUnknownError(error))
      return
    }
    if (Object.keys(patch).length === 0) {
      setFormMessage('No detection changes to save.')
      return
    }
    try {
      await settings.saveConfig({ ...patch, expected_config_version: draft.version })
      setDraft(null)
      setCredentialDraft({})
      setFormMessage('Detection settings saved.')
    } catch {
      // The shared activation panel displays save errors; retain this draft.
      return
    }
  }

  return (
    <section className="page fade-in-up">
      <header className="page__header">
        <div>
          <h1 className="page__title">Detection</h1>
          <p className="page__lead">Adjust object detection and AI scene analysis for your configured backends.</p>
        </div>
      </header>
      <ConfigApplyPanel settings={settings} />
      {baseline.error ? <p className="error-text" role="alert">{baseline.error}</p> : null}
      {!value && !baseline.error ? <p className="subtle">Loading detection settings…</p> : null}
      {value ? (
        <>
          <Card title="Object detection">
            {value.filter.form ? (
              <fieldset className="inline-form" disabled={busy}>
                <FilterConfigForm value={value.filter.form} supportedClasses={SUPPORTED_YOLO_CLASSES} confidenceMin={0} onChange={updateFilter} />
                <p className="subtle">Model, sampling, and camera overrides remain configured in YAML.</p>
              </fieldset>
            ) : <p className="muted">The {value.filter.backend} filter is read-only here. Edit it in YAML.</p>}
          </Card>
          <Card title="AI scene analysis">
            {value.vlm.form ? (
              <fieldset className="inline-form" disabled={busy}>
                <VlmConfigForm value={value.vlm.form} enabled={value.vlm.form.run_mode !== 'never'}
                  credentials={credentials} credentialProbeBlocked={credentialsBlockProbe(credentials, snapshot?.apply_required ?? 'none')}
                  filterClasses={value.filter.form?.config.classes ?? []}
                  onToggle={(enabled) => { if (value.vlm.form) { updateVlm(withVlmEnabled(value.vlm.form, enabled)) } }}
                  onChange={updateVlm} />
                <p className="subtle">Disabling analysis keeps its configuration for later. Preprocessing and token settings remain configured in YAML.</p>
              </fieldset>
            ) : <p className="muted">The {value.vlm.backend} analyzer is read-only here. Edit it in YAML.</p>}
          </Card>
          <div className="inline-form__actions">
            <Button disabled={busy} onClick={() => { void save() }}>{settings.savePending ? 'Saving…' : 'Save detection settings'}</Button>
            {draft ? (
              <Button variant="ghost" disabled={busy} onClick={() => {
                setDraft(null)
                setCredentialDraft({})
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
