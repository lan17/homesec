import { clearApiKey, isUnauthorizedAPIError, saveApiKey } from '../../api/client'
import { ApiKeyGate } from '../../components/ui/ApiKeyGate'
import { Button } from '../../components/ui/Button'
import { Card } from '../../components/ui/Card'
import { describeUnknownError } from '../shared/errorPresentation'
import type { ConfigSettings } from './useConfigSettings'

export function ConfigApplyPanel({ settings }: { settings: ConfigSettings }) {
  const config = settings.configQuery.data
  const pending = settings.savePending || settings.applyPending
  const error = settings.saveError ?? settings.applyError ?? settings.configQuery.error
  const unauthorized = [settings.saveError, settings.applyError, settings.configQuery.error]
    .some(isUnauthorizedAPIError)
  return (
    <Card title="Saved settings and activation">
      {config ? (
        <>
          <p>{config.apply_required === 'none'
            ? 'Saved settings are active.'
            : 'Saved settings are waiting to be applied.'}</p>
          {config.apply_required === 'restart' ? (
            <p className="muted">Applying these settings restarts HomeSec and briefly interrupts recording. Docker or a supervisor starts it again automatically; a manually launched process must be started again.</p>
          ) : null}
          <div className="inline-form__actions">
            <Button
              disabled={pending || config.apply_required === 'none'}
              onClick={() => { void settings.applyConfig() }}
            >
              {settings.applyPending ? 'Applying…' : config.apply_required === 'restart'
                ? 'Apply and restart' : 'Apply saved settings'}
            </Button>
            <Button variant="ghost" disabled={pending || settings.configQuery.isFetching}
              onClick={() => { void settings.refreshConfig() }}>
              Refresh saved settings
            </Button>
          </div>
        </>
      ) : null}
      {settings.applyMessage ? <p role="status">{settings.applyMessage}</p> : null}
      {error ? <p className="error-text" role="alert">{describeUnknownError(error)}</p> : null}
      {settings.conflict ? (
        <p className="error-text">Saved settings changed elsewhere. Refresh saved settings and review them before retrying. Your unsaved draft is retained.</p>
      ) : null}
      {unauthorized ? <ApiKeyGate busy={pending || settings.configQuery.isFetching}
        onSubmit={async (apiKey) => { saveApiKey(apiKey); await settings.refreshConfig() }}
        onClear={async () => { clearApiKey(); await settings.refreshConfig() }} /> : null}
    </Card>
  )
}
