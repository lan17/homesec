import type { RuntimeStatusSnapshot } from '../../../api/client'
import { Button } from '../../../components/ui/Button'
import { Card } from '../../../components/ui/Card'
import { StatusBadge } from '../../../components/ui/StatusBadge'
import { formatRuntimeTimestamp, runtimeStatusTone } from '../presentation'

interface RuntimeReloadBannerProps {
  hasPendingReload: boolean
  pendingReloadMessage: string | null
  actionFeedback: string | null
  runtimeStatus: RuntimeStatusSnapshot | undefined
  runtimeStatusPending: boolean
  reloadPending: boolean
  onApplyRuntimeReload: () => void
}

export function RuntimeReloadBanner({
  hasPendingReload,
  pendingReloadMessage,
  actionFeedback,
  runtimeStatus,
  runtimeStatusPending,
  reloadPending,
  onApplyRuntimeReload,
}: RuntimeReloadBannerProps) {
  return (
    <Card title="Runtime Control" subtitle="Apply pending config changes via runtime reload">
      {hasPendingReload ? (
        <div className="camera-restart-banner">
          <p className="muted">
            {pendingReloadMessage ?? 'One or more camera changes are pending runtime reload.'}
          </p>
          <Button onClick={onApplyRuntimeReload} disabled={reloadPending}>
            {reloadPending ? 'Reloading...' : 'Apply runtime reload'}
          </Button>
        </div>
      ) : (
        <p className="muted">No pending restart-required camera changes.</p>
      )}

      {actionFeedback ? <p className="subtle">{actionFeedback}</p> : null}

      {runtimeStatusPending && !runtimeStatus ? <p className="muted">Fetching runtime status...</p> : null}

      {runtimeStatus ? (
        <dl className="camera-runtime-kv">
          <div className="camera-runtime-row">
            <dt>Status</dt>
            <dd>
              <StatusBadge tone={runtimeStatusTone(runtimeStatus)}>
                {runtimeStatus.state.toUpperCase()}
              </StatusBadge>
            </dd>
          </div>
          <div className="camera-runtime-row">
            <dt>Generation</dt>
            <dd>{runtimeStatus.generation}</dd>
          </div>
          <div className="camera-runtime-row">
            <dt>Active config version</dt>
            <dd className="camera-mono">{runtimeStatus.active_config_version ?? 'n/a'}</dd>
          </div>
          <div className="camera-runtime-row">
            <dt>Last reload at</dt>
            <dd>{formatRuntimeTimestamp(runtimeStatus.last_reload_at)}</dd>
          </div>
          <div className="camera-runtime-row">
            <dt>Last reload error</dt>
            <dd>{runtimeStatus.last_reload_error ?? 'none'}</dd>
          </div>
        </dl>
      ) : null}
    </Card>
  )
}
