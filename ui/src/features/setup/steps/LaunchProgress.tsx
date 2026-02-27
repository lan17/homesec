import { StatusBadge } from '../../../components/ui/StatusBadge'

export type LaunchProgressStatus = 'launching' | 'started' | 'failed'

interface LaunchProgressProps {
  status: LaunchProgressStatus
  error?: string | null
}

function statusTone(status: LaunchProgressStatus): 'healthy' | 'unknown' | 'unhealthy' {
  switch (status) {
    case 'launching':
      return 'unknown'
    case 'started':
      return 'healthy'
    case 'failed':
      return 'unhealthy'
  }
}

function statusLabel(status: LaunchProgressStatus): string {
  switch (status) {
    case 'launching':
      return 'Launching'
    case 'started':
      return 'Started'
    case 'failed':
      return 'Failed'
  }
}

function statusMessage(status: LaunchProgressStatus): string {
  switch (status) {
    case 'launching':
      return 'Writing config and waiting for HomeSec runtime startup...'
    case 'started':
      return 'Setup complete. HomeSec runtime is healthy.'
    case 'failed':
      return 'Launch failed before runtime reached healthy state.'
  }
}

export function LaunchProgress({ status, error = null }: LaunchProgressProps) {
  return (
    <section className="launch-progress" aria-live="polite">
      <header className="launch-progress__header">
        <h3 className="launch-progress__title">Launch progress</h3>
        <StatusBadge tone={statusTone(status)}>{statusLabel(status)}</StatusBadge>
      </header>
      <p className="subtle">{statusMessage(status)}</p>
      {status === 'failed' && error ? (
        <p className="error-text" role="alert">
          {error}
        </p>
      ) : null}
    </section>
  )
}
