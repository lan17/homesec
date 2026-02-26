import { isUnauthorizedAPIError } from '../../../api/client'
import type { PreflightCheckResponse } from '../../../api/generated/types'
import { usePreflightMutation } from '../../../api/hooks/usePreflightMutation'
import { Button } from '../../../components/ui/Button'
import { StatusBadge } from '../../../components/ui/StatusBadge'

interface WelcomeStepProps {
  isComplete: boolean
  onComplete: () => void
}

function describeCheckName(name: string): string {
  return name
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

function checkTone(check: PreflightCheckResponse): 'healthy' | 'unhealthy' {
  return check.passed ? 'healthy' : 'unhealthy'
}

function checkLabel(check: PreflightCheckResponse): string {
  return check.passed ? 'Pass' : 'Fail'
}

function toPreflightErrorMessage(error: unknown): string {
  if (isUnauthorizedAPIError(error)) {
    return 'Authentication required to run setup checks. Apply API key from Dashboard and retry.'
  }
  if (error instanceof Error && error.message.trim().length > 0) {
    return error.message
  }
  return 'Preflight checks failed. Retry to continue.'
}

export function WelcomeStep({ isComplete, onComplete }: WelcomeStepProps) {
  const preflightMutation = usePreflightMutation()
  const checks = preflightMutation.data?.checks ?? []
  const hasRunPreflight = preflightMutation.data !== undefined
  const hasFailures = checks.some((check) => !check.passed)

  return (
    <section className="wizard-step-card welcome-step">
      <header className="wizard-step-card__header">
        <p className="wizard-step-card__status">Step status: {isComplete ? 'Completed' : 'Pending'}</p>
        <Button variant="ghost" onClick={onComplete} disabled={isComplete}>
          {isComplete ? 'Step marked complete' : 'Mark step complete'}
        </Button>
      </header>

      <div className="welcome-step__overview">
        <p className="welcome-step__lead">
          HomeSec helps you capture, analyze, and review security clips from your cameras.
        </p>
        <ul className="welcome-step__bullets">
          <li>Configure camera, storage, detection, and notifier settings.</li>
          <li>Run checks before launch to catch environment issues early.</li>
          <li>Skip steps now and return later from the dashboard.</li>
        </ul>
      </div>

      <div className="welcome-step__actions">
        <Button onClick={() => preflightMutation.mutate()} disabled={preflightMutation.isPending}>
          {preflightMutation.isPending
            ? 'Running checks...'
            : hasRunPreflight
              ? 'Run checks again'
              : 'Run checks'}
        </Button>
      </div>

      {preflightMutation.isError ? (
        <p className="error-text" role="alert">
          {toPreflightErrorMessage(preflightMutation.error)}
        </p>
      ) : null}

      {checks.length > 0 ? (
        <ul className="welcome-step__checks" aria-label="Preflight checks">
          {checks.map((check) => (
            <li key={check.name} className="welcome-step__check">
              <div className="welcome-step__check-header">
                <p className="welcome-step__check-name">{describeCheckName(check.name)}</p>
                <StatusBadge tone={checkTone(check)}>{checkLabel(check)}</StatusBadge>
              </div>
              <p className="subtle">{check.message}</p>
              {typeof check.latency_ms === 'number' ? (
                <p className="subtle">Latency: {Math.round(check.latency_ms)} ms</p>
              ) : null}
            </li>
          ))}
        </ul>
      ) : null}

      {hasRunPreflight && hasFailures ? (
        <p className="subtle">
          Some checks failed. You can continue and return to fix these settings later.
        </p>
      ) : null}
    </section>
  )
}
