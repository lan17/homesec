import { clearApiKey, isAPIError, isUnauthorizedAPIError, saveApiKey } from '../../api/client'
import { useHealthQuery } from '../../api/hooks/useHealthQuery'
import { useStatsQuery } from '../../api/hooks/useStatsQuery'
import { ApiKeyGate } from '../../components/ui/ApiKeyGate'
import { Button } from '../../components/ui/Button'
import { Card } from '../../components/ui/Card'
import { StatusBadge } from '../../components/ui/StatusBadge'
import { describeAPIError, formatLastUpdated, healthTone } from './status'

export function DashboardPage() {
  const healthQuery = useHealthQuery()
  const statsQuery = useStatsQuery()

  const isRefreshing = healthQuery.isFetching || statsQuery.isFetching
  const latestUpdateAt = Math.max(healthQuery.dataUpdatedAt, statsQuery.dataUpdatedAt)

  async function refreshAll(): Promise<void> {
    await Promise.all([healthQuery.refetch(), statsQuery.refetch()])
  }

  async function submitApiKey(apiKey: string): Promise<void> {
    saveApiKey(apiKey)
    await statsQuery.refetch()
  }

  async function clearStoredApiKey(): Promise<void> {
    clearApiKey()
    await statsQuery.refetch()
  }

  const showLoadingState = healthQuery.isPending && !healthQuery.data && statsQuery.isPending && !statsQuery.data
  const statsUnauthorized = isUnauthorizedAPIError(statsQuery.error)
  const statsAPIError = isAPIError(statsQuery.error) ? statsQuery.error : null
  const healthAPIError = isAPIError(healthQuery.error) ? healthQuery.error : null

  return (
    <section className="page fade-in-up">
      <header className="page__header">
        <div>
          <h1 className="page__title">Runtime Overview</h1>
          <p className="page__lead">Live health and stats from FastAPI control plane.</p>
          <p className="subtle">Last updated: {formatLastUpdated(latestUpdateAt)}</p>
        </div>
        <Button variant="ghost" onClick={refreshAll} disabled={isRefreshing}>
          {isRefreshing ? 'Refreshing...' : 'Refresh'}
        </Button>
      </header>

      {showLoadingState ? (
        <Card title="Loading dashboard">
          <p className="muted">Fetching /api/v1/health and /api/v1/stats...</p>
        </Card>
      ) : null}

      {healthQuery.error && !healthQuery.data ? (
        <Card title="Health query failed" subtitle="Connection or API issue">
          <p className="error-text">
            {healthAPIError ? describeAPIError(healthAPIError) : healthQuery.error.message}
          </p>
        </Card>
      ) : null}

      {healthQuery.data ? (
        <div className="grid grid--cards">
          <Card title="System status" subtitle={`HTTP ${healthQuery.data.httpStatus}`}>
            <StatusBadge tone={healthTone(healthQuery.data.status)}>
              {healthQuery.data.status.toUpperCase()}
            </StatusBadge>
          </Card>
          <Card title="Pipeline">
            <p className="metric">{healthQuery.data.pipeline}</p>
          </Card>
          <Card title="Postgres">
            <p className="metric">{healthQuery.data.postgres}</p>
          </Card>
          <Card title="Cameras online">
            <p className="metric">{healthQuery.data.cameras_online}</p>
          </Card>
        </div>
      ) : null}

      <Card title="Daily stats">
        {statsQuery.isPending && !statsQuery.data ? (
          <p className="muted">Fetching /api/v1/stats...</p>
        ) : null}

        {statsUnauthorized ? (
          <ApiKeyGate
            busy={statsQuery.isFetching}
            onSubmit={submitApiKey}
            onClear={clearStoredApiKey}
          />
        ) : null}

        {statsAPIError && !statsUnauthorized ? (
          <p className="error-text">{describeAPIError(statsAPIError)}</p>
        ) : null}

        {statsQuery.data ? (
          <div className="grid grid--cards">
            <Card title="Clips today">
              <p className="metric">{statsQuery.data.clips_today}</p>
            </Card>
            <Card title="Alerts today">
              <p className="metric">{statsQuery.data.alerts_today}</p>
            </Card>
            <Card title="Cameras total">
              <p className="metric">{statsQuery.data.cameras_total}</p>
            </Card>
            <Card title="Cameras online">
              <p className="metric">{statsQuery.data.cameras_online}</p>
            </Card>
            <Card title="Runtime uptime (s)">
              <p className="metric">{Math.round(statsQuery.data.uptime_seconds)}</p>
            </Card>
          </div>
        ) : null}
      </Card>
    </section>
  )
}
