import { useHealthQuery } from '../../api/hooks/useHealthQuery'
import { Button } from '../../components/ui/Button'
import { Card } from '../../components/ui/Card'
import { StatusBadge } from '../../components/ui/StatusBadge'

function healthTone(status: string): 'healthy' | 'degraded' | 'unhealthy' | 'unknown' {
  if (status === 'healthy' || status === 'degraded' || status === 'unhealthy') {
    return status
  }
  return 'unknown'
}

export function DashboardPage() {
  const healthQuery = useHealthQuery()

  return (
    <section className="page fade-in-up">
      <header className="page__header">
        <div>
          <h1 className="page__title">Runtime Overview</h1>
          <p className="page__lead">Live health from FastAPI control plane.</p>
        </div>
        <Button variant="ghost" onClick={() => healthQuery.refetch()} disabled={healthQuery.isFetching}>
          {healthQuery.isFetching ? 'Refreshing...' : 'Refresh'}
        </Button>
      </header>

      {healthQuery.isPending ? (
        <Card title="Loading health snapshot">
          <p className="muted">Fetching /api/v1/health...</p>
        </Card>
      ) : null}

      {healthQuery.error ? (
        <Card title="Health query failed" subtitle="Connection or API issue">
          <p className="error-text">{healthQuery.error.message}</p>
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
    </section>
  )
}
