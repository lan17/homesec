// @vitest-environment happy-dom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'

import type { HealthSnapshot, PostgresBackupStatusSnapshot, StatsSnapshot } from '../../api/client'
import { DashboardPage } from './DashboardPage'

const useSetupRedirectMock = vi.fn()
const useHealthQueryMock = vi.fn()
const useStatsQueryMock = vi.fn()
const usePostgresBackupStatusQueryMock = vi.fn()
const usePostgresBackupRunMutationMock = vi.fn()

vi.mock('../setup/useSetupRedirect', () => ({
  useSetupRedirect: () => useSetupRedirectMock(),
}))

vi.mock('../../api/hooks/useHealthQuery', () => ({
  useHealthQuery: () => useHealthQueryMock(),
}))

vi.mock('../../api/hooks/useStatsQuery', () => ({
  useStatsQuery: () => useStatsQueryMock(),
}))

vi.mock('../../api/hooks/usePostgresBackupStatusQuery', () => ({
  usePostgresBackupStatusQuery: () => usePostgresBackupStatusQueryMock(),
}))

vi.mock('../../api/hooks/usePostgresBackupRunMutation', () => ({
  usePostgresBackupRunMutation: () => usePostgresBackupRunMutationMock(),
}))

function defaultHealthSnapshot(): HealthSnapshot {
  return {
    httpStatus: 200,
    status: 'healthy',
    pipeline: 'running',
    postgres: 'connected',
    cameras_online: 1,
    bootstrap_mode: false,
  }
}

function defaultStatsSnapshot(): StatsSnapshot {
  return {
    httpStatus: 200,
    clips_today: 5,
    alerts_today: 1,
    cameras_total: 2,
    cameras_online: 1,
    uptime_seconds: 120.0,
  }
}

function defaultPostgresBackupStatus(): PostgresBackupStatusSnapshot {
  return {
    httpStatus: 200,
    enabled: true,
    running: false,
    available: true,
    unavailable_reason: null,
    last_attempted_at: '2026-04-23T16:00:00Z',
    last_success_at: '2026-04-23T16:00:00Z',
    last_error: null,
    last_local_path: '/var/lib/homesec/backups/postgres/homesec-postgres-20260423-160000.dump',
    last_uploaded_uri: 'local:/storage/backups/homesec-postgres-20260423-160000.dump',
    next_run_at: '2026-04-24T16:00:00Z',
    pending_remote_delete_count: 0,
  }
}

function setupDashboard({
  isChecking = false,
  shouldRedirect = false,
}: {
  isChecking?: boolean
  shouldRedirect?: boolean
} = {}) {
  useSetupRedirectMock.mockReturnValue({
    isChecking,
    shouldRedirect,
  })
  useHealthQueryMock.mockReturnValue({
    data: defaultHealthSnapshot(),
    isPending: false,
    isFetching: false,
    error: null,
    refetch: vi.fn().mockResolvedValue(undefined),
    dataUpdatedAt: 1_739_590_400_000,
  })
  useStatsQueryMock.mockReturnValue({
    data: defaultStatsSnapshot(),
    isPending: false,
    isFetching: false,
    error: null,
    refetch: vi.fn().mockResolvedValue(undefined),
    dataUpdatedAt: 1_739_590_405_000,
  })
  usePostgresBackupStatusQueryMock.mockReturnValue({
    data: defaultPostgresBackupStatus(),
    isPending: false,
    isFetching: false,
    error: null,
    refetch: vi.fn().mockResolvedValue(undefined),
    dataUpdatedAt: 1_739_590_406_000,
  })
  usePostgresBackupRunMutationMock.mockReturnValue({
    isPending: false,
    error: null,
    mutateAsync: vi.fn().mockResolvedValue({ accepted: true, message: 'accepted', httpStatus: 202 }),
  })

  render(
    <MemoryRouter initialEntries={['/']}>
      <DashboardPage />
    </MemoryRouter>,
  )
}

describe('DashboardPage setup redirect behavior', () => {
  beforeEach(() => {
    useSetupRedirectMock.mockReset()
    useHealthQueryMock.mockReset()
    useStatsQueryMock.mockReset()
    usePostgresBackupStatusQueryMock.mockReset()
    usePostgresBackupRunMutationMock.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it('renders nothing while setup redirect state is still checking', () => {
    // Given: Setup redirect hook reports a pending setup-status check
    setupDashboard({ isChecking: true, shouldRedirect: false })

    // When: Dashboard page is rendered
    const title = screen.queryByRole('heading', { name: 'Runtime Overview' })

    // Then: Dashboard content is withheld to avoid first-run flash
    expect(title).toBeNull()
  })

  it('shows a Re-run setup link during normal dashboard rendering', () => {
    // Given: Setup redirect hook indicates dashboard can render normally
    setupDashboard({ isChecking: false, shouldRedirect: false })

    // When: Dashboard header is rendered
    const setupLink = screen.getByRole('link', { name: 'Re-run setup wizard' })

    // Then: Operator can navigate back to the setup flow from dashboard
    expect(setupLink.getAttribute('href')).toBe('/setup')
  })

  it('renders database backup status and triggers manual backup', async () => {
    // Given: Backup status is available and the manual trigger mutation succeeds
    const user = userEvent.setup()
    const refetch = vi.fn().mockResolvedValue(undefined)
    const mutateAsync = vi.fn().mockResolvedValue({
      accepted: true,
      message: 'Postgres backup accepted',
      httpStatus: 202,
    })
    setupDashboard({ isChecking: false, shouldRedirect: false })
    usePostgresBackupStatusQueryMock.mockReturnValue({
      data: defaultPostgresBackupStatus(),
      isPending: false,
      isFetching: false,
      error: null,
      refetch,
      dataUpdatedAt: 1_739_590_406_000,
    })
    usePostgresBackupRunMutationMock.mockReturnValue({
      isPending: false,
      error: null,
      mutateAsync,
    })
    cleanup()
    render(
      <MemoryRouter initialEntries={['/']}>
        <DashboardPage />
      </MemoryRouter>,
    )

    // When: The operator clicks the backup action
    await user.click(screen.getByRole('button', { name: 'Back up now' }))

    // Then: The mutation runs and status is refreshed
    expect(screen.getByRole('heading', { name: 'Database backups' })).toBeTruthy()
    expect(screen.getByText('Enabled')).toBeTruthy()
    expect(mutateAsync).toHaveBeenCalledTimes(1)
    expect(refetch).toHaveBeenCalledTimes(1)
  })
})
