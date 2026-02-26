// @vitest-environment happy-dom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'

import type { HealthSnapshot, StatsSnapshot } from '../../api/client'
import { DashboardPage } from './DashboardPage'

const useSetupRedirectMock = vi.fn()
const useHealthQueryMock = vi.fn()
const useStatsQueryMock = vi.fn()

vi.mock('../setup/useSetupRedirect', () => ({
  useSetupRedirect: () => useSetupRedirectMock(),
}))

vi.mock('../../api/hooks/useHealthQuery', () => ({
  useHealthQuery: () => useHealthQueryMock(),
}))

vi.mock('../../api/hooks/useStatsQuery', () => ({
  useStatsQuery: () => useStatsQueryMock(),
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
})
