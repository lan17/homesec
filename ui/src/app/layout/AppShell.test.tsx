// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'

import type { CameraResponse } from '../../api/generated/types'
import { ThemeProvider } from '../providers/ThemeProvider'
import { AppShell } from './AppShell'

const useHealthQueryMock = vi.fn()
const useCamerasQueryMock = vi.fn()

vi.mock('../../api/hooks/useHealthQuery', () => ({
  useHealthQuery: () => useHealthQueryMock(),
}))

vi.mock('../../api/hooks/useCamerasQuery', () => ({
  useCamerasQuery: () => useCamerasQueryMock(),
}))

function renderShell(
  path = '/live',
  {
    cameras = [],
  }: {
    cameras?: CameraResponse[]
  } = {},
) {
  useHealthQueryMock.mockReturnValue({
    data: { status: 'healthy' },
    isError: false,
  })
  useCamerasQueryMock.mockReturnValue({
    data: cameras,
    isError: false,
  })

  render(
    <ThemeProvider>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route element={<AppShell />}>
            <Route path="/live" element={<p>Live route</p>} />
            <Route path="/events" element={<p>Events route</p>} />
            <Route path="/settings" element={<p>Settings route</p>} />
            <Route path="/system" element={<p>System route</p>} />
          </Route>
        </Routes>
      </MemoryRouter>
    </ThemeProvider>,
  )
}

describe('AppShell navigation', () => {
  afterEach(() => {
    cleanup()
    useHealthQueryMock.mockReset()
    useCamerasQueryMock.mockReset()
  })

  it('renders homeowner-first desktop navigation with System available', () => {
    // Given: App shell is mounted on the live route
    renderShell('/live')

    // When: Desktop primary navigation is inspected
    const primaryNav = screen.getByRole('navigation', { name: 'Primary' })

    // Then: Main desktop nav keeps camera operation on Live and diagnostics in System
    expect(within(primaryNav).getByRole('link', { name: 'Live' })).toBeTruthy()
    expect(within(primaryNav).getByRole('link', { name: 'Events' })).toBeTruthy()
    expect(within(primaryNav).getByRole('link', { name: 'Settings' })).toBeTruthy()
    expect(within(primaryNav).getByRole('link', { name: 'System' })).toBeTruthy()
    expect(within(primaryNav).queryByRole('link', { name: 'Cameras' })).toBeNull()
    const systemStatus = screen.getByRole('link', { name: 'All systems normal' })
    expect(systemStatus.getAttribute('href')).toBe('/system')
    expect(systemStatus.className).toContain('app-shell__header-status--nominal')
  })

  it('keeps System out of mobile bottom navigation', () => {
    // Given: App shell is mounted on the events route
    renderShell('/events')

    // When: Mobile primary navigation is inspected
    const mobileNav = screen.getByRole('navigation', { name: 'Mobile primary' })

    // Then: Mobile nav exposes only the primary homeowner destinations
    expect(within(mobileNav).getByRole('link', { name: 'Live' })).toBeTruthy()
    expect(within(mobileNav).getByRole('link', { name: 'Events' })).toBeTruthy()
    expect(within(mobileNav).getByRole('link', { name: 'Settings' })).toBeTruthy()
    expect(within(mobileNav).queryByRole('link', { name: 'Cameras' })).toBeNull()
    expect(within(mobileNav).queryByRole('link', { name: 'System' })).toBeNull()
  })

  it('surfaces offline camera issues in the compact system status', () => {
    // Given: The shell has one enabled unhealthy camera from the existing camera list API
    renderShell('/live', {
      cameras: [
        {
          name: 'front',
          enabled: true,
          healthy: false,
          last_heartbeat: 1_739_590_400,
          source_backend: 'rtsp',
          source_config: {},
        },
      ],
    })

    // When: The shell status is rendered
    const systemStatus = screen.getByRole('link', { name: '1 camera offline' })

    // Then: The non-nominal camera state stays visible instead of being hidden as nominal
    expect(systemStatus.getAttribute('href')).toBe('/system')
    expect(systemStatus.className).not.toContain('app-shell__header-status--nominal')
  })
})
