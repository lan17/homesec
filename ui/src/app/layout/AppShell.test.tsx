// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'

import { ThemeProvider } from '../providers/ThemeProvider'
import { AppShell } from './AppShell'

const useHealthQueryMock = vi.fn()

vi.mock('../../api/hooks/useHealthQuery', () => ({
  useHealthQuery: () => useHealthQueryMock(),
}))

function renderShell(path = '/live') {
  useHealthQueryMock.mockReturnValue({
    data: { status: 'healthy' },
    isError: false,
  })

  render(
    <ThemeProvider>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route element={<AppShell />}>
            <Route path="/live" element={<p>Live route</p>} />
            <Route path="/events" element={<p>Events route</p>} />
            <Route path="/cameras" element={<p>Cameras route</p>} />
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
  })

  it('renders homeowner-first desktop navigation with System available', () => {
    // Given: App shell is mounted on the live route
    renderShell('/live')

    // When: Desktop primary navigation is inspected
    const primaryNav = screen.getByRole('navigation', { name: 'Primary' })

    // Then: Main desktop nav uses M1 homeowner labels including System
    expect(within(primaryNav).getByRole('link', { name: 'Live' })).toBeTruthy()
    expect(within(primaryNav).getByRole('link', { name: 'Events' })).toBeTruthy()
    expect(within(primaryNav).getByRole('link', { name: 'Cameras' })).toBeTruthy()
    expect(within(primaryNav).getByRole('link', { name: 'Settings' })).toBeTruthy()
    expect(within(primaryNav).getByRole('link', { name: 'System' })).toBeTruthy()
    expect(screen.getByRole('link', { name: 'System OK' }).getAttribute('href')).toBe('/system')
  })

  it('keeps System out of mobile bottom navigation', () => {
    // Given: App shell is mounted on the events route
    renderShell('/events')

    // When: Mobile primary navigation is inspected
    const mobileNav = screen.getByRole('navigation', { name: 'Mobile primary' })

    // Then: Mobile nav exposes only the four homeowner-first destinations
    expect(within(mobileNav).getByRole('link', { name: 'Live' })).toBeTruthy()
    expect(within(mobileNav).getByRole('link', { name: 'Events' })).toBeTruthy()
    expect(within(mobileNav).getByRole('link', { name: 'Cameras' })).toBeTruthy()
    expect(within(mobileNav).getByRole('link', { name: 'Settings' })).toBeTruthy()
    expect(within(mobileNav).queryByRole('link', { name: 'System' })).toBeNull()
  })
})
