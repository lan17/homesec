// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router-dom'

import { ThemeProvider } from '../app/providers/ThemeProvider'
import { AppRouter } from './AppRouter'

const useHealthQueryMock = vi.fn()
const useCamerasQueryMock = vi.fn()

vi.mock('../api/hooks/useHealthQuery', () => ({
  useHealthQuery: () => useHealthQueryMock(),
}))

vi.mock('../api/hooks/useCamerasQuery', () => ({
  useCamerasQuery: () => useCamerasQueryMock(),
}))

vi.mock('../features/live/LivePage', () => ({
  LivePage: () => <p>Live Page</p>,
}))

vi.mock('../features/clips/ClipsPage', () => ({
  ClipsPage: () => <p>Events Page</p>,
}))

vi.mock('../features/clips/ClipDetailPage', () => ({
  ClipDetailPage: () => <p>Event Detail Page</p>,
}))

vi.mock('../features/cameras/CamerasPage', () => ({
  CamerasPage: () => <p>Cameras Page</p>,
}))

vi.mock('../features/settings/SettingsPage', () => ({
  SettingsPage: () => <p>Settings Page</p>,
}))

vi.mock('../features/system/SystemPage', () => ({
  SystemPage: () => <p>System Page</p>,
}))

vi.mock('../features/setup/SetupPage', () => ({
  SetupPage: () => <p>Setup Page</p>,
}))

function LocationProbe() {
  const location = useLocation()
  return <p data-testid="location">{`${location.pathname}${location.search}`}</p>
}

function renderRouter(initialPath: string) {
  useHealthQueryMock.mockReturnValue({
    data: { status: 'healthy' },
    isError: false,
  })
  useCamerasQueryMock.mockReturnValue({
    data: [],
    isError: false,
  })

  render(
    <ThemeProvider>
      <MemoryRouter initialEntries={[initialPath]}>
        <AppRouter />
        <LocationProbe />
      </MemoryRouter>
    </ThemeProvider>,
  )
}

describe('AppRouter route cleanup', () => {
  afterEach(() => {
    cleanup()
    useHealthQueryMock.mockReset()
    useCamerasQueryMock.mockReset()
  })

  it('redirects the root route to Live', async () => {
    // Given: User opens the historical root route
    renderRouter('/')

    // When: Route redirects settle
    await waitFor(() => {
      expect(screen.getByTestId('location').textContent).toBe('/live')
    })

    // Then: The homeowner-first Live page is the landing surface
    expect(screen.getByText('Live Page')).toBeTruthy()
  })

  it('redirects dashboard to System', async () => {
    // Given: User opens the old dashboard route
    renderRouter('/dashboard')

    // When: Route redirects settle
    await waitFor(() => {
      expect(screen.getByTestId('location').textContent).toBe('/system')
    })

    // Then: Runtime-style content is reached through System
    expect(screen.getByText('System Page')).toBeTruthy()
  })

  it('redirects clips list to Events while preserving filters', async () => {
    // Given: User opens a clips URL with URL-synced filters
    renderRouter('/clips?camera=front_door&detected=any')

    // When: Route redirects settle
    await waitFor(() => {
      expect(screen.getByTestId('location').textContent).toBe('/events?camera=front_door&detected=any')
    })

    // Then: The Events route handles the existing filtered list context
    expect(screen.getByText('Events Page')).toBeTruthy()
  })

  it('redirects clip detail to event detail while preserving search context', async () => {
    // Given: User opens an old clip detail URL from a filtered list
    renderRouter('/clips/garage%2Fclip%201?camera=front_door')

    // When: Route redirects settle
    await waitFor(() => {
      expect(screen.getByTestId('location').textContent).toBe('/events/garage%2Fclip%201?camera=front_door')
    })

    // Then: The event detail route remains backward compatible with the old path
    expect(screen.getByText('Event Detail Page')).toBeTruthy()
  })
})
