// @vitest-environment happy-dom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router-dom'

import { ThemeProvider } from '../app/providers/ThemeProvider'
import { AppRouter } from './AppRouter'

const routeMocks = vi.hoisted(() => ({
  getBaseUrlSync: vi.fn<() => string | null>(() => null),
  isRuntimeAuthSessionReady: vi.fn<() => boolean>(() => true),
  isIOSNativeApp: vi.fn<() => boolean>(() => false),
  useCamerasQuery: vi.fn(),
  useHealthQuery: vi.fn(),
}))

const useHealthQueryMock = routeMocks.useHealthQuery
const useCamerasQueryMock = routeMocks.useCamerasQuery

vi.mock('../api/client', () => ({
  runtimeServerBaseUrlProvider: {
    getBaseUrlSync: () => routeMocks.getBaseUrlSync(),
  },
  isRuntimeAuthSessionReady: () => routeMocks.isRuntimeAuthSessionReady(),
}))

vi.mock('../runtime/nativeRuntime', () => ({
  isIOSNativeApp: () => routeMocks.isIOSNativeApp(),
}))

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

vi.mock('../features/native-setup/NativeSetupPage', () => ({
  NativeSetupPage: () => {
    const location = useLocation()
    const state = location.state as { nativeSetupReturnTo?: string } | null
    return (
      <>
        <p>Native Setup Page</p>
        <p data-testid="native-setup-return-to">{state?.nativeSetupReturnTo ?? ''}</p>
      </>
    )
  },
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
  beforeEach(() => {
    routeMocks.getBaseUrlSync.mockReturnValue(null)
    routeMocks.isRuntimeAuthSessionReady.mockReturnValue(true)
    routeMocks.isIOSNativeApp.mockReturnValue(false)
  })

  afterEach(() => {
    cleanup()
    useHealthQueryMock.mockReset()
    useCamerasQueryMock.mockReset()
    routeMocks.getBaseUrlSync.mockReset()
    routeMocks.isRuntimeAuthSessionReady.mockReset()
    routeMocks.isIOSNativeApp.mockReset()
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

  it('renders native setup without mounting app shell queries', () => {
    // Given: User opens the native setup route before API settings exist
    renderRouter('/native-setup')

    // When / Then: The native setup surface is isolated from shell API queries
    expect(screen.getByText('Native Setup Page')).toBeTruthy()
    expect(useHealthQueryMock).not.toHaveBeenCalled()
    expect(useCamerasQueryMock).not.toHaveBeenCalled()
  })

  it('redirects iOS shell routes to native setup until a server URL is configured', async () => {
    // Given: The iOS shell starts without a configured HomeSec server URL
    routeMocks.isIOSNativeApp.mockReturnValue(true)
    routeMocks.getBaseUrlSync.mockReturnValue(null)

    // When: User opens the default shell route
    renderRouter('/live')

    // Then: The setup route is reached before shell API queries mount
    await waitFor(() => {
      expect(screen.getByTestId('location').textContent).toBe('/native-setup')
    })
    expect(screen.getByText('Native Setup Page')).toBeTruthy()
    expect(screen.getByTestId('native-setup-return-to').textContent).toBe('/live')
    expect(useHealthQueryMock).not.toHaveBeenCalled()
    expect(useCamerasQueryMock).not.toHaveBeenCalled()
  })

  it('redirects iOS shell routes to native setup when the auth session was lost', async () => {
    // Given: The iOS shell retained a server URL but lost in-memory auth state
    routeMocks.isIOSNativeApp.mockReturnValue(true)
    routeMocks.getBaseUrlSync.mockReturnValue('https://homesec.example.com')
    routeMocks.isRuntimeAuthSessionReady.mockReturnValue(false)

    // When: User opens a protected shell route after a WebView reload
    renderRouter('/events')

    // Then: The setup route is reached before unauthenticated API queries mount
    await waitFor(() => {
      expect(screen.getByTestId('location').textContent).toBe('/native-setup')
    })
    expect(screen.getByText('Native Setup Page')).toBeTruthy()
    expect(screen.getByTestId('native-setup-return-to').textContent).toBe('/events')
    expect(useHealthQueryMock).not.toHaveBeenCalled()
    expect(useCamerasQueryMock).not.toHaveBeenCalled()
  })

  it('preserves native setup return intent for deep links with filters', async () => {
    // Given: The iOS shell needs setup before opening a deep-linked event
    routeMocks.isIOSNativeApp.mockReturnValue(true)
    routeMocks.getBaseUrlSync.mockReturnValue('https://homesec.example.com')
    routeMocks.isRuntimeAuthSessionReady.mockReturnValue(false)

    // When: User opens a protected route with list context
    renderRouter('/events/clip-42?camera=front')

    // Then: Setup receives enough state to return to the intended route
    await waitFor(() => {
      expect(screen.getByTestId('location').textContent).toBe('/native-setup')
    })
    expect(screen.getByTestId('native-setup-return-to').textContent).toBe(
      '/events/clip-42?camera=front',
    )
  })

  it('allows iOS shell routes after a server URL is configured', async () => {
    // Given: The iOS shell already has a HomeSec server URL
    routeMocks.isIOSNativeApp.mockReturnValue(true)
    routeMocks.getBaseUrlSync.mockReturnValue('https://homesec.example.com')
    routeMocks.isRuntimeAuthSessionReady.mockReturnValue(true)

    // When: User opens the native shell route
    renderRouter('/live')

    // Then: The app shell renders instead of returning to setup
    await waitFor(() => {
      expect(screen.getByTestId('location').textContent).toBe('/live')
    })
    expect(screen.getByText('Live Page')).toBeTruthy()
  })

  it('redirects the old cameras route to Settings camera setup', async () => {
    // Given: User opens the old top-level camera management route
    renderRouter('/cameras')

    // When: Route redirects settle
    await waitFor(() => {
      expect(screen.getByTestId('location').textContent).toBe('/settings/cameras')
    })

    // Then: Camera setup lives under Settings instead of competing with Live
    expect(screen.getByText('Cameras Page')).toBeTruthy()
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
