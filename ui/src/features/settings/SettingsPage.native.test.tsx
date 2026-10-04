// @vitest-environment happy-dom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'

const nativeMocks = vi.hoisted(() => ({ isIOSNativeApp: vi.fn(() => true) }))

vi.mock('../../runtime/nativeRuntime', () => ({
  isIOSNativeApp: nativeMocks.isIOSNativeApp,
  isNativeApp: nativeMocks.isIOSNativeApp,
}))
vi.mock('../../api/homeSecAuthPlugin', () => ({
  homeSecAuthPlugin: {
    getServerBaseUrl: async () => ({ value: 'https://old-server.example' }),
    getApiToken: async () => ({ value: 'synthetic-token' }),
    getAuthDisabledReady: async () => ({ value: false }),
  },
}))
vi.mock('@capacitor/app', () => ({
  App: {
    getState: async () => ({ isActive: true }),
    addListener: async () => ({ remove: async () => {} }),
  },
}))
vi.mock('@capacitor/push-notifications', () => ({
  PushNotifications: { checkPermissions: async () => ({ receive: 'denied' }) },
}))

import {
  hydrateRuntimeApiProviders,
  runtimeAuthTokenProvider,
  runtimeServerBaseUrlProvider,
} from '../../api/client'
import { ThemeProvider } from '../../app/providers/ThemeProvider'
import { AppRouter } from '../../routes/AppRouter'
import { SettingsPage } from './SettingsPage'

describe('Settings connection recovery', () => {
  beforeEach(() => {
    nativeMocks.isIOSNativeApp.mockReturnValue(true)
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it('reopens native connection setup when the saved server is unreachable', async () => {
    // Given: A native app retains credentials for a server that no longer responds
    await hydrateRuntimeApiProviders()
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new TypeError('Load failed'))
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <ThemeProvider>
        <QueryClientProvider client={queryClient}>
          <MemoryRouter initialEntries={['/live']}>
            <AppRouter />
          </MemoryRouter>
        </QueryClientProvider>
      </ThemeProvider>,
    )
    await screen.findByText('Live view unavailable')

    // When: The user opens Settings and chooses to change the connection
    fireEvent.click(screen.getAllByRole('link', { name: 'Settings' })[0]!)
    fireEvent.click(await screen.findByRole('link', { name: 'Change server' }))

    // Then: The existing connection form is reachable without a working server
    await screen.findByRole('heading', { name: 'Connect to HomeSec' })
    expect(screen.getByLabelText('Server URL')).toBeTruthy()

    // When: Leaving connection setup without saving a replacement
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))

    // Then: The app returns to Live with the existing connection intact
    await screen.findByText('Live view unavailable')
    expect(runtimeServerBaseUrlProvider.getBaseUrlSync()).toBe('https://old-server.example')
    expect(runtimeAuthTokenProvider.getTokenSync()).toBe('synthetic-token')
    queryClient.clear()
  })

  it('keeps native connection setup out of browser settings', () => {
    // Given: Settings is rendered in the browser
    nativeMocks.isIOSNativeApp.mockReturnValue(false)

    // When: Opening Settings
    render(<MemoryRouter><SettingsPage /></MemoryRouter>)

    // Then: Browser settings retain their existing destinations
    expect(screen.queryByRole('link', { name: 'Change server' })).toBeNull()
    expect(screen.getByRole('link', { name: 'Camera setup' })).toBeTruthy()
  })
})
