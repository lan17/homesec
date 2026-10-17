// @vitest-environment happy-dom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
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

const HEALTH_PAYLOAD = {
  status: 'healthy', pipeline: 'running', postgres: 'connected',
  cameras_online: 1, bootstrap_mode: false,
}
const SETUP_PAYLOAD = {
  state: 'complete', has_cameras: true, pipeline_running: true, auth_configured: true,
}

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status, headers: { 'content-type': 'application/json' },
  })
}

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

  it.each(['health', 'status', 'token'])('cancels a stalled %s check without changing the saved connection', async (phase) => {
    // Given: A saved native connection and a replacement server with a stalled read-only API
    await hydrateRuntimeApiProviders()
    let releaseResponse: (response: Response) => void = () => {}
    const isPendingRequest = (url: unknown, options?: RequestInit) => {
      const token = new Headers(options?.headers).get('authorization')
      return String(url).startsWith('https://replacement.example/') && (
        phase === 'health'
          ? String(url).endsWith('/health')
          : String(url).endsWith('/setup/status') && (phase !== 'token' || token != null)
      )
    }
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (url, options) => {
      if (String(url).startsWith('https://old-server.example/')) {
        throw new TypeError('Load failed')
      }
      if (isPendingRequest(url, options)) {
        // Deliberately retain a late response even after abort to verify stale-result suppression.
        return new Promise<Response>((resolve) => { releaseResponse = resolve })
      }
      if (String(url).endsWith('/health')) return jsonResponse(HEALTH_PAYLOAD)
      return jsonResponse({ detail: 'Unauthorized', error_code: 'UNAUTHORIZED' }, 401)
    })
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <ThemeProvider><QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={['/live']}><AppRouter /></MemoryRouter>
      </QueryClientProvider></ThemeProvider>,
    )
    await screen.findByText('Live view unavailable')
    fireEvent.click(screen.getAllByRole('link', { name: 'Settings' })[0]!)
    fireEvent.click(await screen.findByRole('link', { name: 'Change server' }))
    fireEvent.change(screen.getByLabelText('Server URL'), { target: { value: 'https://replacement.example' } })
    fireEvent.click(screen.getByRole('button', { name: 'Check server' }))
    if (phase === 'token') {
      await screen.findByText('Server reachable')
      fireEvent.change(screen.getByLabelText('API token'), { target: { value: 'replacement-token' } })
      fireEvent.click(screen.getByRole('button', { name: 'Save and continue' }))
    }
    await waitFor(() => {
      expect(fetchSpy.mock.calls.some(([url, options]) => isPendingRequest(url, options))).toBe(true)
    })
    const signal = fetchSpy.mock.calls.find(([url, options]) => isPendingRequest(url, options))?.[1]?.signal
    expect(signal).toBeInstanceOf(AbortSignal)

    // When: The user cancels while the endpoint still has not responded
    const cancel = screen.getByRole('button', { name: 'Cancel' }) as HTMLButtonElement
    expect(cancel.disabled).toBe(false)
    fireEvent.click(cancel)
    await screen.findByText('Live view unavailable')
    await act(async () => { releaseResponse(jsonResponse(phase === 'health' ? HEALTH_PAYLOAD : SETUP_PAYLOAD)) })

    // Then: The request is aborted and its late response cannot replace stored settings
    expect(signal?.aborted).toBe(true)
    expect(runtimeServerBaseUrlProvider.getBaseUrlSync()).toBe('https://old-server.example')
    expect(runtimeAuthTokenProvider.getTokenSync()).toBe('synthetic-token')
    expect(screen.queryByRole('heading', { name: 'Connect to HomeSec' })).toBeNull()
    if (phase === 'health') {
      expect(fetchSpy.mock.calls.some(([url]) => String(url) === 'https://replacement.example/api/v1/setup/status')).toBe(false)
    }
    queryClient.clear()
  })
})
