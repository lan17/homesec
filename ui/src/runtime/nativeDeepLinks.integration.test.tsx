// @vitest-environment happy-dom

import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter, useLocation } from 'react-router-dom'
import type { ActionPerformed } from '@capacitor/push-notifications'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

const native = vi.hoisted(() => ({
  server: null as string | null,
  token: null as string | null,
  ready: false,
  completeWrite: (() => {}) as () => void,
  deferredPhase: 'token' as 'token' | 'readiness',
  readyClearCalls: 0,
  tokenWrites: [] as string[],
  openUrl: (() => {}) as (event: { url: string }) => void,
  pushAction: (() => {}) as (action: ActionPerformed) => void,
}))

vi.mock('./nativeRuntime', () => ({ isIOSNativeApp: () => true, isNativeApp: () => true }))
vi.mock('@capacitor/app', () => ({
  App: {
    getLaunchUrl: async () => null,
    getState: async () => ({ isActive: true }),
    addListener: async (name: string, listener: (event: { url: string }) => void) => {
      if (name === 'appUrlOpen') native.openUrl = listener
      return { remove: async () => {} }
    },
  },
}))
vi.mock('@capacitor/push-notifications', () => ({
  PushNotifications: {
    checkPermissions: async () => ({ receive: 'denied' }),
    addListener: async (name: string, listener: (action: ActionPerformed) => void) => {
      if (name === 'pushNotificationActionPerformed') native.pushAction = listener
      return { remove: async () => {} }
    },
  },
}))
vi.mock('../api/homeSecAuthPlugin', () => ({
  homeSecAuthPlugin: {
    getServerBaseUrl: async () => ({ value: native.server }),
    getApiToken: async () => ({ value: native.token }),
    getAuthDisabledReady: async () => ({ value: native.ready }),
    clearApiToken: async () => { native.token = null },
    clearAuthDisabledReady: async () => {
      native.readyClearCalls += 1
      if (native.deferredPhase === 'readiness' && native.readyClearCalls === 2) {
        await new Promise<void>((resolve) => { native.completeWrite = resolve })
      }
      native.ready = false
    },
    setAuthDisabledReady: async ({ value }: { value: boolean }) => { native.ready = value },
    setServerBaseUrl: async ({ value }: { value: string }) => { native.server = value },
    setApiToken: async ({ value }: { value: string }) => {
      native.tokenWrites.push(value)
      if (native.deferredPhase === 'token') {
        await new Promise<void>((resolve) => { native.completeWrite = resolve })
      }
      native.token = value
    },
  },
}))

import { AppRouter } from '../routes/AppRouter'
import { ThemeProvider } from '../app/providers/ThemeProvider'
import {
  hydrateRuntimeApiProviders,
  isRuntimeAuthSessionReady,
  runtimeAuthTokenProvider,
  runtimeServerBaseUrlProvider,
} from '../api/client'
import { NativeDeepLinkRouter } from './nativeDeepLinks'

function LocationProbe() {
  const location = useLocation()
  return <p data-testid="location">{`${location.pathname}${location.search}${location.hash}`}</p>
}

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { 'content-type': 'application/json' },
  })
}

beforeEach(async () => {
  native.server = null
  native.token = null
  native.ready = false
  native.readyClearCalls = 0
  native.tokenWrites = []
  await hydrateRuntimeApiProviders()
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

it.each([
  ['notification', 'token', false],
  ['notification', 'token', true],
  ['custom URL', 'token', false],
  ['custom URL', 'token', true],
  ['notification', 'readiness', false],
  ['notification', 'readiness', true],
  ['custom URL', 'readiness', false],
  ['custom URL', 'readiness', true],
] as const)('opens the latest %s after %s persistence, same-turn completion=%s', async (source, phase, sameTurn) => {
  // Given: The actual native router is connecting to a protected server with a pending Keychain write.
  native.deferredPhase = phase
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (url, options) => {
    if (String(url).endsWith('/health')) {
      return jsonResponse({ status: 'healthy', pipeline: 'running', postgres: 'connected', cameras_online: 0, bootstrap_mode: false })
    }
    if (String(url).endsWith('/setup/status')) {
      return new Headers(options?.headers).get('authorization')
        ? jsonResponse({ state: 'complete', has_cameras: true, pipeline_running: true, auth_configured: true })
        : jsonResponse({ detail: 'Unauthorized', error_code: 'UNAUTHORIZED' }, 401)
    }
    if (String(url).endsWith('/cameras')) return jsonResponse({ cameras: [] })
    return jsonResponse({ detail: 'Not found' }, 404)
  })
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <MemoryRouter initialEntries={['/native-setup']}>
          <LocationProbe />
          <NativeDeepLinkRouter />
          <AppRouter />
        </MemoryRouter>
      </ThemeProvider>
    </QueryClientProvider>,
  )
  fireEvent.change(screen.getByLabelText('Server URL'), { target: { value: 'https://homesec.example' } })
  fireEvent.click(screen.getByRole('button', { name: 'Check server' }))
  await screen.findByText('Server reachable')
  fireEvent.change(screen.getByLabelText('API token'), { target: { value: 'synthetic-token' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save and continue' }))
  await waitFor(() => {
    expect(native.tokenWrites).toEqual(['synthetic-token'])
    if (phase === 'readiness') expect(native.readyClearCalls).toBe(2)
  })

  function emitRoute(route: string): void {
    if (source === 'notification') {
      native.pushAction({ actionId: 'tap', notification: { id: 'push-1', data: { route } } })
    } else {
      native.openUrl({ url: `homesec:/${route}` })
    }
  }

  // When: Incoming native links update the destination while the connection save is pending.
  await act(async () => { emitRoute('/events/first?from=notification#summary') })
  expect(screen.getByTestId('location').textContent).toBe('/native-setup')
  expect((screen.getByLabelText('Server URL') as HTMLInputElement).value).toBe('https://homesec.example')
  expect((screen.getByLabelText('API token') as HTMLInputElement).value).toBe('synthetic-token')
  expect((screen.getByRole('button', { name: 'Cancel' }) as HTMLButtonElement).disabled).toBe(true)
  if (sameTurn) {
    // Native callbacks run outside React events; act() would flush the new route too early.
    emitRoute('/events/latest?camera=front#summary')
    native.completeWrite()
    await new Promise<void>((resolve) => setTimeout(resolve, 0))
  } else {
    await act(async () => { emitRoute('/events/latest?camera=front#summary') })
    await act(async () => { native.completeWrite() })
  }

  // Then: The saved connection opens the latest destination without remounting or repeating the write.
  await waitFor(() => expect(screen.getByTestId('location').textContent).toBe('/events/latest?camera=front#summary'))
  expect(runtimeAuthTokenProvider.getTokenSync()).toBe('synthetic-token')
  expect(runtimeServerBaseUrlProvider.getBaseUrlSync()).toBe('https://homesec.example')
  expect(isRuntimeAuthSessionReady()).toBe(true)
  expect(native.tokenWrites).toEqual(['synthetic-token'])
  expect(screen.queryByRole('heading', { name: 'Connect to HomeSec' })).toBeNull()
})
