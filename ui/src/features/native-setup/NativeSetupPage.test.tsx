// @vitest-environment happy-dom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'

import {
  BROWSER_AUTH_TOKEN_STORAGE_KEY,
  InMemoryAuthTokenProvider,
} from '../../api/tokenProvider'
import { BROWSER_SERVER_BASE_URL_STORAGE_KEY } from '../../api/serverBaseUrlProvider'
import { BrowserServerBaseUrlProvider } from '../../api/serverBaseUrlProvider'
import { HomeSecApiClient } from '../../api/client'
import { WIZARD_STATE_STORAGE_KEY } from '../../runtime/setupWizardStorage'
import { useSetupRedirect } from '../setup/useSetupRedirect'
import { NativeSetupPage, type NativeSetupPageProps } from './NativeSetupPage'

const HEALTH_PAYLOAD = {
  status: 'healthy',
  pipeline: 'running',
  postgres: 'connected',
  cameras_online: 1,
  bootstrap_mode: false,
}

const SETUP_PAYLOAD = {
  state: 'complete',
  has_cameras: true,
  pipeline_running: true,
  auth_configured: true,
}

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { 'content-type': 'application/json' },
  })
}

function unauthorizedResponse(): Response {
  return jsonResponse({ detail: 'Unauthorized', error_code: 'UNAUTHORIZED' }, 401)
}

function authorizationHeader(call: Parameters<typeof fetch>[1] | undefined): string | undefined {
  const headers = call?.headers
  return headers && !Array.isArray(headers) && !(headers instanceof Headers)
    ? headers.Authorization
    : undefined
}

function createTestQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  })
}

function renderNativeSetup(
  props: NativeSetupPageProps = {},
  queryClient: QueryClient = createTestQueryClient(),
  setupState: unknown = undefined,
): QueryClient {
  const initialEntry =
    setupState === undefined
      ? '/native-setup'
      : {
          pathname: '/native-setup',
          state: setupState,
        }

  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[initialEntry]}>
        <Routes>
          <Route path="/native-setup" element={<NativeSetupPage {...props} />} />
          <Route path="/live" element={<p>Live route</p>} />
          <Route path="/events/:clipId" element={<p>Event route</p>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
  return queryClient
}

function FreshServerLive() {
  useSetupRedirect()
  return <p>Live route</p>
}

describe('NativeSetupPage', () => {
  beforeEach(() => {
    window.sessionStorage.clear()
    window.localStorage.clear()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    window.sessionStorage.clear()
    window.localStorage.clear()
  })


  it.each([false, true])('isolates unfinished server setup drafts, same server=%s', async (sameServer) => {
    // Given: Server A has unfinished wizard progress and stored credentials
    const user = userEvent.setup()
    const draft = JSON.stringify({ schemaVersion: 1, currentStep: 5, stepData: { storage: { root: '/server-a-storage' } }, completedSteps: ['storage'], skippedSteps: [] })
    window.localStorage.setItem(WIZARD_STATE_STORAGE_KEY, draft)
    const authTokenProvider = new InMemoryAuthTokenProvider()
    authTokenProvider.setTokenSync('server-a-token')
    const serverBaseUrlProvider = new BrowserServerBaseUrlProvider('https://a.example')
    const freshStatus = { ...SETUP_PAYLOAD, state: 'fresh', has_cameras: false, pipeline_running: false }
    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(jsonResponse(HEALTH_PAYLOAD))
      .mockResolvedValueOnce(unauthorizedResponse())
      .mockImplementation(async () => jsonResponse(freshStatus))
    render(
      <QueryClientProvider client={createTestQueryClient()}>
        <MemoryRouter initialEntries={['/native-setup']}>
          <Routes>
            <Route path="/native-setup" element={<NativeSetupPage authTokenProvider={authTokenProvider} serverBaseUrlProvider={serverBaseUrlProvider} />} />
            <Route path="/live" element={<FreshServerLive />} />
            <Route path="/setup" element={<p>Fresh server onboarding</p>} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    )

    // When: A validated connection is saved, with normalized URL equality as the boundary
    await user.type(screen.getByLabelText('Server URL'), sameServer ? ' https://a.example/ ' : 'https://b.example')
    await user.click(screen.getByRole('button', { name: 'Check server' }))
    await screen.findByText('Server reachable')
    await user.type(screen.getByLabelText('API token'), 'replacement-token')
    await user.click(screen.getByRole('button', { name: 'Save and continue' }))

    // Then: A different fresh server starts onboarding without A's configuration; A retains its draft
    await screen.findByText(sameServer ? 'Live route' : 'Fresh server onboarding')
    expect(window.localStorage.getItem(WIZARD_STATE_STORAGE_KEY)).toBe(sameServer ? draft : null)
  })

  it.each([false, true])('prevents cancellation during credential persistence, saved connection=%s', async (savedConnection) => {
    // Given: Token validation succeeds but the secure credential write remains pending
    const user = userEvent.setup()
    const authTokenProvider = new InMemoryAuthTokenProvider()
    if (savedConnection) authTokenProvider.setTokenSync('old-token')
    const serverBaseUrlProvider = new BrowserServerBaseUrlProvider(savedConnection ? 'https://old.example' : '')
    let completeWrite: () => void = () => {}
    vi.spyOn(authTokenProvider, 'setToken').mockImplementation(async (token) => {
      await new Promise<void>((resolve) => { completeWrite = resolve })
      authTokenProvider.setTokenSync(token)
    })
    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(jsonResponse(HEALTH_PAYLOAD))
      .mockResolvedValueOnce(unauthorizedResponse())
      .mockResolvedValueOnce(jsonResponse(SETUP_PAYLOAD))
    renderNativeSetup({ authTokenProvider, serverBaseUrlProvider })

    // When: Saving reaches the credential persistence boundary
    await user.type(screen.getByLabelText('Server URL'), 'https://replacement.example')
    await user.click(screen.getByRole('button', { name: 'Check server' }))
    await screen.findByText('Server reachable')
    await user.type(screen.getByLabelText('API token'), 'replacement-token')
    await user.click(screen.getByRole('button', { name: 'Save and continue' }))

    // Then: Cancellation cannot interrupt the settings write
    await waitFor(() => {
      expect((screen.getByRole('button', { name: 'Cancel' }) as HTMLButtonElement).disabled).toBe(true)
    })
    await act(async () => { completeWrite() })
    await screen.findByText('Live route')
    expect(authTokenProvider.getTokenSync()).toBe('replacement-token')
    expect(serverBaseUrlProvider.getBaseUrlSync()).toBe('https://replacement.example')
  })

  it.each(['health', 'status', 'token'])('cancels first-connection %s validation without bypassing setup', async (phase) => {
    // Given: First connection setup has no saved credentials and a stalled read-only request.
    const user = userEvent.setup()
    const authTokenProvider = new InMemoryAuthTokenProvider()
    const serverBaseUrlProvider = new BrowserServerBaseUrlProvider('')
    let releaseResponse: (response: Response) => void = () => {}
    const isPendingRequest = (url: unknown, options?: RequestInit) => {
      const token = new Headers(options?.headers).get('authorization')
      return String(url).startsWith('https://stalled.example/') && (
        phase === 'health'
          ? String(url).endsWith('/health')
          : String(url).endsWith('/setup/status') && (phase !== 'token' || token != null)
      )
    }
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (url, options) => {
      if (isPendingRequest(url, options)) {
        // A late response must be ignored even when the transport does not honor abort.
        return new Promise<Response>((resolve) => { releaseResponse = resolve })
      }
      if (String(url).endsWith('/health')) return jsonResponse(HEALTH_PAYLOAD)
      return unauthorizedResponse()
    })
    renderNativeSetup({ authTokenProvider, serverBaseUrlProvider })
    await user.type(screen.getByLabelText('Server URL'), 'https://stalled.example')
    await user.click(screen.getByRole('button', { name: 'Check server' }))
    if (phase === 'token') {
      await screen.findByText('Server reachable')
      await user.type(screen.getByLabelText('API token'), 'candidate-token')
      await user.click(screen.getByRole('button', { name: 'Save and continue' }))
    }
    await waitFor(() => {
      expect(fetchSpy.mock.calls.some(([url, options]) => isPendingRequest(url, options))).toBe(true)
    })
    const signal = fetchSpy.mock.calls.find(([url, options]) => isPendingRequest(url, options))?.[1]?.signal

    // When: Cancelling the pending check before the server responds.
    await user.click(screen.getByRole('button', { name: 'Cancel' }))

    // Then: Inputs immediately recover, but no connection is saved and setup remains required.
    expect(signal?.aborted).toBe(true)
    expect((screen.getByLabelText('Server URL') as HTMLInputElement).disabled).toBe(false)
    expect((screen.getByRole('button', { name: 'Check server' }) as HTMLButtonElement).disabled).toBe(false)
    expect(screen.queryByText('Live route')).toBeNull()
    await act(async () => { releaseResponse(jsonResponse(phase === 'health' ? HEALTH_PAYLOAD : SETUP_PAYLOAD)) })
    expect(authTokenProvider.getTokenSync()).toBeNull()
    expect(serverBaseUrlProvider.getBaseUrlSync()).toBeNull()
    expect(screen.getByRole('heading', { name: 'Connect to HomeSec' })).toBeTruthy()
    if (phase === 'health') {
      expect(fetchSpy.mock.calls.some(([url]) => String(url).endsWith('/setup/status'))).toBe(false)
    }
    await user.clear(screen.getByLabelText('Server URL'))
    await user.type(screen.getByLabelText('Server URL'), 'https://retry.example')
    await user.click(screen.getByRole('button', { name: 'Check server' }))
    await screen.findByText('Server reachable')
    expect(serverBaseUrlProvider.getBaseUrlSync()).toBeNull()
  })

  it('fails closed without old credentials or cached data when a server switch cannot save its token', async () => {
    // Given: Server A auth/cache and server B whose token validates but cannot be persisted
    const user = userEvent.setup()
    const authTokenProvider = new InMemoryAuthTokenProvider()
    authTokenProvider.setTokenSync('server-a-secret')
    vi.spyOn(authTokenProvider, 'setToken').mockRejectedValue(new Error('Keychain write failed'))
    const serverBaseUrlProvider = new BrowserServerBaseUrlProvider('https://a.example')
    const queryClient = createTestQueryClient()
    queryClient.setQueryData(['private-server-a-data'], ['private-event'])
    window.localStorage.setItem(WIZARD_STATE_STORAGE_KEY, 'server-a-draft')
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(jsonResponse(HEALTH_PAYLOAD))
      .mockResolvedValueOnce(unauthorizedResponse())
      .mockResolvedValueOnce(jsonResponse(SETUP_PAYLOAD))
      .mockResolvedValueOnce(jsonResponse([]))
    renderNativeSetup({ authTokenProvider, serverBaseUrlProvider }, queryClient)

    // When: Switching to B fails after its URL is saved
    await user.type(screen.getByLabelText('Server URL'), 'https://b.example')
    await user.click(screen.getByRole('button', { name: 'Check server' }))
    await screen.findByText('Server reachable')
    await user.type(screen.getByLabelText('API token'), 'server-b-secret')
    await user.click(screen.getByRole('button', { name: 'Save and continue' }))
    await screen.findByText('Unable to validate token: Keychain write failed')

    // Then: Setup remains visible and neither the old token nor its data crosses into B
    expect(screen.queryByText('Live route')).toBeNull()
    expect(authTokenProvider.getTokenSync()).toBeNull()
    expect(queryClient.getQueryData(['private-server-a-data'])).toBeUndefined()
    expect(window.localStorage.getItem(WIZARD_STATE_STORAGE_KEY)).toBeNull()
    const client = new HomeSecApiClient('', { authTokenProvider, serverBaseUrlProvider })
    await client.getCameras()
    expect(fetchSpy.mock.calls[3]?.[0]).toBe('https://b.example/api/v1/cameras')
    expect(authorizationHeader(fetchSpy.mock.calls[3]?.[1])).toBeUndefined()
  })

  it('validates server and token before saving settings and routing to Live', async () => {
    // Given: A reachable HTTP LAN server and an old stored token from another server
    const user = userEvent.setup()
    window.sessionStorage.setItem(BROWSER_AUTH_TOKEN_STORAGE_KEY, 'old-secret')
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(jsonResponse(HEALTH_PAYLOAD))
      .mockResolvedValueOnce(unauthorizedResponse())
      .mockResolvedValueOnce(jsonResponse(SETUP_PAYLOAD))
    renderNativeSetup()

    // When: User checks the server URL and submits a valid token
    await user.type(screen.getByLabelText('Server URL'), ' http://192.168.1.10:8081/// ')
    await user.click(screen.getByRole('button', { name: 'Check server' }))
    await screen.findByText('Server reachable')
    expect(screen.getByText('Plain HTTP is visible on the network. Prefer HTTPS or VPN for iOS access.')).toBeTruthy()
    await user.type(screen.getByLabelText('API token'), ' token-123 ')
    await user.click(screen.getByRole('button', { name: 'Save and continue' }))

    // Then: Requests use the runtime base URL, settings are saved, and the app opens Live
    await waitFor(() => {
      expect(screen.getByText('Live route')).toBeTruthy()
    })
    expect(fetchSpy.mock.calls[0]?.[0]).toBe('http://192.168.1.10:8081/api/v1/health')
    expect(fetchSpy.mock.calls[1]?.[0]).toBe('http://192.168.1.10:8081/api/v1/setup/status')
    expect(authorizationHeader(fetchSpy.mock.calls[0]?.[1])).toBeUndefined()
    expect(authorizationHeader(fetchSpy.mock.calls[1]?.[1])).toBeUndefined()
    expect(fetchSpy.mock.calls[1]?.[1]).toMatchObject({
      headers: {
        Accept: 'application/json',
      },
    })
    expect(fetchSpy.mock.calls[2]?.[1]).toMatchObject({
      headers: {
        Accept: 'application/json',
        Authorization: 'Bearer token-123',
      },
    })
    expect(window.sessionStorage.getItem(BROWSER_SERVER_BASE_URL_STORAGE_KEY)).toBe(
      'http://192.168.1.10:8081',
    )
    expect(window.sessionStorage.getItem(BROWSER_AUTH_TOKEN_STORAGE_KEY)).toBe('token-123')
  })

  it.each([false, true])('connects to a bootstrap server with auth disabled=%s', async (authDisabled) => {
    // Given: A fresh server whose camera API is unavailable until setup is complete
    const user = userEvent.setup()
    const authTokenProvider = new InMemoryAuthTokenProvider()
    const serverBaseUrlProvider = new BrowserServerBaseUrlProvider('')
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async (url, options) => {
      if (String(url).endsWith('/health')) {
        return jsonResponse({ ...HEALTH_PAYLOAD, bootstrap_mode: true })
      }
      if (String(url).endsWith('/setup/status')) {
        if (!authDisabled && authorizationHeader(options) !== 'Bearer bootstrap-token') {
          return unauthorizedResponse()
        }
        return jsonResponse({
          state: 'fresh',
          has_cameras: false,
          pipeline_running: false,
          auth_configured: !authDisabled,
        })
      }
      return jsonResponse({ error_code: 'SETUP_REQUIRED' }, 503)
    })
    renderNativeSetup({ authTokenProvider, serverBaseUrlProvider })

    // When: Connecting to the fresh server with its configured authentication mode
    await user.type(screen.getByLabelText('Server URL'), 'https://fresh.example.com')
    await user.click(screen.getByRole('button', { name: 'Check server' }))
    await screen.findByText('Server reachable')
    if (!authDisabled) {
      await user.type(screen.getByLabelText('API token'), 'bootstrap-token')
    }
    await user.click(screen.getByRole('button', { name: 'Save and continue' }))

    // Then: Connection settings are saved so the app can reach server onboarding
    await screen.findByText('Live route')
    expect(serverBaseUrlProvider.getBaseUrlSync()).toBe('https://fresh.example.com')
    expect(authTokenProvider.getTokenSync()).toBe(authDisabled ? null : 'bootstrap-token')
    expect(fetchSpy.mock.calls.some(([url]) => String(url).endsWith('/cameras'))).toBe(false)
  })

  it('shows actionable validation errors for bad server URLs and rejected tokens', async () => {
    // Given: Setup is rendered with a protected server
    const user = userEvent.setup()
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(jsonResponse(HEALTH_PAYLOAD))
      .mockResolvedValueOnce(unauthorizedResponse())
      .mockResolvedValueOnce(unauthorizedResponse())
    renderNativeSetup()

    // When: User submits an unsupported URL and then a rejected token
    await user.type(screen.getByLabelText('Server URL'), 'homesec.local:8081')
    await user.click(screen.getByRole('button', { name: 'Check server' }))
    await screen.findByText('Only http:// and https:// server URLs are supported.')
    await user.clear(screen.getByLabelText('Server URL'))
    await user.type(screen.getByLabelText('Server URL'), 'https://homesec.example.com')
    await user.click(screen.getByRole('button', { name: 'Check server' }))
    await screen.findByText('Server reachable')
    await user.type(screen.getByLabelText('API token'), 'wrong-token')
    await user.click(screen.getByRole('button', { name: 'Save and continue' }))

    // Then: Invalid states do not persist settings or navigate away
    await screen.findByText('API token was rejected. Paste the HomeSec API token and try again.')
    expect(fetchSpy).toHaveBeenCalledTimes(3)
    expect(window.sessionStorage.getItem(BROWSER_SERVER_BASE_URL_STORAGE_KEY)).toBeNull()
    expect(window.sessionStorage.getItem(BROWSER_AUTH_TOKEN_STORAGE_KEY)).toBeNull()
    expect(screen.queryByText('Live route')).toBeNull()
  })

  it('clears stale plain HTTP warning when server URL changes', async () => {
    // Given: User validated a plain-HTTP server URL
    const user = userEvent.setup()
    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(jsonResponse(HEALTH_PAYLOAD))
      .mockResolvedValueOnce(unauthorizedResponse())
    renderNativeSetup()
    await user.type(screen.getByLabelText('Server URL'), 'http://192.168.1.10:8081')
    await user.click(screen.getByRole('button', { name: 'Check server' }))
    await screen.findByText('Plain HTTP is visible on the network. Prefer HTTPS or VPN for iOS access.')

    // When: The server URL field changes
    await user.clear(screen.getByLabelText('Server URL'))
    await user.type(screen.getByLabelText('Server URL'), 'https://homesec.example.com')

    // Then: Warning state from the previous validated URL is cleared
    expect(screen.queryByText('Plain HTTP is visible on the network. Prefer HTTPS or VPN for iOS access.')).toBeNull()
  })

  it('clears stale token input when the validated server URL changes', async () => {
    // Given: User validated a protected server and entered its API token
    const user = userEvent.setup()
    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(jsonResponse(HEALTH_PAYLOAD))
      .mockResolvedValueOnce(unauthorizedResponse())
    renderNativeSetup()
    await user.type(screen.getByLabelText('Server URL'), 'https://homesec.example.com')
    await user.click(screen.getByRole('button', { name: 'Check server' }))
    await screen.findByText('Server reachable')
    await user.type(screen.getByLabelText('API token'), 'token-for-first-server')

    // When: The server URL field changes before saving
    await user.clear(screen.getByLabelText('Server URL'))
    await user.type(screen.getByLabelText('Server URL'), 'https://homesec-new.example.com')

    // Then: The previous server token cannot be saved against the new server
    expect((screen.getByLabelText('API token') as HTMLInputElement).value).toBe('')
  })

  it('can save tokens through an in-memory provider without writing browser token storage', async () => {
    // Given: Native setup uses an in-memory token provider
    const user = userEvent.setup()
    const authTokenProvider = new InMemoryAuthTokenProvider()
    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(jsonResponse(HEALTH_PAYLOAD))
      .mockResolvedValueOnce(unauthorizedResponse())
      .mockResolvedValueOnce(jsonResponse(SETUP_PAYLOAD))
    renderNativeSetup({ authTokenProvider })

    // When: User validates and saves a protected server
    await user.type(screen.getByLabelText('Server URL'), 'https://homesec.example.com')
    await user.click(screen.getByRole('button', { name: 'Check server' }))
    await screen.findByText('Server reachable')
    await user.type(screen.getByLabelText('API token'), 'native-token')
    await user.click(screen.getByRole('button', { name: 'Save and continue' }))

    // Then: The token is available to API clients through the provider only
    await waitFor(() => {
      expect(screen.getByText('Live route')).toBeTruthy()
    })
    expect(await authTokenProvider.getToken()).toBe('native-token')
    expect(window.sessionStorage.getItem(BROWSER_AUTH_TOKEN_STORAGE_KEY)).toBeNull()
  })

  it('clears cached API data after saving a server URL', async () => {
    // Given: Cached data from a previous HomeSec server
    const user = userEvent.setup()
    const queryClient = createTestQueryClient()
    queryClient.setQueryData(['cameras'], [{ name: 'old-server-camera' }])
    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(jsonResponse(HEALTH_PAYLOAD))
      .mockResolvedValueOnce(unauthorizedResponse())
      .mockResolvedValueOnce(jsonResponse(SETUP_PAYLOAD))
    renderNativeSetup({}, queryClient)

    // When: User validates and saves a new server
    await user.type(screen.getByLabelText('Server URL'), 'https://homesec.example.com')
    await user.click(screen.getByRole('button', { name: 'Check server' }))
    await screen.findByText('Server reachable')
    await user.type(screen.getByLabelText('API token'), 'token-123')
    await user.click(screen.getByRole('button', { name: 'Save and continue' }))

    // Then: Server-agnostic React Query cache entries cannot leak across servers
    await waitFor(() => {
      expect(screen.getByText('Live route')).toBeTruthy()
    })
    expect(queryClient.getQueryData(['cameras'])).toBeUndefined()
  })

  it('returns to the requested route after native setup succeeds', async () => {
    // Given: Setup was opened by a guard for an event deep link
    const user = userEvent.setup()
    vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(jsonResponse(HEALTH_PAYLOAD))
      .mockResolvedValueOnce(unauthorizedResponse())
      .mockResolvedValueOnce(jsonResponse(SETUP_PAYLOAD))
    renderNativeSetup(
      {},
      createTestQueryClient(),
      { nativeSetupReturnTo: '/events/clip-42?camera=front' },
    )

    // When: User completes setup
    await user.type(screen.getByLabelText('Server URL'), 'https://homesec.example.com')
    await user.click(screen.getByRole('button', { name: 'Check server' }))
    await screen.findByText('Server reachable')
    await user.type(screen.getByLabelText('API token'), 'token-123')
    await user.click(screen.getByRole('button', { name: 'Save and continue' }))

    // Then: The original route intent wins over the default Live destination
    await waitFor(() => {
      expect(screen.getByText('Event route')).toBeTruthy()
    })
  })

  it('warns and allows continuing when auth-disabled mode is detectable', async () => {
    // Given: Setup status succeeds without an API token and an old token is stored
    const user = userEvent.setup()
    window.sessionStorage.setItem(BROWSER_AUTH_TOKEN_STORAGE_KEY, 'old-secret')
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(jsonResponse(HEALTH_PAYLOAD))
      .mockResolvedValueOnce(jsonResponse(SETUP_PAYLOAD))
    renderNativeSetup()

    // When: User checks the server and continues without a token
    await user.type(screen.getByLabelText('Server URL'), 'https://homesec.example.com')
    await user.click(screen.getByRole('button', { name: 'Check server' }))
    await screen.findByText(/accepted setup requests without an API token/)
    expect((screen.getByLabelText('API token') as HTMLInputElement).disabled).toBe(true)
    await user.click(screen.getByRole('button', { name: 'Save and continue' }))

    // Then: The server URL is saved, the old token is cleared, and no token validation is faked
    await waitFor(() => {
      expect(screen.getByText('Live route')).toBeTruthy()
    })
    expect(fetchSpy).toHaveBeenCalledTimes(2)
    expect(window.sessionStorage.getItem(BROWSER_SERVER_BASE_URL_STORAGE_KEY)).toBe(
      'https://homesec.example.com',
    )
    expect(window.sessionStorage.getItem(BROWSER_AUTH_TOKEN_STORAGE_KEY)).toBeNull()
  })
})
