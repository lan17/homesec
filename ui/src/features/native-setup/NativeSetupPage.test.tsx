// @vitest-environment happy-dom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'

import {
  BROWSER_AUTH_TOKEN_STORAGE_KEY,
  InMemoryAuthTokenProvider,
} from '../../api/tokenProvider'
import { BROWSER_SERVER_BASE_URL_STORAGE_KEY } from '../../api/serverBaseUrlProvider'
import { NativeSetupPage, type NativeSetupPageProps } from './NativeSetupPage'

const HEALTH_PAYLOAD = {
  status: 'healthy',
  pipeline: 'running',
  postgres: 'connected',
  cameras_online: 1,
  bootstrap_mode: false,
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

describe('NativeSetupPage', () => {
  beforeEach(() => {
    window.sessionStorage.clear()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    window.sessionStorage.clear()
  })

  it('validates server and token before saving settings and routing to Live', async () => {
    // Given: A reachable HTTP LAN server and an old stored token from another server
    const user = userEvent.setup()
    window.sessionStorage.setItem(BROWSER_AUTH_TOKEN_STORAGE_KEY, 'old-secret')
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(jsonResponse(HEALTH_PAYLOAD))
      .mockResolvedValueOnce(unauthorizedResponse())
      .mockResolvedValueOnce(jsonResponse([]))
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
    expect(fetchSpy.mock.calls[1]?.[0]).toBe('http://192.168.1.10:8081/api/v1/cameras')
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
      .mockResolvedValueOnce(jsonResponse([]))
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
      .mockResolvedValueOnce(jsonResponse([]))
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
      .mockResolvedValueOnce(jsonResponse([]))
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
    // Given: Camera list succeeds without an API token and an old token is stored
    const user = userEvent.setup()
    window.sessionStorage.setItem(BROWSER_AUTH_TOKEN_STORAGE_KEY, 'old-secret')
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(jsonResponse(HEALTH_PAYLOAD))
      .mockResolvedValueOnce(jsonResponse([]))
    renderNativeSetup()

    // When: User checks the server and continues without a token
    await user.type(screen.getByLabelText('Server URL'), 'https://homesec.example.com')
    await user.click(screen.getByRole('button', { name: 'Check server' }))
    await screen.findByText(/accepted camera requests without an API token/)
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
