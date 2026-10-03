// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'

import type { ConfigResponse } from '../../../api/generated/types'
import { StorageSettingsPage } from './StorageSettingsPage'

function config(root = '/saved', version = 'v1'): ConfigResponse {
  return {
    config: { storage: { backend: 'local', config: { root, advanced_option: 'preserve' } } },
    saved_config_version: version, active_config_version: version, apply_required: 'none',
    credentials: {}, credentials_editable: true,
  }
}

function response(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), { status, headers: { 'content-type': 'application/json' } })
}

function renderPage() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
  render(<QueryClientProvider client={queryClient}>
    <MemoryRouter><StorageSettingsPage /></MemoryRouter>
  </QueryClientProvider>)
  return queryClient
}

describe('StorageSettingsPage', () => {
  afterEach(() => { cleanup(); vi.restoreAllMocks() })

  it('saves only changed fields and leaves activation as a deliberate separate action', async () => {
    // Given: The saved backend includes an advanced option that the guided form does not expose
    const initial = config()
    const saved = { ...config('/new', 'v2'), active_config_version: 'v1', apply_required: 'restart' }
    const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, request) =>
      response(request?.method === 'PATCH' ? saved : initial))
    renderPage()
    const user = userEvent.setup()
    const root = await screen.findByLabelText('Storage root directory')
    // When: Updating the configured root and saving
    await user.clear(root)
    await user.type(root, '/new')
    await user.click(screen.getByRole('button', { name: 'Save storage settings' }))
    await screen.findByRole('button', { name: 'Apply and restart' })
    // Then: PATCH carries the captured precondition and minimal diff, and no apply request is sent
    const calls = fetch.mock.calls.filter(([, request]) => request?.method === 'PATCH')
    expect(calls).toHaveLength(1)
    expect(JSON.parse(String(calls[0]?.[1]?.body))).toEqual({
      expected_config_version: 'v1', storage: { config: { root: '/new' } },
    })
    expect(fetch.mock.calls.some(([url]) => String(url).endsWith('/config/apply'))).toBe(false)
    expect(screen.getByText('Saved settings are waiting to be applied.')).toBeTruthy()
    expect((screen.getByRole('button', { name: 'Save storage settings' }) as HTMLButtonElement).disabled).toBe(true)
    expect(screen.queryByRole('button', { name: 'Dropbox' })).toBeNull()
  })

  it('retains an outdated draft through a conflict and refresh until explicitly discarded', async () => {
    // Given: A concurrent writer replaces the saved configuration after the draft is created
    let latest = config()
    const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, request) =>
      request?.method === 'PATCH'
        ? response({ detail: 'Saved settings changed', error_code: 'CONFIG_VERSION_CONFLICT' }, 409)
        : response(latest))
    renderPage()
    const user = userEvent.setup()
    const root = await screen.findByLabelText('Storage root directory')
    await user.clear(root)
    await user.type(root, '/my-draft')
    latest = config('/other-writer', 'v2')
    // When: Saving conflicts, then the latest saved settings are refreshed
    await user.click(screen.getByRole('button', { name: 'Save storage settings' }))
    await screen.findByText(/Your unsaved draft is retained/)
    await user.click(screen.getByRole('button', { name: 'Refresh saved settings' }))
    await waitFor(() => expect(fetch.mock.calls.filter(([, request]) => request?.method === 'GET')).toHaveLength(2))
    // Then: Draft values remain visible, and discarding deliberately loads the concurrent writer's values
    expect((root as HTMLInputElement).value).toBe('/my-draft')
    const patch = fetch.mock.calls.find(([, request]) => request?.method === 'PATCH')?.[1]?.body
    expect(JSON.parse(String(patch)).expected_config_version).toBe('v1')
    await user.click(screen.getByRole('button', { name: 'Discard draft' }))
    expect((root as HTMLInputElement).value).toBe('/other-writer')
  })

  it('keeps a failed save editable without falsely reporting it as saved', async () => {
    // Given: Saving fails at the HTTP boundary
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, request) =>
      request?.method === 'PATCH' ? response({ detail: 'Disk is read-only' }, 500) : response(config()))
    renderPage()
    const user = userEvent.setup()
    const root = await screen.findByLabelText('Storage root directory')
    // When: Editing and attempting to save
    await user.clear(root)
    await user.type(root, '/draft')
    await user.click(screen.getByRole('button', { name: 'Save storage settings' }))
    await waitFor(() => expect(screen.getAllByText('Disk is read-only').length).toBeGreaterThan(0))
    // Then: The draft remains retryable and the server's active status remains separate
    expect((root as HTMLInputElement).value).toBe('/draft')
    expect((screen.getByRole('button', { name: 'Save storage settings' }) as HTMLButtonElement).disabled).toBe(false)
    expect(screen.queryByText(/Settings saved. Apply/)).toBeNull()
  })

  it('shows persisted activation requirements after reopening the page', async () => {
    // Given: A prior session saved changes but never restarted HomeSec
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(response({
      ...config('/saved-new', 'v2'), active_config_version: 'v1', apply_required: 'restart',
    }))
    // When: Opening the settings page in a fresh query cache
    renderPage()
    await screen.findByRole('button', { name: 'Apply and restart' })
    // Then: Pending activation comes from the server and includes manual-process recovery guidance
    expect(screen.getByText('Saved settings are waiting to be applied.')).toBeTruthy()
    expect(screen.getByText(/a manually launched process must be started again/)).toBeTruthy()
    expect((screen.getByLabelText('Storage root directory') as HTMLInputElement).value).toBe('/saved-new')
  })

  it('shows busy activation without revision-conflict instructions', async () => {
    // Given: Saved changes require activation but another application operation is busy
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, request) =>
      request?.method === 'POST'
        ? response({ detail: 'A runtime reload is already in progress', error_code: 'RELOAD_IN_PROGRESS' }, 409)
        : response({ ...config('/saved-new', 'v2'), active_config_version: 'v1', apply_required: 'restart' }))
    renderPage()
    const user = userEvent.setup()
    // When: Applying the saved version is refused because the runtime is busy
    await user.click(await screen.findByRole('button', { name: 'Apply and restart' }))
    await screen.findByText(/A runtime reload is already in progress/)
    // Then: Busy detail and pending settings remain visible without suggesting a concurrent config edit
    expect(screen.getByText('Saved settings are waiting to be applied.')).toBeTruthy()
    expect(screen.getByText('Settings remain saved. Activation has not been confirmed.')).toBeTruthy()
    expect(screen.queryByText(/Saved settings changed elsewhere/)).toBeNull()
  })

  it('preserves unsupported storage backends as read-only settings', async () => {
    // Given: A deployment uses a plugin not represented by this guided editor
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(response({
      ...config(), config: { storage: { backend: 'custom', config: { opaque: 'keep' } } },
    }))
    // When: Opening its storage settings
    renderPage()
    await screen.findByText(/not supported by the editor/)
    // Then: No editable configuration or save action is offered and no mutation occurs
    expect(screen.queryByRole('button', { name: 'Save storage settings' })).toBeNull()
    expect(screen.queryByRole('textbox')).toBeNull()
    expect(fetch.mock.calls.every(([, request]) => request?.method === 'GET')).toBe(true)
  })

  it('preserves inline credentials without probing their redacted placeholders', async () => {
    // Given: Saved Dropbox settings use an inline credential hidden by the API
    const initial = { ...config(), config: { storage: {
      backend: 'dropbox', config: { root: '/saved', token: '***redacted***', timeout: 45 },
    } } }
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(response(initial))
    renderPage()
    const user = userEvent.setup()
    const root = await screen.findByLabelText('Dropbox root path')
    // When: Editing a public field and saving
    await user.clear(root)
    await user.type(root, '/new')
    await user.click(screen.getByRole('button', { name: 'Save storage settings' }))
    await waitFor(() => expect(fetch.mock.calls.some(([, request]) => request?.method === 'PATCH')).toBe(true))
    // Then: Credentials are neither exposed, rewritten, nor used as literal probe credentials
    expect(screen.queryByRole('button', { name: 'Check storage connection' })).toBeNull()
    expect(screen.getByText(/Connectivity checks are unavailable/)).toBeTruthy()
    expect((screen.getByLabelText('Dropbox token env var') as HTMLInputElement).value).toBe('DROPBOX_TOKEN')
    const patch = fetch.mock.calls.find(([, request]) => request?.method === 'PATCH')?.[1]?.body
    expect(JSON.parse(String(patch))).toEqual({
      expected_config_version: 'v1', storage: { config: { root: '/new' } },
    })
    expect(fetch.mock.calls.some(([url]) => String(url).includes('/setup/test-connection'))).toBe(false)
  })

  it('saves a write-only token without caching it or applying it automatically', async () => {
    // Given: Authenticated Dropbox settings currently use an environment token
    const initial = { ...config(), credentials: { 'storage.config.token_env': { configured: true, source: 'environment' } },
      config: { storage: { backend: 'dropbox', config: { root: '/saved', token_env: 'DROPBOX_TOKEN' } } } }
    const saved = { ...initial, saved_config_version: 'v2', apply_required: 'restart',
      credentials: { 'storage.config.token_env': { configured: true, source: 'managed' } },
      config: { storage: { backend: 'dropbox', config: { root: '/saved', token_env: 'HOMESEC_MANAGED_token_test' } } } }
    let completeSave: ((value: Response) => void) | undefined
    const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, request) => request?.method === 'PATCH'
      ? new Promise<Response>((resolve) => { completeSave = resolve }) : response(initial))
    const client = renderPage()
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: 'Replace Dropbox token' }))
    const input = screen.getByLabelText('Replace Dropbox token')
    // When: Replacing the token and saving without pressing Apply
    await user.type(input, 'private-token-value')
    expect(screen.queryByRole('button', { name: 'Check storage connection' })).toBeNull()
    await user.click(screen.getByRole('button', { name: 'Save storage settings' }))
    await waitFor(() => expect(completeSave).toBeTruthy())
    // Then: The authenticated PATCH is minimal, both React Query caches exclude the secret, and Apply remains separate
    const patch = fetch.mock.calls.find(([, request]) => request?.method === 'PATCH')?.[1]
    expect(JSON.parse(String(patch?.body))).toEqual({ expected_config_version: 'v1', credentials: { 'storage.config.token_env': 'private-token-value' } })
    expect(client.getMutationCache().getAll()).toHaveLength(0)
    expect(JSON.stringify(client.getQueryCache().getAll().map((query) => query.state.data))).not.toContain('private-token-value')
    completeSave?.(response(saved))
    await screen.findByRole('button', { name: 'Apply and restart' })
    expect((screen.getByLabelText('Replace Dropbox token') as HTMLInputElement).value).toBe('')
    expect(document.body.textContent).not.toContain('private-token-value')
    expect(screen.queryByRole('button', { name: 'Check storage connection' })).toBeNull()
    expect(fetch.mock.calls.some(([url]) => String(url).endsWith('/config/apply'))).toBe(false)
  })

  it('saves an explicit credential clear while leaving refresh authentication intact', async () => {
    // Given: Dropbox has both access-token and refresh-token credential slots
    const initial = { ...config(), credentials: {
      'storage.config.token_env': { configured: true, source: 'managed' },
      'storage.config.refresh_token_env': { configured: true, source: 'environment' },
    }, config: { storage: { backend: 'dropbox', config: { root: '/saved', token_env: 'HOMESEC_MANAGED_token_test',
      app_key_env: 'CUSTOM_APP', app_secret_env: 'CUSTOM_SECRET', refresh_token_env: 'CUSTOM_REFRESH' } } } }
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(response(initial))
    renderPage()
    const user = userEvent.setup()
    // When: Clearing only the access token so Dropbox may use its existing refresh credentials
    await user.click(await screen.findByRole('button', { name: 'Clear Dropbox token' }))
    expect(screen.getByText('Will stop being used after Save and Apply.')).toBeTruthy()
    await user.click(screen.getByRole('button', { name: 'Save storage settings' }))
    // Then: Clear is a null write; no refresh credentials or configuration values are rewritten
    await waitFor(() => expect(fetch.mock.calls.some(([, request]) => request?.method === 'PATCH')).toBe(true))
    const patch = fetch.mock.calls.find(([, request]) => request?.method === 'PATCH')?.[1]?.body
    expect(JSON.parse(String(patch))).toEqual({ expected_config_version: 'v1', credentials: { 'storage.config.token_env': null } })
  })

  it('restores an external environment reference without sending a secret draft', async () => {
    // Given: A managed token can be replaced by an operator-provided environment variable
    const initial = { ...config(), credentials: { 'storage.config.token_env': { configured: true, source: 'managed' } },
      config: { storage: { backend: 'dropbox', config: { root: '/saved', token_env: 'HOMESEC_MANAGED_token_test' } } } }
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(response(initial))
    renderPage()
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: 'Replace Dropbox token' }))
    await user.type(screen.getByLabelText('Replace Dropbox token'), 'discard-this-secret')
    // When: Editing the Advanced reference instead of saving the replacement
    const env = screen.getByLabelText('Dropbox token env var')
    await user.clear(env)
    await user.type(env, 'EXTERNAL_DROPBOX_TOKEN')
    await user.click(screen.getByRole('button', { name: 'Save storage settings' }))
    // Then: The secret draft is discarded and only a config reference is submitted
    await waitFor(() => expect(fetch.mock.calls.some(([, request]) => request?.method === 'PATCH')).toBe(true))
    const patch = fetch.mock.calls.find(([, request]) => request?.method === 'PATCH')?.[1]?.body
    expect(JSON.parse(String(patch))).toEqual({ expected_config_version: 'v1', storage: { config: { token_env: 'EXTERNAL_DROPBOX_TOKEN' } } })
    expect(String(patch)).not.toContain('discard-this-secret')
  })

  it('requires server authentication for secret entry while public settings remain editable', async () => {
    // Given: API authentication is disabled on this HomeSec host
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(response({ ...config(), credentials_editable: false,
      config: { storage: { backend: 'dropbox', config: { root: '/saved', token_env: 'DROPBOX_TOKEN' } } } }))
    // When: Opening Dropbox settings
    renderPage()
    await screen.findByLabelText('Dropbox root path')
    // Then: The UI explains the requirement and provides no secret input or Clear action
    expect(screen.getAllByText(/Enable API authentication/).length).toBeGreaterThan(0)
    expect(document.querySelector('input[type="password"]')).toBeNull()
    expect(screen.queryByRole('button', { name: 'Clear Dropbox token' })).toBeNull()
    expect(screen.getByLabelText('Dropbox token env var')).toBeTruthy()
  })

  it('retains a secret draft and its original version through conflict and refresh until discard', async () => {
    // Given: Another writer saves settings while this credential replacement is being edited
    const initial = { ...config(), credentials: { 'storage.config.token_env': { configured: true, source: 'environment' } },
      config: { storage: { backend: 'dropbox', config: { root: '/saved', token_env: 'DROPBOX_TOKEN' } } } }
    let latest = initial
    const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, request) => request?.method === 'PATCH'
      ? response({ detail: 'Saved settings changed', error_code: 'CONFIG_VERSION_CONFLICT' }, 409) : response(latest))
    const client = renderPage()
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: 'Replace Dropbox token' }))
    const input = screen.getByLabelText('Replace Dropbox token')
    await user.type(input, 'retain-private-draft')
    latest = { ...initial, saved_config_version: 'v2' }
    // When: Save conflicts and the current saved settings are refreshed
    await user.click(screen.getByRole('button', { name: 'Save storage settings' }))
    await screen.findByText(/Your unsaved draft is retained/)
    await user.click(screen.getByRole('button', { name: 'Refresh saved settings' }))
    await waitFor(() => expect(fetch.mock.calls.filter(([, request]) => request?.method === 'GET')).toHaveLength(2))
    // Then: The replacement stays local with its old precondition; discard clears it deliberately
    expect((input as HTMLInputElement).value).toBe('retain-private-draft')
    const patch = fetch.mock.calls.find(([, request]) => request?.method === 'PATCH')?.[1]?.body
    expect(JSON.parse(String(patch)).expected_config_version).toBe('v1')
    expect(JSON.stringify(client.getQueryCache().getAll().map((query) => query.state.data))).not.toContain('retain-private-draft')
    expect(client.getMutationCache().getAll()).toHaveLength(0)
    await user.click(screen.getByRole('button', { name: 'Discard draft' }))
    expect((screen.getByLabelText('Replace Dropbox token') as HTMLInputElement).value).toBe('')
  })

  it.each([
    ['Dropbox app key', 'app_key_env'],
    ['Dropbox app secret', 'app_secret_env'],
    ['Dropbox refresh token', 'refresh_token_env'],
  ])('rejects raw values in the Advanced %s environment-reference field', async (label, key) => {
    // Given: Dropbox settings expose a writable Advanced credential reference
    const initial = { ...config(), config: { storage: { backend: 'dropbox', config: {
      root: '/saved', token_env: 'DROPBOX_TOKEN', [key]: 'EXTERNAL_KEY',
    } } } }
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(response(initial))
    const client = renderPage()
    const user = userEvent.setup()
    const input = await screen.findByLabelText(`${label} env var`)

    // When: A user pastes a raw credential into Advanced and attempts both Check and Save
    await user.clear(input)
    await user.type(input, 'raw-private-secret/value')
    await user.click(screen.getByRole('button', { name: 'Check storage connection' }))
    await screen.findByText('Credentials must reference environment variable names.')
    await user.click(screen.getByRole('button', { name: 'Save storage settings' }))

    // Then: Validation is safe and no unredacted value reaches YAML or either query cache
    await screen.findAllByText('Credentials must reference environment variable names.')
    expect(fetch.mock.calls.every(([, request]) => request?.method === 'GET')).toBe(true)
    expect(JSON.stringify(client.getQueryData(['config']))).not.toContain('raw-private-secret/value')
    expect(client.getMutationCache().getAll()).toHaveLength(0)
  })

  it('preserves unchanged external reference settings when only the Dropbox root changes', async () => {
    // Given: Existing deployment-specific env references need no rewrite or extra validation
    const initial = { ...config(), config: { storage: { backend: 'dropbox', config: {
      root: '/saved', token_env: 'EXTERNAL-TOKEN', app_key_env: 'EXTERNAL-APP',
      app_secret_env: 'EXTERNAL-SECRET', refresh_token_env: 'EXTERNAL-REFRESH',
    } } } }
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(response(initial))
    renderPage()
    const user = userEvent.setup()
    const input = await screen.findByLabelText('Dropbox root path')

    // When: Only the root changes
    await user.clear(input)
    await user.type(input, '/new-root')
    await user.click(screen.getByRole('button', { name: 'Save storage settings' }))

    // Then: Only that field is sent, preserving every existing external reference
    await waitFor(() => expect(fetch.mock.calls.some(([, request]) => request?.method === 'PATCH')).toBe(true))
    expect(JSON.parse(String(fetch.mock.calls.find(([, request]) => request?.method === 'PATCH')?.[1]?.body))).toEqual({
      expected_config_version: 'v1', storage: { config: { root: '/new-root' } },
    })
  })

  it.each(['success', 'error'])('discards a stale probe %s after editable inputs change and return', async (outcome) => {
    // Given: A connection probe is pending while the storage form remains editable
    let finish: ((value: Response) => void) | undefined
    const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, request) => {
      if (request?.method === 'POST') { return new Promise<Response>((resolve) => { finish = resolve }) }
      return response(config())
    })
    renderPage()
    const user = userEvent.setup()
    const input = await screen.findByLabelText('Storage root directory')
    await user.click(screen.getByRole('button', { name: 'Check storage connection' }))
    await waitFor(() => expect(finish).toBeTruthy())

    // When: The inputs change A to B to A before the old request completes despite cancellation
    expect((input as HTMLInputElement).disabled).toBe(false)
    await user.clear(input)
    await user.type(input, '/different')
    await user.clear(input)
    await user.type(input, '/saved')
    await act(async () => {
      finish?.(outcome === 'success' ? response({ success: true, message: 'Old probe success', latency_ms: 1 })
        : response({ detail: 'Old probe error' }, 500))
    })
    await waitFor(() => expect(screen.getByRole('button', { name: 'Check storage connection' })).toBeTruthy())

    // Then: The cancelled request cannot publish success or error for the untested draft
    expect(fetch.mock.calls.find(([, request]) => request?.method === 'POST')?.[1]?.signal?.aborted).toBe(true)
    expect(screen.queryByText('PASS')).toBeNull()
    expect(screen.queryByText('Old probe success')).toBeNull()
    expect(screen.queryByText('Old probe error')).toBeNull()
  })

  it('discards a pending probe when a credential draft hides the probe component', async () => {
    // Given: An external Dropbox token permits a probe before a replacement draft is entered
    const initial = { ...config(), credentials: { 'storage.config.token_env': { configured: true, source: 'environment' } },
      config: { storage: { backend: 'dropbox', config: { root: '/saved', token_env: 'DROPBOX_TOKEN' } } } }
    let finish: ((value: Response) => void) | undefined
    const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, request) => request?.method === 'POST'
      ? new Promise<Response>((resolve) => { finish = resolve }) : response(initial))
    renderPage()
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: 'Check storage connection' }))
    await waitFor(() => expect(finish).toBeTruthy())

    // When: Entering a secret replacement unmounts the pending probe, then its old result arrives
    await user.click(screen.getByRole('button', { name: 'Replace Dropbox token' }))
    await user.type(screen.getByLabelText('Replace Dropbox token'), 'private-new-token')
    await act(async () => { finish?.(response({ success: true, message: 'Old token works', latency_ms: 1 })) })
    await user.click(screen.getByRole('button', { name: 'Keep saved Dropbox token' }))

    // Then: Neither the old success nor a PASS badge is restored when the component remounts
    expect(fetch.mock.calls.find(([, request]) => request?.method === 'POST')?.[1]?.signal?.aborted).toBe(true)
    expect(screen.queryByText('PASS')).toBeNull()
    expect(screen.queryByText('Old token works')).toBeNull()
  })


  it.each([true, false])('invalidates a completed connection result when Refresh replaces tested settings (success=%s)', async (success) => {
    // Given: A completed provider test belongs to the current saved storage root
    let latest = config()
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, request) =>
      response(request?.method === 'POST' ? { success, message: 'Probe was for the old root', latency_ms: 1 } : latest))
    renderPage()
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: 'Check storage connection' }))
    await screen.findByText(success ? 'PASS' : 'FAIL')

    // When: Refresh loads a concurrent writer's new saved configuration without an unsaved draft
    latest = config('/new-root', 'v2')
    await user.click(screen.getByRole('button', { name: 'Refresh saved settings' }))
    await waitFor(() => expect((screen.getByLabelText('Storage root directory') as HTMLInputElement).value).toBe('/new-root'))

    // Then: The old completed success/failure is not attributed to the new provider inputs
    expect(screen.queryByText('PASS')).toBeNull()
    expect(screen.queryByText('FAIL')).toBeNull()
    expect(screen.queryByText('Probe was for the old root')).toBeNull()
  })

  it.each(['/absolute', '../escape', 'nested/../escape'])('rejects changed invalid storage destination %s before Save', async (destination) => {
    // Given: Storage destinations are provider-relative subdirectories rather than camera recording paths
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(response(config()))
    renderPage()
    const user = userEvent.setup()
    const input = await screen.findByLabelText('Clip destination directory')

    // When: Editing a destination to an absolute or parent-traversing path
    await user.clear(input)
    await user.type(input, destination)
    await user.click(screen.getByRole('button', { name: 'Save storage settings' }))

    // Then: The UI describes the storage root relationship and refuses an unusable upload path
    await screen.findByText("Storage destination directories must be relative without '..' segments.")
    expect(screen.getByText(/relative to the configured storage root/)).toBeTruthy()
    expect(fetch.mock.calls.every(([, request]) => request?.method === 'GET')).toBe(true)
    expect(screen.queryByText('Local working paths')).toBeNull()
  })

})
