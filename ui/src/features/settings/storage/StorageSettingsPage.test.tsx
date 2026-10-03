// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'

import type { ConfigResponse } from '../../../api/generated/types'
import { StorageSettingsPage } from './StorageSettingsPage'

function config(root = '/saved', version = 'v1'): ConfigResponse {
  return {
    config: { storage: { backend: 'local', config: { root, advanced_option: 'preserve' } } },
    saved_config_version: version, active_config_version: version, apply_required: 'none',
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
})
