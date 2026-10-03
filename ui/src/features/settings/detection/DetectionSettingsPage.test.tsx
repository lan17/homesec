// @vitest-environment happy-dom

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { DetectionSettingsPage } from './DetectionSettingsPage'

const clients: QueryClient[] = []

function response(payload: unknown): Response {
  return new Response(JSON.stringify(payload), { headers: { 'content-type': 'application/json' } })
}

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  clients.push(client)
  render(<QueryClientProvider client={client}><DetectionSettingsPage /></QueryClientProvider>)
}

function snapshot() {
  return {
    saved_config_version: 'saved-one', active_config_version: 'saved-one', apply_required: 'none',
    config: {
      filter: { backend: 'yolo', config: { classes: ['person'], min_confidence: 0.05, model_path: 'my-model.pt', sample_fps: 3 } },
      vlm: { backend: 'openai', run_mode: 'always', trigger_classes: ['person'], preprocessing: { quality: 70 }, config: { api_key_env: 'MY_AI_KEY', model: 'my-model', base_url: 'https://ai.test', max_tokens: 123 } },
    },
  }
}

describe('detection settings page', () => {
  afterEach(() => {
    cleanup()
    clients.splice(0).forEach((client) => client.clear())
    vi.restoreAllMocks()
  })

  it('saves explicit analysis disable without replacing its configuration', async () => {
    // Given: Existing analysis uses always mode and advanced settings
    const saved = snapshot()
    const patches: unknown[] = []
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, options) => {
      if (options?.method === 'PATCH') {
        patches.push(JSON.parse(String(options.body)))
        return response({ ...saved, saved_config_version: 'saved-two', apply_required: 'reload', config: { ...saved.config, vlm: { ...saved.config.vlm, run_mode: 'never' } } })
      }
      return response(saved)
    })
    const user = userEvent.setup()
    renderPage()
    await screen.findByLabelText('Run mode')

    // When: The operator disables analysis and saves
    await user.click(screen.getByLabelText('Enable AI scene analysis (VLM)'))
    await user.click(screen.getByRole('button', { name: 'Save detection settings' }))

    // Then: Only run_mode changes and activation remains a separate operation
    await waitFor(() => expect(patches).toEqual([{ expected_config_version: 'saved-one', vlm: { run_mode: 'never' } }]))
    await screen.findByText('Saved settings are waiting to be applied.')
    expect(screen.queryByLabelText('Model')).toBeNull()
  })

  it('offers supported classes and the full backend confidence range', async () => {
    // Given: A configured confidence below 0.1 and an existing AI mode
    const saved = snapshot()
    const patches: unknown[] = []
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, options) => {
      if (options?.method === 'PATCH') {
        patches.push(JSON.parse(String(options.body)))
        return response({ ...saved, saved_config_version: 'saved-two', apply_required: 'reload' })
      }
      return response(saved)
    })
    const user = userEvent.setup()
    renderPage()
    const slider = await screen.findByLabelText('Confidence threshold: 0.05')

    // When: Confidence becomes zero and a supported bird class is added
    fireEvent.change(slider, { target: { value: '0' } })
    await user.selectOptions(screen.getByLabelText('Detection class to add'), 'bird')
    await user.click(screen.getByRole('button', { name: 'Add class' }))
    await user.click(screen.getByRole('button', { name: 'Save detection settings' }))

    // Then: The patch only changes guided filter values, leaving always mode and advanced fields alone
    await waitFor(() => expect(patches).toEqual([{ expected_config_version: 'saved-one', filter: { config: { classes: ['person', 'bird'], min_confidence: 0 } } }]))
    expect(slider.getAttribute('min')).toBe('0')
    expect(screen.queryByRole('option', { name: 'car' })).toBeNull()
  })

  it('saves an AI key separately from model settings and blocks readiness checks until Apply', async () => {
    // Given: The existing OpenAI-compatible analyzer has a configured environment credential
    const saved = { ...snapshot(), credentials_editable: true,
      credentials: { 'vlm.config.api_key_env': { configured: true, source: 'environment' } } }
    const fetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, request) => request?.method === 'PATCH'
      ? response({ ...saved, saved_config_version: 'saved-two', apply_required: 'restart',
        credentials: { 'vlm.config.api_key_env': { configured: true, source: 'managed' } } }) : response(saved))
    const user = userEvent.setup()
    renderPage()
    await user.click(await screen.findByRole('button', { name: 'Replace AI API key' }))
    // When: Entering a replacement API key, changing the model, and saving
    await user.type(screen.getByLabelText('Replace AI API key'), 'private-ai-key')
    const model = screen.getByLabelText('Model')
    await user.clear(model)
    await user.type(model, 'new-model')
    expect(screen.queryByRole('button', { name: 'Check AI readiness' })).toBeNull()
    await user.click(screen.getByRole('button', { name: 'Save detection settings' }))
    // Then: The credential uses its write-only patch slot, advanced analyzer settings remain untouched, and restart is explicit
    await screen.findByRole('button', { name: 'Apply and restart' })
    const patch = fetch.mock.calls.find(([, request]) => request?.method === 'PATCH')?.[1]?.body
    expect(JSON.parse(String(patch))).toEqual({ expected_config_version: 'saved-one',
      vlm: { config: { model: 'new-model' } }, credentials: { 'vlm.config.api_key_env': 'private-ai-key' } })
    expect((screen.getByLabelText('Replace AI API key') as HTMLInputElement).value).toBe('')
    expect(screen.queryByRole('button', { name: 'Check AI readiness' })).toBeNull()
    expect(fetch.mock.calls.some(([url]) => String(url).includes('/setup/test-connection'))).toBe(false)
  })
})
