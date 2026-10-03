// @vitest-environment happy-dom

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { NotificationSettingsPage } from './NotificationSettingsPage'

const clients: QueryClient[] = []

function response(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), { status, headers: { 'content-type': 'application/json' } })
}

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  clients.push(client)
  render(<QueryClientProvider client={client}><NotificationSettingsPage /></QueryClientProvider>)
}

function snapshot(version = 'version-one') {
  return {
    saved_config_version: version, active_config_version: 'version-one', apply_required: 'none',
    config: {
      notifiers: [
        { backend: 'mqtt', enabled: true, config: { host: 'one', qos: 2 } },
        { backend: 'mqtt', enabled: true, config: { host: 'two', retain: true } },
        { backend: 'custom-channel', enabled: false, config: { preserve: true } },
      ],
      alert_policy: { backend: 'default', enabled: true, config: { min_risk_level: 'medium', overrides: { front: { min_risk_level: 'critical' } } } },
    },
  }
}

describe('notification settings page', () => {
  afterEach(() => {
    cleanup()
    clients.splice(0).forEach((client) => client.clear())
    vi.restoreAllMocks()
  })

  it('edits the second duplicate notifier through its own labelled fields', async () => {
    // Given: Two saved MQTT instances and a disabled custom destination
    const saved = snapshot()
    const patches: unknown[] = []
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, options) => {
      if (options?.method === 'PATCH') {
        patches.push(JSON.parse(String(options.body)))
        const next = snapshot('version-two')
        next.config.notifiers[1].config.host = 'second-updated'
        return response({ ...next, apply_required: 'reload' })
      }
      return response(saved)
    })
    const user = userEvent.setup()
    renderPage()
    const hosts = await screen.findAllByLabelText('MQTT host')

    // When: Only the second host is changed and settings are saved
    await user.clear(hosts[1])
    await user.type(hosts[1], 'second-updated')
    await user.click(screen.getByRole('button', { name: 'Save notification settings' }))

    // Then: Distinct labels target the right instance and the merge patch preserves duplicates/advanced fields
    await waitFor(() => expect(patches).toEqual([{ expected_config_version: 'version-one', notifiers: [{ index: 1, config: { host: 'second-updated' } }] }]))
    expect(hosts[0].getAttribute('id')).not.toBe(hosts[1].getAttribute('id'))
    expect(screen.getByText('This integration is read-only here. It is disabled; edit its settings in YAML.')).toBeTruthy()
    await screen.findByText('Saved settings are waiting to be applied.')
  })

  it('retains an outdated draft after conflict and refresh until explicitly discarded', async () => {
    // Given: A configuration save will conflict with another client
    let saved = snapshot()
    const patches: Array<{ expected_config_version: string }> = []
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (_url, options) => {
      if (options?.method === 'PATCH') {
        patches.push(JSON.parse(String(options.body)))
        return response({ detail: 'Saved settings changed', error_code: 'CONFIG_VERSION_CONFLICT' }, 409)
      }
      return response(saved)
    })
    const user = userEvent.setup()
    renderPage()
    const hosts = await screen.findAllByLabelText('MQTT host')
    await user.clear(hosts[1])
    await user.type(hosts[1], 'my-draft')

    // When: Save conflicts and the operator refreshes the newer saved configuration
    await user.click(screen.getByRole('button', { name: 'Save notification settings' }))
    await screen.findByText(/Saved settings changed elsewhere/)
    saved = snapshot('version-two')
    saved.config.notifiers[1].config.host = 'remote-change'
    await user.click(screen.getByRole('button', { name: 'Refresh saved settings' }))
    await user.click(screen.getByRole('button', { name: 'Save notification settings' }))

    // Then: Draft and version remain paired; discarding reveals the refreshed saved value
    await waitFor(() => expect(patches).toHaveLength(2))
    expect(patches.map((patch) => patch.expected_config_version)).toEqual(['version-one', 'version-one'])
    expect((screen.getAllByLabelText('MQTT host')[1] as HTMLInputElement).value).toBe('my-draft')
    await user.click(screen.getByRole('button', { name: 'Discard unsaved changes' }))
    expect((screen.getAllByLabelText('MQTT host')[1] as HTMLInputElement).value).toBe('remote-change')
  })

  it('targets credentials by notifier index and preserves unrelated duplicate destinations', async () => {
    // Given: Two MQTT instances have separately configured passwords
    const saved = { ...snapshot(), credentials_editable: true, credentials: {
      'notifiers.0.config.auth.password_env': { configured: true, source: 'environment' },
      'notifiers.1.config.auth.password_env': { configured: true, source: 'environment' },
    } }
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(response(saved))
    const user = userEvent.setup()
    renderPage()
    const replace = await screen.findAllByRole('button', { name: 'Replace MQTT password' })
    // When: Replacing only the second instance's password
    await user.click(replace[1])
    await user.type(screen.getByLabelText('Replace MQTT password'), 'second-private-password')
    expect(screen.getAllByRole('button', { name: 'Check connection' })).toHaveLength(1)
    await user.click(screen.getByRole('button', { name: 'Save notification settings' }))
    // Then: Only its credential slot is submitted and no notifier configuration or first password is rewritten
    await waitFor(() => expect(fetch.mock.calls.some(([, request]) => request?.method === 'PATCH')).toBe(true))
    const patch = fetch.mock.calls.find(([, request]) => request?.method === 'PATCH')?.[1]?.body
    expect(JSON.parse(String(patch))).toEqual({ expected_config_version: 'version-one',
      credentials: { 'notifiers.1.config.auth.password_env': 'second-private-password' } })
  })

  it('saves an email API key without testing or sending a sample notification', async () => {
    // Given: A saved email integration has no configured API key
    const saved = { ...snapshot(), credentials_editable: true,
      credentials: { 'notifiers.0.config.api_key_env': { configured: false, source: 'environment' } },
      config: { ...snapshot().config, notifiers: [{ backend: 'sendgrid_email', enabled: true,
        config: { from_email: 'alerts@test.local', to_emails: ['owner@test.local'], api_key_env: 'SENDGRID_API_KEY' } }] } }
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(response(saved))
    const user = userEvent.setup()
    renderPage()
    // When: Entering the email API key and saving
    await user.type(await screen.findByLabelText('SendGrid API key'), 'private-email-key')
    expect(screen.queryByRole('button', { name: 'Check connection' })).toBeNull()
    await user.click(screen.getByRole('button', { name: 'Save notification settings' }))
    // Then: The key is written only to the credential slot; connectivity and sending remain separate
    await waitFor(() => expect(fetch.mock.calls.some(([, request]) => request?.method === 'PATCH')).toBe(true))
    const patch = fetch.mock.calls.find(([, request]) => request?.method === 'PATCH')?.[1]?.body
    expect(JSON.parse(String(patch))).toEqual({ expected_config_version: 'version-one',
      credentials: { 'notifiers.0.config.api_key_env': 'private-email-key' } })
    expect(fetch.mock.calls.some(([url]) => String(url).includes('/setup/test-connection'))).toBe(false)
  })

  it.each(['mqtt', 'sendgrid_email'])('rejects a raw Advanced reference after the %s notifier is disabled', async (backend) => {
    // Given: An enabled notifier exposes a credential environment-reference field
    const saved = { ...snapshot(), config: { ...snapshot().config, notifiers: [{ backend, enabled: true,
      config: backend === 'mqtt' ? { host: 'mqtt.local', auth: { password_env: 'MQTT_PASSWORD' } }
        : { from_email: 'alerts@test.local', to_emails: ['owner@test.local'], api_key_env: 'SENDGRID_API_KEY' },
    }] } }
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(response(saved))
    renderPage()
    const user = userEvent.setup()
    const label = backend === 'mqtt' ? 'MQTT password env var' : 'SendGrid API key env var'
    const input = await screen.findByLabelText(label)

    // When: A raw reference is entered, then the notifier is disabled before saving
    await user.clear(input)
    await user.type(input, 'raw-private-key/value')
    await user.click(screen.getByLabelText('Enable notifier 1'))
    await user.click(screen.getByRole('button', { name: 'Save notification settings' }))

    // Then: Disabled state cannot persist or cache the raw reference
    await screen.findByText('Credentials must reference environment variable names.')
    expect(fetch.mock.calls.every(([, request]) => request?.method === 'GET')).toBe(true)
    expect(clients[0].getMutationCache().getAll()).toHaveLength(0)
    expect(JSON.stringify(clients[0].getQueryData(['config']))).not.toContain('raw-private-key/value')
  })

})
