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
})
