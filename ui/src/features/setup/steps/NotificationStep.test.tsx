// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { NotificationStep } from './NotificationStep'

vi.mock('../../shared/TestConnectionButton', () => ({
  TestConnectionButton: ({
    request,
    result,
    onResult,
  }: {
    request: { backend?: string; config?: Record<string, unknown> }
    result: { success: boolean; message: string } | null
    onResult: (value: { success: boolean; message: string; latency_ms?: number }) => void
  }) => {
    const backend = String(request.backend ?? 'unknown')
    const host = String(request.config?.host ?? '')
    return (
      <section data-testid={`test-connection-${backend}`}>
        <p data-testid={`test-request-host-${backend}`}>{host}</p>
        <p>{result ? 'PASS' : 'NO_RESULT'}</p>
        <button
          type="button"
          onClick={() => {
            onResult({
              success: true,
              message: `${backend} connection succeeded`,
              latency_ms: 7,
            })
          }}
        >
          Trigger {backend} result
        </button>
      </section>
    )
  },
}))

describe('NotificationStep', () => {
  afterEach(() => {
    cleanup()
  })

  it('emits mqtt notifier and alert policy payload on save', async () => {
    // Given: Notification step starts with no notifier enabled
    const onComplete = vi.fn()
    const onUpdateData = vi.fn()
    const user = userEvent.setup()
    render(
      <NotificationStep
        initialData={null}
        onComplete={onComplete}
        onUpdateData={onUpdateData}
        onSkip={vi.fn()}
      />,
    )

    // When: Operator enables MQTT and saves step
    await user.click(screen.getByRole('checkbox', { name: 'MQTT' }))
    await user.click(screen.getByRole('button', { name: 'Save and continue' }))

    // Then: Step emits notifier list + alert policy baseline
    expect(onUpdateData).toHaveBeenCalledWith({
      notifiers: [
        {
          backend: 'mqtt',
          enabled: true,
          config: {
            host: 'localhost',
            port: 1883,
            topic_template: 'homecam/alerts/{camera_name}',
          },
        },
      ],
      alert_policy: {
        backend: 'default',
        enabled: true,
        config: {
          min_risk_level: 'high',
        },
      },
    })
    expect(onComplete).toHaveBeenCalledTimes(1)
  })

  it('supports multiple enabled notifier backends', async () => {
    // Given: Notification step with multiple backend options
    const onUpdateData = vi.fn()
    const user = userEvent.setup()
    render(
      <NotificationStep
        initialData={null}
        onComplete={vi.fn()}
        onUpdateData={onUpdateData}
        onSkip={vi.fn()}
      />,
    )

    // When: Operator enables MQTT and SendGrid then saves
    await user.click(screen.getByRole('checkbox', { name: 'MQTT' }))
    await user.click(screen.getByRole('checkbox', { name: 'SendGrid Email' }))
    await user.click(screen.getByRole('button', { name: 'Save and continue' }))

    // Then: Payload includes both notifier entries
    expect(onUpdateData).toHaveBeenCalled()
    expect(onUpdateData.mock.calls.at(-1)?.[0]).toMatchObject({
      notifiers: [
        expect.objectContaining({ backend: 'mqtt', enabled: true }),
        expect.objectContaining({ backend: 'sendgrid_email', enabled: true }),
      ],
    })
  })

  it('maps skip button to wizard skip callback', async () => {
    // Given: Notification step with skip callback
    const onSkip = vi.fn()
    const user = userEvent.setup()
    render(
      <NotificationStep
        initialData={null}
        onComplete={vi.fn()}
        onUpdateData={vi.fn()}
        onSkip={onSkip}
      />,
    )

    // When: Operator skips notification step
    await user.click(screen.getByRole('button', { name: 'Skip notification step' }))

    // Then: Step requests wizard skip transition
    expect(onSkip).toHaveBeenCalledTimes(1)
  })

  it('renders validation error when notifier config is invalid', async () => {
    // Given: Notification step with mqtt enabled and blank host value
    const onComplete = vi.fn()
    const onUpdateData = vi.fn()
    const user = userEvent.setup()
    render(
      <NotificationStep
        initialData={null}
        onComplete={onComplete}
        onUpdateData={onUpdateData}
        onSkip={vi.fn()}
      />,
    )
    await user.click(screen.getByRole('checkbox', { name: 'MQTT' }))
    await user.clear(screen.getByLabelText('MQTT host'))

    // When: Operator saves an invalid notifier config
    await user.click(screen.getByRole('button', { name: 'Save and continue' }))

    // Then: Validation error is shown and wizard completion is blocked
    expect(screen.getByText('MQTT host is required.')).toBeTruthy()
    expect(onUpdateData).not.toHaveBeenCalled()
    expect(onComplete).not.toHaveBeenCalled()
  })

  it('hydrates notifier and alert-policy fields from initial step data', () => {
    // Given: Wizard step draft with mqtt config and critical-only alert policy
    render(
      <NotificationStep
        initialData={{
          notifiers: [
            {
              backend: 'mqtt',
              enabled: true,
              config: {
                host: 'broker.internal',
                port: 1993,
                topic_template: 'alerts/{camera_name}',
              },
            },
          ],
          alert_policy: {
            backend: 'default',
            enabled: true,
            config: {
              min_risk_level: 'critical',
            },
          },
        }}
        onComplete={vi.fn()}
        onUpdateData={vi.fn()}
        onSkip={vi.fn()}
      />,
    )

    // When: Step renders with restored wizard data
    const mqttEnabled = screen.getByRole('checkbox', { name: 'MQTT' }) as HTMLInputElement
    const criticalRisk = screen.getByRole('checkbox', { name: 'critical' }) as HTMLInputElement
    const highRisk = screen.getByRole('checkbox', { name: 'high' }) as HTMLInputElement
    const mqttHost = screen.getByLabelText('MQTT host') as HTMLInputElement

    // Then: Form fields and risk selections match persisted draft payload
    expect(mqttEnabled.checked).toBe(true)
    expect(mqttHost.value).toBe('broker.internal')
    expect(criticalRisk.checked).toBe(true)
    expect(highRisk.checked).toBe(false)
  })

  it('clears test results when notifier config changes or backend is toggled', async () => {
    // Given: MQTT notifier enabled and a successful connection test result
    const user = userEvent.setup()
    render(
      <NotificationStep
        initialData={null}
        onComplete={vi.fn()}
        onUpdateData={vi.fn()}
        onSkip={vi.fn()}
      />,
    )
    await user.click(screen.getByRole('checkbox', { name: 'MQTT' }))
    await user.click(screen.getByRole('button', { name: 'Trigger mqtt result' }))

    // When: Operator updates mqtt host and then disables/re-enables mqtt
    const mqttResultBeforeEdit = screen.getByTestId('test-connection-mqtt')
    expect(within(mqttResultBeforeEdit).getByText('PASS')).toBeTruthy()
    fireEvent.change(screen.getByLabelText('MQTT host'), {
      target: { value: 'mqtt.changed.local' },
    })
    const mqttResultAfterEdit = screen.getByTestId('test-connection-mqtt')
    await user.click(screen.getByRole('checkbox', { name: 'MQTT' }))
    expect(screen.queryByTestId('test-connection-mqtt')).toBeNull()
    expect(screen.queryByLabelText('MQTT host')).toBeNull()
    await user.click(screen.getByRole('checkbox', { name: 'MQTT' }))

    // Then: Result resets after config edit and remains cleared after toggle cycle
    expect(within(mqttResultAfterEdit).getByText('NO_RESULT')).toBeTruthy()
    expect(screen.getByTestId('test-request-host-mqtt').textContent).toContain('mqtt.changed.local')
    expect(screen.getByLabelText('MQTT host')).toBeTruthy()
    expect(within(screen.getByTestId('test-connection-mqtt')).getByText('NO_RESULT')).toBeTruthy()
  })
})
