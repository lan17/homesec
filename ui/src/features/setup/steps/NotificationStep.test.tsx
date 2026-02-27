// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { NotificationStep } from './NotificationStep'

vi.mock('../../shared/TestConnectionButton', () => ({
  TestConnectionButton: () => <div data-testid="test-connection-button">Test connection button</div>,
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
})
