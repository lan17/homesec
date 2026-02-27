// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'

import { NOTIFIER_BACKENDS } from './backends'
import { NotifierConfigForm } from './NotifierConfigForm'
import type { NotifierFormState } from './types'

describe('NotifierConfigForm', () => {
  afterEach(() => {
    cleanup()
  })

  it('supports enabling multiple notifier backends', async () => {
    // Given: Notifier config form with all backends disabled
    const onChange = vi.fn()
    const user = userEvent.setup()

    function Harness() {
      const [value, setValue] = useState<NotifierFormState>({
        mqtt: {
          enabled: false,
          config: NOTIFIER_BACKENDS.mqtt.defaultConfig,
        },
        sendgrid_email: {
          enabled: false,
          config: NOTIFIER_BACKENDS.sendgrid_email.defaultConfig,
        },
      })
      return (
        <NotifierConfigForm
          value={value}
          onChange={(nextValue) => {
            onChange(nextValue)
            setValue(nextValue)
          }}
        />
      )
    }

    render(<Harness />)

    // When: Operator enables MQTT and SendGrid channels
    await user.click(screen.getByRole('checkbox', { name: 'MQTT' }))
    await user.click(screen.getByRole('checkbox', { name: 'SendGrid Email' }))

    // Then: Form emits state with both backends enabled
    expect(onChange).toHaveBeenCalled()
    expect(onChange.mock.calls.at(-1)?.[0]).toMatchObject({
      mqtt: { enabled: true },
      sendgrid_email: { enabled: true },
    })
  })
})
