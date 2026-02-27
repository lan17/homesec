// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'

import { MqttForm } from './MqttForm'

describe('MqttForm', () => {
  afterEach(() => {
    cleanup()
  })

  it('emits mqtt config updates for host and topic template', async () => {
    // Given: MQTT form with default local broker settings
    const onChange = vi.fn()
    const user = userEvent.setup()

    function Harness() {
      const [value, setValue] = useState<Record<string, unknown>>({
        host: 'localhost',
        port: 1883,
        topic_template: 'homecam/alerts/{camera_name}',
      })
      return (
        <MqttForm
          config={value}
          onChange={(nextValue) => {
            onChange(nextValue)
            setValue(nextValue)
          }}
        />
      )
    }

    render(<Harness />)

    // When: Operator updates host and topic template
    await user.clear(screen.getByLabelText('MQTT host'))
    await user.type(screen.getByLabelText('MQTT host'), 'mqtt.local')
    fireEvent.change(screen.getByLabelText('Topic template'), {
      target: { value: 'homesec/alerts/{camera_name}' },
    })

    // Then: Form emits updated mqtt notifier config
    expect(onChange).toHaveBeenCalled()
    expect(onChange.mock.calls.at(-1)?.[0]).toMatchObject({
      host: 'mqtt.local',
      topic_template: 'homesec/alerts/{camera_name}',
    })
  })
})
