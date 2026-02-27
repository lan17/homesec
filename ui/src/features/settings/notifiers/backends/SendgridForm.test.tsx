// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useState } from 'react'

import { SendgridForm } from './SendgridForm'

describe('SendgridForm', () => {
  afterEach(() => {
    cleanup()
  })

  it('parses comma-separated recipients into notifier config array', async () => {
    // Given: SendGrid form with one recipient configured
    const onChange = vi.fn()

    function Harness() {
      const [value, setValue] = useState<Record<string, unknown>>({
        api_key_env: 'SENDGRID_API_KEY',
        from_email: 'homesec@localhost',
        to_emails: ['ops@localhost'],
      })
      return (
        <SendgridForm
          config={value}
          onChange={(nextValue) => {
            onChange(nextValue)
            setValue(nextValue)
          }}
        />
      )
    }

    render(<Harness />)

    // When: Operator updates comma-separated recipient input
    fireEvent.change(screen.getByLabelText('Recipient emails (comma separated)'), {
      target: { value: 'a@example.com, b@example.com' },
    })

    // Then: Form emits parsed recipient array for backend config
    expect(onChange).toHaveBeenCalled()
    expect(onChange.mock.calls.at(-1)?.[0]).toMatchObject({
      to_emails: ['a@example.com', 'b@example.com'],
    })
  })
})
