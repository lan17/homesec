// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'

import { FtpForm } from './FtpForm'

describe('FtpForm', () => {
  afterEach(() => {
    cleanup()
  })

  it('updates host and port values while preserving sibling ftp fields', () => {
    // Given: FTP form initialized with valid baseline config
    const onChange = vi.fn()

    function Harness() {
      const [config, setConfig] = useState<Record<string, unknown>>({
        host: '0.0.0.0',
        port: 2121,
        root_dir: './ftp_incoming',
        anonymous: true,
      })
      return (
        <FtpForm
          config={config}
          onChange={(nextConfig) => {
            onChange(nextConfig)
            setConfig(nextConfig)
          }}
          stepIndex={0}
          onSuggestedNameChange={vi.fn()}
        />
      )
    }

    render(<Harness />)

    // When: Operator changes host and port fields
    fireEvent.change(screen.getByLabelText('Bind host'), {
      target: { value: '127.0.0.1' },
    })
    fireEvent.change(screen.getByLabelText('Bind port'), {
      target: { value: '2200' },
    })

    // Then: Form emits updated values and preserves unrelated keys
    expect(onChange).toHaveBeenCalled()
    expect(onChange.mock.calls.at(-1)?.[0]).toMatchObject({
      host: '127.0.0.1',
      port: 2200,
      root_dir: './ftp_incoming',
      anonymous: true,
    })
  })

  it('toggles anonymous uploads flag', async () => {
    // Given: FTP form with anonymous uploads enabled
    const onChange = vi.fn()
    const user = userEvent.setup()

    function Harness() {
      const [config, setConfig] = useState<Record<string, unknown>>({
        host: '0.0.0.0',
        port: 2121,
        root_dir: './ftp_incoming',
        anonymous: true,
      })
      return (
        <FtpForm
          config={config}
          onChange={(nextConfig) => {
            onChange(nextConfig)
            setConfig(nextConfig)
          }}
          stepIndex={0}
          onSuggestedNameChange={vi.fn()}
        />
      )
    }

    render(<Harness />)

    // When: Operator disables anonymous uploads
    await user.click(screen.getByLabelText('Allow anonymous uploads'))

    // Then: Form emits boolean change for anonymous field
    expect(onChange.mock.calls.at(-1)?.[0]).toMatchObject({
      anonymous: false,
    })
  })
})
