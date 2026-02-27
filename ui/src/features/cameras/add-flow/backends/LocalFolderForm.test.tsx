// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useState } from 'react'

import { LocalFolderForm } from './LocalFolderForm'

describe('LocalFolderForm', () => {
  afterEach(() => {
    cleanup()
  })

  it('updates watch directory and numeric polling settings', () => {
    // Given: Local-folder form with baseline watch/poll config
    const onChange = vi.fn()

    function Harness() {
      const [config, setConfig] = useState<Record<string, unknown>>({
        watch_dir: './recordings',
        poll_interval: 1.0,
        stability_threshold_s: 3.0,
      })
      return (
        <LocalFolderForm
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

    // When: Operator edits directory and timing fields
    fireEvent.change(screen.getByLabelText('Watch directory'), {
      target: { value: '/tmp/camera' },
    })
    fireEvent.change(screen.getByLabelText('Poll interval (seconds)'), {
      target: { value: '2.5' },
    })
    fireEvent.change(screen.getByLabelText('Stability threshold (seconds)'), {
      target: { value: '4.5' },
    })

    // Then: Form emits converted numeric values in source config
    expect(onChange.mock.calls.at(-1)?.[0]).toMatchObject({
      watch_dir: '/tmp/camera',
      poll_interval: 2.5,
      stability_threshold_s: 4.5,
    })
  })
})
