// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'

import { RtspForm } from './RtspForm'

describe('RtspForm', () => {
  afterEach(() => {
    cleanup()
  })

  it('renders RTSP fields and emits updated config values on change', async () => {
    // Given: RTSP form with baseline config and update callback
    const onChange = vi.fn()
    const user = userEvent.setup()
    function RtspHarness() {
      const [config, setConfig] = useState<Record<string, unknown>>({
        rtsp_url: 'rtsp://old-user:old-pass@camera.local/stream',
        output_dir: './recordings',
      })
      return (
        <RtspForm
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
    render(<RtspHarness />)

    // When: Operator updates RTSP URL value
    await user.clear(screen.getByLabelText('RTSP URL'))
    await user.type(screen.getByLabelText('RTSP URL'), 'rtsp://new-user:new-pass@camera.local/live')

    // Then: Form emits updated config preserving sibling fields
    expect(screen.getByLabelText('Output directory')).toBeTruthy()
    expect(onChange).toHaveBeenCalled()
    expect(onChange.mock.calls.at(-1)?.[0]).toMatchObject({
      rtsp_url: 'rtsp://new-user:new-pass@camera.local/live',
      output_dir: './recordings',
    })
  })
})
