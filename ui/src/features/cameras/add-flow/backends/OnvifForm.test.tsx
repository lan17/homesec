// @vitest-environment happy-dom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'

import { OnvifForm } from './OnvifForm'

const wizardPropsSpy = vi.fn()

vi.mock('../../components/OnvifDiscoveryWizard', () => ({
  OnvifDiscoveryWizard: (props: {
    onCreateCamera: (payload: {
      name: string
      enabled: boolean
      source_backend: 'rtsp'
      source_config: Record<string, unknown>
    }) => Promise<{ ok: true }>
    onClose: () => void
  }) => {
    wizardPropsSpy(props)
    return (
      <div data-testid="onvif-wizard">
        <button
          type="button"
          onClick={() => {
            void props.onCreateCamera({
              name: 'front_door',
              enabled: true,
              source_backend: 'rtsp',
              source_config: {
                rtsp_url: 'rtsp://camera.local/live',
                output_dir: './recordings',
              },
            })
          }}
        >
          Resolve stream
        </button>
        <button type="button" onClick={props.onClose}>
          Close wizard
        </button>
      </div>
    )
  },
}))

describe('OnvifForm', () => {
  beforeEach(() => {
    wizardPropsSpy.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it('collapses wizard after stream resolution and allows reopening', async () => {
    // Given: ONVIF form with no selected stream URI
    const onSuggestedNameChange = vi.fn()
    const user = userEvent.setup()
    function Harness() {
      const [config, setConfig] = useState<Record<string, unknown>>({
        rtsp_url: '',
        output_dir: './recordings',
      })
      return (
        <OnvifForm
          config={config}
          onChange={setConfig}
          stepIndex={0}
          onSuggestedNameChange={onSuggestedNameChange}
        />
      )
    }

    render(<Harness />)
    expect(screen.getByTestId('onvif-wizard')).toBeTruthy()

    // When: Wizard resolves a stream candidate
    await user.click(screen.getByRole('button', { name: 'Resolve stream' }))

    // Then: Selected URI is shown and wizard is collapsed
    expect(screen.queryByTestId('onvif-wizard')).toBeNull()
    expect(screen.getByText(/Selected stream URI:/)).toBeTruthy()
    expect(onSuggestedNameChange).toHaveBeenCalledWith('front_door')

    // When: Operator reopens stream selection
    await user.click(screen.getByRole('button', { name: 'Choose different stream' }))

    // Then: Wizard is rendered again for reselection
    expect(screen.getByTestId('onvif-wizard')).toBeTruthy()
  })
})
