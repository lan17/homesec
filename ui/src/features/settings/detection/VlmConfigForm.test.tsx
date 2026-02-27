// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'

import type { TestConnectionResponse } from '../../../api/generated/types'
import { VlmConfigForm } from './VlmConfigForm'
import type { VlmFormState } from './types'

const testConnectionButtonPropsSpy = vi.fn()

vi.mock('../../shared/TestConnectionButton', () => ({
  TestConnectionButton: (props: {
    request: unknown
    result: TestConnectionResponse | null
    onResult: (result: TestConnectionResponse) => void
  }) => {
    testConnectionButtonPropsSpy(props)
    return <div data-testid="test-connection-button">Test connection button</div>
  },
}))

describe('VlmConfigForm', () => {
  afterEach(() => {
    cleanup()
    testConnectionButtonPropsSpy.mockReset()
  })

  it('toggles VLM section and coerces run mode to trigger-only when enabled', async () => {
    // Given: VLM section starts disabled with run mode set to never
    const user = userEvent.setup()

    function Harness() {
      const [enabled, setEnabled] = useState(false)
      const [value, setValue] = useState<VlmFormState>({
        backend: 'openai',
        run_mode: 'never',
        trigger_classes: ['person'],
        config: {
          api_key_env: 'OPENAI_API_KEY',
          model: 'gpt-4o',
          base_url: 'https://api.openai.com/v1',
        },
      })
      return (
        <VlmConfigForm
          enabled={enabled}
          value={value}
          filterClasses={['person', 'car']}
          onToggle={setEnabled}
          onChange={setValue}
        />
      )
    }

    render(<Harness />)

    // When: Operator enables VLM analysis
    await user.click(screen.getByLabelText('Enable AI scene analysis (VLM)'))

    // Then: Expanded section is shown with run mode set to trigger-only
    expect(screen.getByLabelText('API endpoint')).toBeTruthy()
    const runModeInput = screen.getByLabelText('Run mode') as HTMLSelectElement
    expect(runModeInput.value).toBe('trigger_only')
    expect(screen.getByTestId('test-connection-button')).toBeTruthy()
  })

  it('emits model and trigger-class updates from the form', async () => {
    // Given: VLM section starts enabled with default values
    const onChange = vi.fn()
    const user = userEvent.setup()

    function Harness() {
      const [value, setValue] = useState<VlmFormState>({
        backend: 'openai',
        run_mode: 'trigger_only',
        trigger_classes: ['person'],
        config: {
          api_key_env: 'OPENAI_API_KEY',
          model: 'gpt-4o',
          base_url: 'https://api.openai.com/v1',
        },
      })
      return (
        <VlmConfigForm
          enabled
          value={value}
          filterClasses={['person', 'car']}
          onToggle={vi.fn()}
          onChange={(nextValue) => {
            onChange(nextValue)
            setValue(nextValue)
          }}
        />
      )
    }

    render(<Harness />)

    // When: Operator edits model and toggles a trigger class
    await user.clear(screen.getByLabelText('Model'))
    await user.type(screen.getByLabelText('Model'), 'gpt-4.1-mini')
    await user.click(screen.getByRole('checkbox', { name: 'car' }))

    // Then: Form emits the updated VLM state
    expect(onChange).toHaveBeenCalled()
    expect(onChange.mock.calls.at(-1)?.[0]).toMatchObject({
      config: {
        model: 'gpt-4.1-mini',
      },
      trigger_classes: ['person', 'car'],
    })
  })
})
