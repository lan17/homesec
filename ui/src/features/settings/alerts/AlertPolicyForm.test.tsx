// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'

import { AlertPolicyForm } from './AlertPolicyForm'
import type { AlertPolicyFormState } from '../notifiers/types'

describe('AlertPolicyForm', () => {
  afterEach(() => {
    cleanup()
  })

  it('toggles selected risk levels for alert policy baseline', async () => {
    // Given: Alert policy defaults to high and critical selections
    const onChange = vi.fn()
    const user = userEvent.setup()

    function Harness() {
      const [value, setValue] = useState<AlertPolicyFormState>({
        selectedRiskLevels: ['high', 'critical'],
      })
      return (
        <AlertPolicyForm
          value={value}
          onChange={(nextValue) => {
            onChange(nextValue)
            setValue(nextValue)
          }}
        />
      )
    }

    render(<Harness />)

    // When: Operator enables medium risk notifications
    await user.click(screen.getByRole('checkbox', { name: 'medium' }))

    // Then: Form emits updated selected risk levels
    expect(onChange).toHaveBeenCalled()
    expect(onChange.mock.calls.at(-1)?.[0]).toEqual({
      selectedRiskLevels: ['high', 'critical', 'medium'],
    })
  })
})
