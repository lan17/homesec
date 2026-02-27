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

  it('updates minimum risk threshold for alert policy baseline', async () => {
    // Given: Alert policy defaults to high threshold
    const onChange = vi.fn()
    const user = userEvent.setup()

    function Harness() {
      const [value, setValue] = useState<AlertPolicyFormState>({
        minRiskLevel: 'high',
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

    // When: Operator lowers threshold to medium
    await user.selectOptions(screen.getByLabelText('Minimum risk level'), 'medium')

    // Then: Form emits updated threshold value
    expect(onChange).toHaveBeenCalled()
    expect(onChange.mock.calls.at(-1)?.[0]).toEqual({
      minRiskLevel: 'medium',
    })
  })
})
