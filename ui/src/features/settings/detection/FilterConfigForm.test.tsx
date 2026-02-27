// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'

import { FilterConfigForm } from './FilterConfigForm'
import type { FilterFormState } from './types'

describe('FilterConfigForm', () => {
  afterEach(() => {
    cleanup()
  })

  it('adds and removes detection classes', async () => {
    // Given: Filter config form with initial default classes
    const onChange = vi.fn()
    const user = userEvent.setup()

    function Harness() {
      const [value, setValue] = useState<FilterFormState>({
        backend: 'yolo',
        config: {
          classes: ['person', 'car'],
          min_confidence: 0.5,
        },
      })
      return (
        <FilterConfigForm
          value={value}
          onChange={(nextValue) => {
            onChange(nextValue)
            setValue(nextValue)
          }}
        />
      )
    }

    render(<Harness />)

    // When: Operator adds a new class then removes an existing class
    await user.type(screen.getByLabelText('Detection classes'), 'dog')
    await user.click(screen.getByRole('button', { name: 'Add class' }))
    await user.click(screen.getByRole('button', { name: 'Remove class car' }))

    // Then: Emitted state reflects class additions/removals
    expect(onChange).toHaveBeenCalled()
    expect(onChange.mock.calls.at(-1)?.[0]).toMatchObject({
      config: {
        classes: ['person', 'dog'],
      },
    })
  })

  it('updates minimum confidence from slider input', () => {
    // Given: Filter config form with default confidence
    const onChange = vi.fn()

    function Harness() {
      const [value, setValue] = useState<FilterFormState>({
        backend: 'yolo',
        config: {
          classes: ['person'],
          min_confidence: 0.5,
        },
      })
      return (
        <FilterConfigForm
          value={value}
          onChange={(nextValue) => {
            onChange(nextValue)
            setValue(nextValue)
          }}
        />
      )
    }

    render(<Harness />)

    // When: Operator changes confidence slider value
    const slider = screen.getByLabelText('Confidence threshold: 0.50')
    fireEvent.change(slider, { target: { value: '0.7' } })

    // Then: Form emits updated confidence value
    expect(onChange).toHaveBeenCalled()
    expect(onChange.mock.calls.at(-1)?.[0]).toMatchObject({
      config: {
        min_confidence: 0.7,
      },
    })
  })
})
