// @vitest-environment happy-dom

import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { DetectionStep } from './DetectionStep'

vi.mock('../../shared/TestConnectionButton', () => ({
  TestConnectionButton: () => <div data-testid="test-connection-button">Test connection button</div>,
}))

describe('DetectionStep', () => {
  afterEach(() => {
    cleanup()
  })

  it('saves default detection config with VLM disabled', async () => {
    // Given: Detection step starts with default wizard data
    const onComplete = vi.fn()
    const onUpdateData = vi.fn()
    const user = userEvent.setup()
    render(
      <DetectionStep
        initialData={null}
        onComplete={onComplete}
        onUpdateData={onUpdateData}
        onSkip={vi.fn()}
      />,
    )

    // When: Operator saves and continues without enabling VLM
    await user.click(screen.getByRole('button', { name: 'Save and continue' }))

    // Then: Step emits default filter config and null VLM config
    expect(onUpdateData).toHaveBeenCalledWith({
      filter: {
        backend: 'yolo',
        config: {
          classes: ['person', 'car', 'dog', 'cat'],
          min_confidence: 0.5,
        },
      },
      vlm: null,
    })
    expect(onComplete).toHaveBeenCalledTimes(1)
  })

  it('enables VLM and emits trigger-only run mode by default', async () => {
    // Given: Detection step starts with default state and VLM disabled
    const onUpdateData = vi.fn()
    const user = userEvent.setup()
    render(
      <DetectionStep
        initialData={null}
        onComplete={vi.fn()}
        onUpdateData={onUpdateData}
        onSkip={vi.fn()}
      />,
    )

    // When: Operator enables VLM and saves detection settings
    await user.click(screen.getByLabelText('Enable AI scene analysis (VLM)'))
    await user.click(screen.getByRole('button', { name: 'Save and continue' }))

    // Then: Step emits non-null VLM config with trigger-only run mode
    expect(onUpdateData).toHaveBeenCalledWith(
      expect.objectContaining({
        vlm: expect.objectContaining({
          backend: 'openai',
          run_mode: 'trigger_only',
        }),
      }),
    )
  })

  it('maps skip button to wizard skip callback', async () => {
    // Given: Detection step with skip callback
    const onSkip = vi.fn()
    const user = userEvent.setup()
    render(
      <DetectionStep
        initialData={null}
        onComplete={vi.fn()}
        onUpdateData={vi.fn()}
        onSkip={onSkip}
      />,
    )

    // When: Operator skips detection step
    await user.click(screen.getByRole('button', { name: 'Skip detection step' }))

    // Then: Step requests skip transition
    expect(onSkip).toHaveBeenCalledTimes(1)
  })
})
