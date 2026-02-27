// @vitest-environment happy-dom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import type { CameraCreateActionResult } from '../../cameras/actions'
import { CameraStep } from './CameraStep'

const cameraAddFlowPropsSpy = vi.fn()

vi.mock('../../cameras/add-flow/CameraAddFlow', () => ({
  CameraAddFlow: (props: {
    existingCameraNames: readonly string[]
    defaultApplyChangesImmediately?: boolean
    onComplete: (
      payload: {
        name: string
        enabled: boolean
        source_backend: 'rtsp'
        source_config: Record<string, unknown>
      },
      options: { applyChangesImmediately: boolean },
    ) => Promise<CameraCreateActionResult>
    onDone: () => void
    onCancel: () => void
  }) => {
    cameraAddFlowPropsSpy(props)
    return (
      <section>
        <button
          type="button"
          onClick={() => {
            void props
              .onComplete(
                {
                  name: 'front_door',
                  enabled: true,
                  source_backend: 'rtsp',
                  source_config: {
                    rtsp_url: 'rtsp://camera.local/stream',
                  },
                },
                { applyChangesImmediately: false },
              )
              .then((result) => {
                if (result.ok) {
                  props.onDone()
                }
              })
          }}
        >
          Complete camera flow
        </button>
        <button type="button" onClick={props.onCancel}>
          Skip camera flow
        </button>
      </section>
    )
  },
}))

describe('CameraStep', () => {
  beforeEach(() => {
    cameraAddFlowPropsSpy.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it('stores camera draft and completes setup step when flow succeeds', async () => {
    // Given: Camera step wrapper with completion/update callbacks
    const onComplete = vi.fn()
    const onUpdateData = vi.fn()
    const user = userEvent.setup()
    render(
      <CameraStep
        existingCameraNames={['existing_camera']}
        onComplete={onComplete}
        onUpdateData={onUpdateData}
        onSkip={vi.fn()}
      />,
    )

    // When: Embedded camera flow completes successfully
    await user.click(screen.getByRole('button', { name: 'Complete camera flow' }))

    // Then: Wrapper records camera payload and advances wizard
    expect(cameraAddFlowPropsSpy).toHaveBeenCalled()
    expect(cameraAddFlowPropsSpy.mock.calls[0]?.[0].defaultApplyChangesImmediately).toBe(false)
    expect(onUpdateData).toHaveBeenCalledWith(
      expect.objectContaining({
        name: 'front_door',
        source_backend: 'rtsp',
      }),
    )
    expect(onComplete).toHaveBeenCalledTimes(1)
  })

  it('maps flow cancel to step skip callback', async () => {
    // Given: Camera step wrapper with skip callback
    const onSkip = vi.fn()
    const user = userEvent.setup()
    render(
      <CameraStep
        existingCameraNames={[]}
        onComplete={vi.fn()}
        onUpdateData={vi.fn()}
        onSkip={onSkip}
      />,
    )

    // When: Operator cancels camera flow
    await user.click(screen.getByRole('button', { name: 'Skip camera flow' }))

    // Then: Step wrapper requests wizard skip transition
    expect(onSkip).toHaveBeenCalledTimes(1)
  })
})
