// @vitest-environment happy-dom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { StorageStep } from './StorageStep'

const testConnectionButtonPropsSpy = vi.fn()

vi.mock('../../shared/TestConnectionButton', () => ({
  TestConnectionButton: (props: unknown) => {
    testConnectionButtonPropsSpy(props)
    return <div data-testid="test-connection-button">Test connection button</div>
  },
}))

describe('StorageStep', () => {
  beforeEach(() => {
    testConnectionButtonPropsSpy.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it('saves storage config and advances step on valid input', async () => {
    // Given: Storage step with no pre-existing wizard storage draft
    const onComplete = vi.fn()
    const onUpdateData = vi.fn()
    const user = userEvent.setup()
    render(
      <StorageStep
        initialData={null}
        onComplete={onComplete}
        onUpdateData={onUpdateData}
        onSkip={vi.fn()}
      />,
    )

    // When: Operator saves defaults and continues
    await user.click(screen.getByRole('button', { name: 'Save and continue' }))

    // Then: Storage draft is emitted and wizard advances
    expect(onUpdateData).toHaveBeenCalledWith({
      backend: 'local',
      config: { root: './storage' },
    })
    expect(onComplete).toHaveBeenCalledTimes(1)
  })

  it('maps skip button to wizard skip callback', async () => {
    // Given: Storage step with skip callback
    const onSkip = vi.fn()
    const user = userEvent.setup()
    render(
      <StorageStep
        initialData={null}
        onComplete={vi.fn()}
        onUpdateData={vi.fn()}
        onSkip={onSkip}
      />,
    )

    // When: Operator skips storage step
    await user.click(screen.getByRole('button', { name: 'Skip storage step' }))

    // Then: Step requests skip transition
    expect(onSkip).toHaveBeenCalledTimes(1)
  })

  it('builds storage test request from active backend selection', async () => {
    // Given: Storage step initially targeting local backend
    const user = userEvent.setup()
    render(
      <StorageStep
        initialData={null}
        onComplete={vi.fn()}
        onUpdateData={vi.fn()}
        onSkip={vi.fn()}
      />,
    )

    // When: Operator switches to Dropbox backend
    await user.click(screen.getByRole('button', { name: 'Dropbox' }))

    // Then: Test-connection request uses storage type + dropbox backend/config
    expect(testConnectionButtonPropsSpy).toHaveBeenCalled()
    expect(testConnectionButtonPropsSpy.mock.calls.at(-1)?.[0]).toMatchObject({
      request: {
        type: 'storage',
        backend: 'dropbox',
        config: { root: '/homesec', token_env: 'DROPBOX_TOKEN' },
      },
    })
  })
})
