// @vitest-environment happy-dom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import type { useStorageBackendsQuery } from '../../../api/hooks/useStorageBackendsQuery'
import { StorageStep } from './StorageStep'

const testConnectionButtonPropsSpy = vi.fn()
const useStorageBackendsQueryMock = vi.fn()

vi.mock('../../shared/TestConnectionButton', () => ({
  TestConnectionButton: (props: unknown) => {
    testConnectionButtonPropsSpy(props)
    return <div data-testid="test-connection-button">Test connection button</div>
  },
}))

vi.mock('../../../api/hooks/useStorageBackendsQuery', () => ({
  useStorageBackendsQuery: () => useStorageBackendsQueryMock(),
}))

describe('StorageStep', () => {
  beforeEach(() => {
    testConnectionButtonPropsSpy.mockReset()
    useStorageBackendsQueryMock.mockReset()
    useStorageBackendsQueryMock.mockReturnValue({
      data: null,
      error: null,
      isPending: false,
      isFetching: false,
      refetch: vi.fn(),
    } as unknown as ReturnType<typeof useStorageBackendsQuery>)
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
    // Given: Storage step initially targeting local backend with metadata-driven labels/defaults
    const user = userEvent.setup()
    useStorageBackendsQueryMock.mockReturnValue({
      data: [
        {
          backend: 'local',
          label: 'Local FS',
          description: 'Store on local disk',
          config_schema: {},
          fields: [
            {
              name: 'root',
              type: 'string',
              required: true,
              description: 'Root path',
              default: './storage',
              secret: false,
            },
          ],
          secret_fields: [],
        },
        {
          backend: 'dropbox',
          label: 'Dropbox Cloud',
          description: 'Upload to Dropbox',
          config_schema: {},
          fields: [
            {
              name: 'root',
              type: 'string',
              required: true,
              description: 'Dropbox root',
              default: null,
              secret: false,
            },
            {
              name: 'token_env',
              type: 'string',
              required: true,
              description: 'Token env var',
              default: 'DROPBOX_TOKEN',
              secret: false,
            },
          ],
          secret_fields: [],
        },
      ],
      error: null,
      isPending: false,
      isFetching: false,
      refetch: vi.fn(),
    } as unknown as ReturnType<typeof useStorageBackendsQuery>)
    render(
      <StorageStep
        initialData={null}
        onComplete={vi.fn()}
        onUpdateData={vi.fn()}
        onSkip={vi.fn()}
      />,
    )

    // When: Operator switches to Dropbox backend
    await user.click(screen.getByRole('button', { name: 'Dropbox Cloud' }))

    // Then: Test-connection request uses storage type + metadata-driven dropbox defaults
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
