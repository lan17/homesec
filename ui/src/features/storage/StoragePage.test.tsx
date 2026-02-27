// @vitest-environment happy-dom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import type { RuntimeStatusSnapshot } from '../../api/client'
import type { StorageBackendsResponse, StorageResponse } from '../../api/generated/types'
import type { useRuntimeStatusQuery } from '../../api/hooks/useRuntimeStatusQuery'
import type { useStorageBackendsQuery } from '../../api/hooks/useStorageBackendsQuery'
import type { useStorageQuery } from '../../api/hooks/useStorageQuery'
import type { useUpdateStorageMutation } from '../../api/hooks/useStorageMutation'
import { StoragePage } from './StoragePage'

const useStorageQueryMock = vi.fn()
const useStorageBackendsQueryMock = vi.fn()
const useRuntimeStatusQueryMock = vi.fn()
const useUpdateStorageMutationMock = vi.fn()
const useRuntimeReloadMutationMock = vi.fn()
const setupTestConnectionMutateAsyncMock = vi.fn()

vi.mock('../../api/hooks/useStorageQuery', () => ({
  useStorageQuery: () => useStorageQueryMock(),
}))

vi.mock('../../api/hooks/useStorageBackendsQuery', () => ({
  useStorageBackendsQuery: () => useStorageBackendsQueryMock(),
}))

vi.mock('../../api/hooks/useRuntimeStatusQuery', () => ({
  useRuntimeStatusQuery: () => useRuntimeStatusQueryMock(),
}))

vi.mock('../../api/hooks/useStorageMutation', () => ({
  useUpdateStorageMutation: () => useUpdateStorageMutationMock(),
}))

vi.mock('../../api/hooks/useRuntimeReloadMutation', () => ({
  useRuntimeReloadMutation: () => useRuntimeReloadMutationMock(),
}))

vi.mock('../../api/hooks/useSetupTestConnectionMutation', () => ({
  useSetupTestConnectionMutation: () => ({
    mutateAsync: setupTestConnectionMutateAsyncMock,
    isPending: false,
  }),
}))

function defaultStorageSnapshot(): StorageResponse {
  return {
    backend: 'local',
    config: { root: './storage' },
    paths: {
      clips_dir: 'clips',
      backups_dir: 'backups',
      artifacts_dir: 'artifacts',
    },
  }
}

function defaultStorageBackends(): StorageBackendsResponse {
  return [
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
      label: 'Dropbox',
      description: 'Upload to Dropbox',
      config_schema: {},
      fields: [
        {
          name: 'root',
          type: 'string',
          required: true,
          description: 'Dropbox root',
          default: '/homesec',
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
  ]
}

function defaultRuntimeStatus(): RuntimeStatusSnapshot {
  return {
    httpStatus: 200,
    state: 'idle',
    generation: 3,
    reload_in_progress: false,
    active_config_version: 'cfg-v3',
    last_reload_at: null,
    last_reload_error: null,
  }
}

function setupPage({
  storage = defaultStorageSnapshot(),
  backends = defaultStorageBackends(),
  updateResponse = {
    restart_required: true,
    storage: {
      backend: 'dropbox',
      config: {
        root: '/homesec',
        token_env: 'DROPBOX_TOKEN',
      },
      paths: {
        clips_dir: 'clips',
        backups_dir: 'backups',
        artifacts_dir: 'artifacts',
      },
    },
    runtime_reload: null,
  },
}: {
  storage?: StorageResponse
  backends?: StorageBackendsResponse
  updateResponse?: {
    restart_required: boolean
    storage: {
      backend: string
      config: Record<string, unknown>
      paths: Record<string, unknown>
    } | null
    runtime_reload: {
      accepted: boolean
      message: string
      target_generation: number
    } | null
  }
} = {}) {
  const storageRefetch = vi.fn().mockResolvedValue({ data: storage })
  const backendsRefetch = vi.fn().mockResolvedValue({ data: backends })
  const runtimeRefetch = vi.fn().mockResolvedValue({ data: defaultRuntimeStatus() })

  useStorageQueryMock.mockReturnValue({
    data: storage,
    isPending: false,
    isFetching: false,
    error: null,
    refetch: storageRefetch,
  } as unknown as ReturnType<typeof useStorageQuery>)

  useStorageBackendsQueryMock.mockReturnValue({
    data: backends,
    isPending: false,
    isFetching: false,
    error: null,
    refetch: backendsRefetch,
  } as unknown as ReturnType<typeof useStorageBackendsQuery>)

  useRuntimeStatusQueryMock.mockReturnValue({
    data: defaultRuntimeStatus(),
    isPending: false,
    isFetching: false,
    error: null,
    refetch: runtimeRefetch,
  } as unknown as ReturnType<typeof useRuntimeStatusQuery>)

  const updateMutateAsync = vi.fn().mockResolvedValue(updateResponse)
  useUpdateStorageMutationMock.mockReturnValue({
    mutateAsync: updateMutateAsync,
    isPending: false,
    error: null,
  } as unknown as ReturnType<typeof useUpdateStorageMutation>)

  const reloadMutateAsync = vi.fn().mockResolvedValue({
    accepted: true,
    message: 'Runtime reload accepted',
    target_generation: 8,
    httpStatus: 202,
  })
  useRuntimeReloadMutationMock.mockReturnValue({
    mutateAsync: reloadMutateAsync,
    isPending: false,
    error: null,
  })

  setupTestConnectionMutateAsyncMock.mockResolvedValue({
    httpStatus: 200,
    success: true,
    message: 'Storage probe succeeded.',
    latency_ms: 10.1,
    details: null,
  })

  render(<StoragePage />)

  return {
    updateMutateAsync,
    reloadMutateAsync,
  }
}

describe('StoragePage', () => {
  beforeEach(() => {
    useStorageQueryMock.mockReset()
    useStorageBackendsQueryMock.mockReset()
    useRuntimeStatusQueryMock.mockReset()
    useUpdateStorageMutationMock.mockReset()
    useRuntimeReloadMutationMock.mockReset()
    setupTestConnectionMutateAsyncMock.mockReset()
    vi.restoreAllMocks()
  })

  afterEach(() => {
    cleanup()
  })

  it('renders backend selector labels from backend metadata response', () => {
    // Given: Storage backend metadata with custom labels is available
    setupPage()

    // When: The storage page renders
    const localButton = screen.getByRole('button', { name: 'Local FS' })
    const dropboxButton = screen.getByRole('button', { name: 'Dropbox' })

    // Then: Selector cards use metadata-driven labels
    expect(localButton).toBeTruthy()
    expect(dropboxButton).toBeTruthy()
  })

  it('does not mutate when backend switch confirmation is cancelled', async () => {
    // Given: A page where operator changes backend and declines switch confirmation
    const harness = setupPage()
    const user = userEvent.setup()
    Object.defineProperty(window, 'confirm', {
      configurable: true,
      value: vi.fn(() => false),
    })

    // When: Operator switches to dropbox and clicks save
    await user.click(screen.getByRole('button', { name: 'Dropbox' }))
    await user.click(screen.getByRole('button', { name: 'Save storage settings' }))

    // Then: No storage update mutation is sent
    expect(harness.updateMutateAsync).not.toHaveBeenCalled()
  })

  it('submits backend switch payload and surfaces pending runtime reload message', async () => {
    // Given: A page where backend switch is confirmed and response requires runtime reload
    const harness = setupPage()
    const user = userEvent.setup()
    Object.defineProperty(window, 'confirm', {
      configurable: true,
      value: vi.fn(() => true),
    })

    // When: Operator switches backend and saves settings
    await user.click(screen.getByRole('button', { name: 'Dropbox' }))
    await user.click(screen.getByRole('button', { name: 'Save storage settings' }))

    // Then: PATCH payload includes backend + config and page shows pending reload guidance
    expect(harness.updateMutateAsync).toHaveBeenCalledTimes(1)
    expect(harness.updateMutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        payload: expect.objectContaining({
          backend: 'dropbox',
          config: expect.objectContaining({
            root: '/homesec',
            token_env: 'DROPBOX_TOKEN',
          }),
        }),
      }),
    )
    expect(
      screen.getByText('Storage configuration updated. Apply runtime reload to activate changes.'),
    ).toBeTruthy()
  })

  it('passes applyChanges=true and handles immediate runtime reload response', async () => {
    // Given: Storage update responds with runtime reload acceptance and apply-immediately enabled
    const harness = setupPage({
      updateResponse: {
        restart_required: false,
        storage: {
          backend: 'local',
          config: {
            root: '/var/lib/homesec',
          },
          paths: {
            clips_dir: 'clips',
            backups_dir: 'backups',
            artifacts_dir: 'artifacts',
          },
        },
        runtime_reload: {
          accepted: true,
          message: 'Runtime reload accepted',
          target_generation: 10,
        },
      },
    })
    const user = userEvent.setup()

    // When: Operator edits local root, enables immediate apply, and saves
    await user.clear(screen.getByLabelText('Storage root directory'))
    await user.type(screen.getByLabelText('Storage root directory'), '/var/lib/homesec')
    await user.click(screen.getByLabelText('Apply changes immediately (runtime reload)'))
    await user.click(screen.getByRole('button', { name: 'Save storage settings' }))

    // Then: Mutation includes applyChanges=true and success message reflects runtime reload
    expect(harness.updateMutateAsync).toHaveBeenCalledTimes(1)
    expect(harness.updateMutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        applyChanges: true,
      }),
    )
    expect(screen.getByText('Storage configuration updated. Runtime reload accepted.')).toBeTruthy()
  })

  it('submits secret field patch values without requiring non-secret config edits', async () => {
    // Given: Backend metadata includes a write-only secret field for local backend
    const harness = setupPage({
      backends: [
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
            {
              name: 'access_token',
              type: 'string',
              required: false,
              description: 'Optional token',
              default: null,
              secret: true,
            },
          ],
          secret_fields: ['access_token'],
        },
      ],
      updateResponse: {
        restart_required: true,
        storage: {
          backend: 'local',
          config: {
            root: './storage',
          },
          paths: {
            clips_dir: 'clips',
            backups_dir: 'backups',
            artifacts_dir: 'artifacts',
          },
        },
        runtime_reload: null,
      },
    })
    const user = userEvent.setup()

    // When: Operator enters secret replacement value and saves
    await user.type(screen.getByLabelText('access_token'), 'replace-me')
    await user.click(screen.getByRole('button', { name: 'Save storage settings' }))

    // Then: Mutation payload includes write-only secret patch field
    expect(harness.updateMutateAsync).toHaveBeenCalledTimes(1)
    expect(harness.updateMutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        payload: expect.objectContaining({
          config: expect.objectContaining({
            access_token: 'replace-me',
          }),
        }),
      }),
    )
  })
})
