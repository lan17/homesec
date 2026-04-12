import { useMemo, useState } from 'react'

import {
  clearApiKey,
  isUnauthorizedAPIError,
  saveApiKey,
} from '../../api/client'
import { useRuntimeReloadMutation } from '../../api/hooks/useRuntimeReloadMutation'
import { useRuntimeStatusQuery } from '../../api/hooks/useRuntimeStatusQuery'
import { useStorageBackendsQuery } from '../../api/hooks/useStorageBackendsQuery'
import { useUpdateStorageMutation } from '../../api/hooks/useStorageMutation'
import { useStorageQuery } from '../../api/hooks/useStorageQuery'
import type {
  StorageBackendMetadata,
  StorageUpdate,
  TestConnectionResponse,
} from '../../api/generated/types'
import { ApiKeyGate } from '../../components/ui/ApiKeyGate'
import { Button } from '../../components/ui/Button'
import { Card } from '../../components/ui/Card'
import { StatusBadge } from '../../components/ui/StatusBadge'
import { TestConnectionButton } from '../shared/TestConnectionButton'
import { describeUnknownError } from '../shared/errorPresentation'
import {
  STORAGE_BACKENDS,
  STORAGE_BACKEND_ORDER,
} from '../settings/storage/backends'
import {
  buildStorageConfigPatch,
  buildStorageSecretPatch,
  cloneStorageConfig,
  defaultConfigForBackend,
  isSupportedStorageBackend,
} from '../settings/storage/editorModel'

const REDACTED_PLACEHOLDER = '***redacted***'

interface StorageDraft {
  backend: string
  config: Record<string, unknown>
}

interface BackendOption {
  backend: string
  label: string
  description: string
  metadata: StorageBackendMetadata | null
}

function runtimeStatusTone(status: {
  state: 'idle' | 'reloading' | 'failed'
  reload_in_progress: boolean
}): 'degraded' | 'healthy' | 'unhealthy' {
  if (status.state === 'failed') {
    return 'unhealthy'
  }
  if (status.reload_in_progress || status.state === 'reloading') {
    return 'degraded'
  }
  return 'healthy'
}

function formatRuntimeTimestamp(value: string | null): string {
  if (!value) {
    return 'n/a'
  }
  const parsed = new Date(value)
  if (Number.isNaN(parsed.valueOf())) {
    return value
  }
  return parsed.toLocaleString()
}

function runtimeStatusLabel(state: 'idle' | 'reloading' | 'failed'): string {
  if (state === 'failed') {
    return 'Failed'
  }
  if (state === 'reloading') {
    return 'Reloading'
  }
  return 'Idle'
}

export function StoragePage() {
  const storageQuery = useStorageQuery()
  const storageBackendsQuery = useStorageBackendsQuery()
  const runtimeStatusQuery = useRuntimeStatusQuery()
  const updateStorageMutation = useUpdateStorageMutation()
  const runtimeReloadMutation = useRuntimeReloadMutation()

  const [editedDraft, setEditedDraft] = useState<StorageDraft | null>(null)
  const [validationError, setValidationError] = useState<string | null>(null)
  const [actionFeedback, setActionFeedback] = useState<string | null>(null)
  const [hasPendingReload, setHasPendingReload] = useState(false)
  const [pendingReloadMessage, setPendingReloadMessage] = useState<string | null>(null)
  const [applyChangesImmediately, setApplyChangesImmediately] = useState(false)
  const [testResult, setTestResult] = useState<TestConnectionResponse | null>(null)
  const [secretInputs, setSecretInputs] = useState<Record<string, string>>({})

  const baseline = useMemo<StorageDraft | null>(() => {
    if (!storageQuery.data) {
      return null
    }
    return {
      backend: storageQuery.data.backend,
      config: cloneStorageConfig(storageQuery.data.config),
    }
  }, [storageQuery.data])

  const draft = editedDraft ?? baseline

  const backendMetadataByName = useMemo(() => {
    const map = new Map<string, StorageBackendMetadata>()
    for (const backend of storageBackendsQuery.data ?? []) {
      map.set(backend.backend, backend)
    }
    return map
  }, [storageBackendsQuery.data])

  const backendOptions = useMemo<BackendOption[]>(() => {
    if (storageBackendsQuery.data && storageBackendsQuery.data.length > 0) {
      return storageBackendsQuery.data.map((backend) => ({
        backend: backend.backend,
        label: backend.label,
        description: backend.description,
        metadata: backend,
      }))
    }

    return STORAGE_BACKEND_ORDER.map((backend) => ({
      backend,
      label: STORAGE_BACKENDS[backend].label,
      description: STORAGE_BACKENDS[backend].description,
      metadata: null,
    }))
  }, [storageBackendsQuery.data])

  const selectedMetadata = useMemo(() => {
    if (!draft) {
      return null
    }
    return backendMetadataByName.get(draft.backend) ?? null
  }, [backendMetadataByName, draft])

  const selectedSecretFields = useMemo(() => {
    if (!selectedMetadata) {
      return [] as string[]
    }
    return selectedMetadata.secret_fields
  }, [selectedMetadata])

  const validationConfig = useMemo(() => {
    if (!draft) {
      return null
    }
    return {
      ...draft.config,
      ...buildStorageSecretPatch(secretInputs),
    }
  }, [draft, secretInputs])

  const unauthorized =
    isUnauthorizedAPIError(storageQuery.error)
    || isUnauthorizedAPIError(storageBackendsQuery.error)
    || isUnauthorizedAPIError(runtimeStatusQuery.error)
    || isUnauthorizedAPIError(updateStorageMutation.error)
    || isUnauthorizedAPIError(runtimeReloadMutation.error)

  const isMutating =
    updateStorageMutation.isPending || runtimeReloadMutation.isPending

  const pageError = useMemo(() => {
    if (unauthorized) {
      return null
    }

    const errors = [
      storageQuery.error,
      storageBackendsQuery.error,
      runtimeStatusQuery.error,
      updateStorageMutation.error,
      runtimeReloadMutation.error,
    ]
    const firstError = errors.find((item) => item !== null)
    return firstError ? describeUnknownError(firstError) : null
  }, [
    runtimeReloadMutation.error,
    runtimeStatusQuery.error,
    storageBackendsQuery.error,
    storageQuery.error,
    unauthorized,
    updateStorageMutation.error,
  ])

  function setStorageDraft(nextDraft: StorageDraft): void {
    setEditedDraft(nextDraft)
    setValidationError(null)
    setActionFeedback(null)
    setTestResult(null)
  }

  const activeBackendOption = useMemo(() => {
    if (!storageQuery.data) {
      return null
    }
    return backendOptions.find((option) => option.backend === storageQuery.data.backend) ?? null
  }, [backendOptions, storageQuery.data])
  const selectedBackendForm = useMemo(() => {
    if (!draft) {
      return null
    }
    if (!isSupportedStorageBackend(draft.backend)) {
      return (
        <p className="error-text">
          This storage backend is not supported by the guided editor. Select a supported backend.
        </p>
      )
    }

    const BackendForm = STORAGE_BACKENDS[draft.backend].component
    return (
      <BackendForm
        config={draft.config}
        onChange={(nextConfig) => {
          setStorageDraft({
            ...draft,
            config: nextConfig,
          })
        }}
      />
    )
  }, [draft])

  const activeRuntimeStatus = runtimeStatusQuery.data

  async function refreshAll(): Promise<void> {
    const [storageResult] = await Promise.all([
      storageQuery.refetch(),
      storageBackendsQuery.refetch(),
      runtimeStatusQuery.refetch(),
    ])
    if (storageResult.data) {
      setEditedDraft(null)
      setSecretInputs({})
      setValidationError(null)
      setTestResult(null)
    }
  }

  async function submitApiKey(apiKey: string): Promise<void> {
    saveApiKey(apiKey)
    await refreshAll()
  }

  async function clearStoredApiKey(): Promise<void> {
    clearApiKey()
    await refreshAll()
  }

  function markPendingReload(message: string): void {
    setHasPendingReload(true)
    setPendingReloadMessage(message)
  }

  function clearPendingReload(): void {
    setHasPendingReload(false)
    setPendingReloadMessage(null)
  }

  function handleSelectBackend(nextBackend: string): void {
    if (!draft || nextBackend === draft.backend || !isSupportedStorageBackend(nextBackend)) {
      return
    }

    const nextMetadata = backendMetadataByName.get(nextBackend) ?? null
    const nextConfig = defaultConfigForBackend(nextBackend, nextMetadata)

    setStorageDraft({
      backend: nextBackend,
      config: cloneStorageConfig(nextConfig),
    })
    setSecretInputs({})
  }

  async function handleSaveChanges(): Promise<void> {
    if (!draft) {
      return
    }

    const isBackendSwitch = baseline !== null && draft.backend !== baseline.backend

    if (isSupportedStorageBackend(draft.backend)) {
      const maybeError = STORAGE_BACKENDS[draft.backend].validate(draft.config)
      if (maybeError) {
        setValidationError(maybeError)
        return
      }
    }

    if (isBackendSwitch) {
      const confirmed = window.confirm(
        'Switch storage backend? This affects new uploads only. Existing clip URIs remain unchanged.',
      )
      if (!confirmed) {
        return
      }
    }

    const baseConfig = baseline?.config ?? {}
    const nonSecretPatch = isBackendSwitch
      ? cloneStorageConfig(draft.config)
      : buildStorageConfigPatch(baseConfig, draft.config, REDACTED_PLACEHOLDER)

    const secretPatch = buildStorageSecretPatch(secretInputs)

    const nextConfigPatch: Record<string, unknown> = {
      ...nonSecretPatch,
      ...secretPatch,
    }

    const payload: StorageUpdate = {}
    if (isBackendSwitch) {
      payload.backend = draft.backend
    }
    if (Object.keys(nextConfigPatch).length > 0) {
      payload.config = nextConfigPatch
    }

    if (!isBackendSwitch && payload.config === undefined) {
      setActionFeedback('No storage changes to apply.')
      return
    }

    setActionFeedback(null)
    setValidationError(null)

    try {
      const response = await updateStorageMutation.mutateAsync({
        payload,
        applyChanges: applyChangesImmediately,
      })

      if (response.storage) {
        await storageQuery.refetch()
        setEditedDraft(null)
      }
      setSecretInputs({})

      if (response.restart_required) {
        markPendingReload('Storage configuration updated. Runtime reload required.')
        setActionFeedback('Storage configuration updated. Apply runtime reload to activate changes.')
        return
      }

      clearPendingReload()
      if (response.runtime_reload) {
        setActionFeedback(`Storage configuration updated. ${response.runtime_reload.message}.`)
        await runtimeStatusQuery.refetch()
        return
      }

      setActionFeedback('Storage configuration updated.')
    } catch (error) {
      setActionFeedback(`Storage update failed: ${describeUnknownError(error)}`)
    }
  }

  async function handleApplyRuntimeReload(): Promise<void> {
    setActionFeedback(null)
    try {
      const response = await runtimeReloadMutation.mutateAsync()
      clearPendingReload()
      setActionFeedback(response.message)
      await runtimeStatusQuery.refetch()
    } catch (error) {
      setActionFeedback(`Runtime reload failed: ${describeUnknownError(error)}`)
    }
  }

  return (
    <section className="page fade-in-up">
      <header className="page__header">
        <div>
          <h1 className="page__title">Storage</h1>
          <p className="page__lead">Configure storage backends and apply runtime reload safely.</p>
        </div>
        <Button
          variant="ghost"
          onClick={() => {
            void refreshAll()
          }}
          disabled={isMutating || storageQuery.isFetching || storageBackendsQuery.isFetching}
        >
          {storageQuery.isFetching || storageBackendsQuery.isFetching ? 'Refreshing...' : 'Refresh'}
        </Button>
      </header>

      {unauthorized ? (
        <Card title="Authentication required">
          <ApiKeyGate
            busy={isMutating || storageQuery.isFetching}
            onSubmit={submitApiKey}
            onClear={clearStoredApiKey}
          />
        </Card>
      ) : null}

      {pageError ? (
        <Card title="Storage operations failed">
          <p className="error-text">{pageError}</p>
        </Card>
      ) : null}

      <Card title="Active backend" subtitle="Current storage target for new uploads">
        {storageQuery.isPending || !storageQuery.data ? (
          <p className="subtle">Loading storage configuration...</p>
        ) : (
          <div className="inline-form">
            <div className="inline-form__actions">
              <StatusBadge tone="unknown">
                {activeBackendOption?.label ?? storageQuery.data.backend}
              </StatusBadge>
              <p className="subtle">Backend id: {storageQuery.data.backend}</p>
            </div>
            <p className="subtle">
              Switching backend affects new uploads only. Existing clip URIs are unchanged.
            </p>
          </div>
        )}
      </Card>

      <Card title="Storage settings" subtitle="Choose backend and update config safely">
        {!draft ? (
          <p className="subtle">Loading form...</p>
        ) : (
          <div className="inline-form">
            <div className="backend-picker__grid">
              {backendOptions.map((option) => {
                const selected = option.backend === draft.backend
                const supported = isSupportedStorageBackend(option.backend)
                return (
                  <button
                    key={option.backend}
                    type="button"
                    className="backend-picker__card"
                    aria-label={option.label}
                    aria-pressed={selected}
                    disabled={isMutating || !supported}
                    onClick={() => {
                      handleSelectBackend(option.backend)
                    }}
                  >
                    <p className="backend-picker__title-text">
                      {option.label}
                      {selected ? ' (selected)' : ''}
                    </p>
                    <p className="backend-picker__description">{option.description}</p>
                    {!supported ? (
                      <p className="subtle">Unsupported in guided editor for this build.</p>
                    ) : null}
                  </button>
                )
              })}
            </div>

            {selectedBackendForm}

            {selectedSecretFields.length > 0 ? (
              <section className="inline-form">
                <h3 className="backend-picker__title">Secret updates</h3>
                <p className="subtle">
                  Secret fields are write-only. Leave blank to keep the current value unchanged.
                </p>
                {selectedSecretFields.map((field) => (
                  <label key={field} className="field-label" htmlFor={`storage-secret-${field}`}>
                    {field}
                    <input
                      id={`storage-secret-${field}`}
                      className="input"
                      type="password"
                      autoComplete="off"
                      value={secretInputs[field] ?? ''}
                      onChange={(event) => {
                        setSecretInputs((current) => ({
                          ...current,
                          [field]: event.target.value,
                        }))
                        setActionFeedback(null)
                        setTestResult(null)
                      }}
                    />
                  </label>
                ))}
              </section>
            ) : null}

            {validationError ? <p className="error-text">{validationError}</p> : null}

            <label className="form-checkbox-field" htmlFor="storage-apply-immediately">
              <input
                id="storage-apply-immediately"
                type="checkbox"
                checked={applyChangesImmediately}
                onChange={(event) => {
                  setApplyChangesImmediately(event.target.checked)
                }}
              />
              Apply changes immediately (runtime reload)
            </label>

            <TestConnectionButton
              request={{
                type: 'storage',
                backend: draft.backend,
                config: validationConfig ?? draft.config,
              }}
              result={testResult}
              onResult={(result) => {
                setTestResult(result)
              }}
              idleLabel="Validate storage"
              retryLabel="Retry validation"
              pendingLabel="Validating..."
              description="Run storage connectivity validation before applying updates."
            />

            <div className="inline-form__actions">
              <Button
                onClick={() => {
                  void handleSaveChanges()
                }}
                disabled={
                  isMutating
                  || !isSupportedStorageBackend(draft.backend)
                }
              >
                {updateStorageMutation.isPending ? 'Saving...' : 'Save storage settings'}
              </Button>
            </div>
          </div>
        )}
      </Card>

      <Card title="Runtime control" subtitle="Apply pending config changes to active runtime">
        {hasPendingReload ? (
          <div className="camera-restart-banner" role="status" aria-live="polite">
            <p className="subtle">
              {pendingReloadMessage ?? 'Storage changes are pending runtime reload.'}
            </p>
            <Button
              onClick={() => {
                void handleApplyRuntimeReload()
              }}
              disabled={runtimeReloadMutation.isPending}
            >
              {runtimeReloadMutation.isPending ? 'Reloading...' : 'Apply runtime reload'}
            </Button>
          </div>
        ) : (
          <p className="subtle">No pending storage reload tasks.</p>
        )}

        {actionFeedback ? <p className="subtle">{actionFeedback}</p> : null}

        {activeRuntimeStatus ? (
          <dl className="camera-runtime-kv">
            <div className="camera-runtime-row">
              <dt>State</dt>
              <dd>
                <StatusBadge tone={runtimeStatusTone(activeRuntimeStatus)}>
                  {runtimeStatusLabel(activeRuntimeStatus.state)}
                </StatusBadge>
              </dd>
            </div>
            <div className="camera-runtime-row">
              <dt>Generation</dt>
              <dd>{activeRuntimeStatus.generation}</dd>
            </div>
            <div className="camera-runtime-row">
              <dt>Last reload</dt>
              <dd>{formatRuntimeTimestamp(activeRuntimeStatus.last_reload_at)}</dd>
            </div>
            <div className="camera-runtime-row">
              <dt>Last reload error</dt>
              <dd>{activeRuntimeStatus.last_reload_error ?? 'none'}</dd>
            </div>
          </dl>
        ) : (
          <p className="subtle">Runtime status unavailable.</p>
        )}
      </Card>
    </section>
  )
}
