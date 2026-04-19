import { useMemo, useState } from 'react'

import { useStorageBackendsQuery } from '../../../api/hooks/useStorageBackendsQuery'
import { Button } from '../../../components/ui/Button'
import { StorageConfigForm } from '../../settings/storage/StorageConfigForm'
import { STORAGE_BACKENDS } from '../../settings/storage/backends'
import {
  cloneStorageConfig,
  defaultConfigForBackend,
  getStorageBackendMetadata,
  sameJsonValue,
} from '../../settings/storage/editorModel'
import type { StorageFormState } from '../../settings/storage/types'
import { buildStorageTestRequest } from '../../settings/storage/types'
import { TestConnectionButton } from '../../shared/TestConnectionButton'
import type { TestConnectionResponse } from '../../../api/generated/types'

interface StorageStepProps {
  initialData: StorageFormState | null
  onComplete: () => void
  onUpdateData: (data: StorageFormState) => void
  onSkip: () => void
}

const DEFAULT_STORAGE_FORM_STATE: StorageFormState = {
  backend: 'local',
  config: cloneStorageConfig(STORAGE_BACKENDS.local.defaultConfig),
}

function normalizeStorageFormState(value: StorageFormState | null): StorageFormState {
  if (!value) {
    return DEFAULT_STORAGE_FORM_STATE
  }
  const backend = value.backend in STORAGE_BACKENDS ? value.backend : 'local'
  return {
    backend,
    config: value.config,
  }
}

export function StorageStep({
  initialData,
  onComplete,
  onUpdateData,
  onSkip,
}: StorageStepProps) {
  const storageBackendsQuery = useStorageBackendsQuery()
  const [value, setValue] = useState<StorageFormState>(() => normalizeStorageFormState(initialData))
  const [result, setResult] = useState<TestConnectionResponse | null>(null)
  const [validationError, setValidationError] = useState<string | null>(null)
  const effectiveValue = useMemo(() => {
    if (initialData !== null) {
      return value
    }

    const metadata = getStorageBackendMetadata(storageBackendsQuery.data, value.backend)
    if (!metadata) {
      return value
    }

    const builtInDefaults = defaultConfigForBackend(value.backend, null)
    if (!sameJsonValue(value.config, builtInDefaults)) {
      return value
    }

    return {
      ...value,
      config: defaultConfigForBackend(value.backend, metadata),
    }
  }, [initialData, storageBackendsQuery.data, value])

  const backendDef = STORAGE_BACKENDS[effectiveValue.backend]
  const testRequest = useMemo(() => buildStorageTestRequest(effectiveValue), [effectiveValue])

  function handleSaveAndContinue(): void {
    const maybeError = backendDef.validate(effectiveValue.config)
    if (maybeError) {
      setValidationError(maybeError)
      return
    }
    setValidationError(null)
    onUpdateData(effectiveValue)
    onComplete()
  }

  return (
    <section className="wizard-step-card">
      <StorageConfigForm
        value={effectiveValue}
        backends={storageBackendsQuery.data}
        onChange={(nextValue) => {
          setValue(nextValue)
          setResult(null)
          setValidationError(null)
        }}
      />

      {validationError ? <p className="error-text">{validationError}</p> : null}

      <TestConnectionButton
        request={testRequest}
        result={result}
        onResult={setResult}
        idleLabel="Test upload"
        retryLabel="Retry upload test"
        pendingLabel="Testing upload..."
        description="Validate storage connectivity before moving on."
      />

      <div className="inline-form__actions">
        <Button variant="ghost" onClick={onSkip}>
          Skip storage step
        </Button>
        <Button onClick={handleSaveAndContinue}>Save and continue</Button>
      </div>
    </section>
  )
}
