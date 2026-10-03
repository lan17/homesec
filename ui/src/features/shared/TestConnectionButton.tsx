import { useEffect, useRef, useState } from 'react'

import type {
  TestConnectionRequest,
  TestConnectionResponse,
} from '../../api/generated/types'
import { useSetupTestConnectionMutation } from '../../api/hooks/useSetupTestConnectionMutation'
import { Button } from '../../components/ui/Button'
import { StatusBadge } from '../../components/ui/StatusBadge'
import { validateEnvReferences } from './envReferences'

interface TestConnectionButtonProps {
  request: TestConnectionRequest
  result: TestConnectionResponse | null
  onResult: (result: TestConnectionResponse) => void
  idleLabel?: string
  retryLabel?: string
  pendingLabel?: string
  description?: string
}

function describeMutationError(error: unknown): string {
  if (error instanceof Error && error.message.trim().length > 0) {
    return error.message
  }
  return 'Connection test failed due to an unexpected error.'
}

export function TestConnectionButton({
  request,
  result,
  onResult,
  idleLabel = 'Run connection test',
  retryLabel = 'Retry test',
  pendingLabel = 'Testing...',
  description,
}: TestConnectionButtonProps) {
  const mutation = useSetupTestConnectionMutation()
  const [error, setError] = useState<{ requestKey: string; signal: AbortSignal; message: string } | null>(null)
  const controllerRef = useRef<AbortController | null>(null)
  const requestKey = JSON.stringify(request)
  const errorMessage = error?.requestKey === requestKey && !error.signal.aborted ? error.message : null

  useEffect(() => () => { controllerRef.current?.abort() }, [requestKey])

  async function runTest(): Promise<void> {
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    setError(null)
    try {
      // Validate before mutation variables can retain a pasted raw credential.
      validateEnvReferences(request.config)
      const response = await mutation.mutateAsync({ request, signal: controller.signal })
      if (!controller.signal.aborted) { onResult(response) }
    } catch (error) {
      if (!controller.signal.aborted) {
        setError({ requestKey, signal: controller.signal, message: describeMutationError(error) })
      }
    }
  }

  return (
    <section className="inline-form">
      {description ? <p className="subtle">{description}</p> : null}

      <div className="inline-form__actions">
        <Button
          onClick={() => {
            void runTest()
          }}
          disabled={mutation.isPending}
        >
          {mutation.isPending ? pendingLabel : result ? retryLabel : idleLabel}
        </Button>
      </div>

      {errorMessage ? <p className="error-text">{errorMessage}</p> : null}

      {result ? (
        <div className="test-connection__result">
          <StatusBadge tone={result.success ? 'healthy' : 'unhealthy'}>
            {result.success ? 'PASS' : 'FAIL'}
          </StatusBadge>
          <p className={result.success ? 'subtle' : 'error-text'}>{result.message}</p>
          {typeof result.latency_ms === 'number' ? (
            <p className="subtle">Latency: {result.latency_ms.toFixed(1)} ms</p>
          ) : null}
        </div>
      ) : null}
    </section>
  )
}
