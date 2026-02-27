import { useState } from 'react'

import { useSetupTestConnectionMutation } from '../../../api/hooks/useSetupTestConnectionMutation'
import type { TestConnectionRequest, TestConnectionResponse } from '../../../api/generated/types'
import { Button } from '../../../components/ui/Button'
import { StatusBadge } from '../../../components/ui/StatusBadge'

interface TestConnectionStepProps {
  request: TestConnectionRequest
  result: TestConnectionResponse | null
  onResult: (result: TestConnectionResponse) => void
}

function describeMutationError(error: unknown): string {
  if (error instanceof Error && error.message.trim().length > 0) {
    return error.message
  }
  return 'Connection test failed due to an unexpected error.'
}

export function TestConnectionStep({ request, result, onResult }: TestConnectionStepProps) {
  const mutation = useSetupTestConnectionMutation()
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  async function runTest(): Promise<void> {
    setErrorMessage(null)
    try {
      const response = await mutation.mutateAsync(request)
      onResult(response)
    } catch (error) {
      setErrorMessage(describeMutationError(error))
    }
  }

  return (
    <section className="inline-form">
      <h3 className="camera-add-flow__title">Test connection</h3>
      <p className="subtle">
        Run a non-persistent connectivity test before confirming camera creation. You can continue
        even if test fails.
      </p>

      <div className="inline-form__actions">
        <Button
          onClick={() => {
            void runTest()
          }}
          disabled={mutation.isPending}
        >
          {mutation.isPending ? 'Testing...' : result ? 'Retry test' : 'Run connection test'}
        </Button>
      </div>

      {errorMessage ? <p className="error-text">{errorMessage}</p> : null}

      {result ? (
        <div className="camera-add-flow__test-result">
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

