import type { TestConnectionRequest, TestConnectionResponse } from '../../../api/generated/types'
import { TestConnectionButton } from '../../shared/TestConnectionButton'

interface TestConnectionStepProps {
  request: TestConnectionRequest
  result: TestConnectionResponse | null
  onResult: (result: TestConnectionResponse) => void
}

export function TestConnectionStep({ request, result, onResult }: TestConnectionStepProps) {
  return (
    <section className="inline-form">
      <h3 className="camera-add-flow__title">Test connection</h3>
      <p className="subtle">
        Run a non-persistent connectivity test before confirming camera creation. You can continue
        even if test fails.
      </p>
      <TestConnectionButton request={request} result={result} onResult={onResult} />
    </section>
  )
}
