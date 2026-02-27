import { useEffect, useMemo, useRef, useState } from 'react'

import { apiClient, isAPIError } from '../../../api/client'
import { useFinalizeMutation } from '../../../api/hooks/useFinalizeMutation'
import type { FinalizeRequest } from '../../../api/generated/types'
import { Button } from '../../../components/ui/Button'
import {
  buildFinalizeRequestFromDrafts,
  buildReviewSectionSummaries,
  type ReviewSectionStepId,
  type ReviewWizardDrafts,
} from '../review'
import { LaunchProgress, type LaunchProgressStatus } from './LaunchProgress'
import { ReviewSummaryCard } from './ReviewSummaryCard'

const LAUNCH_POLL_INTERVAL_MS = 2_000
const LAUNCH_POLL_TIMEOUT_MS = 30_000

interface ReviewStepProps {
  wizardData: ReviewWizardDrafts
  skippedSteps: ReadonlySet<string>
  onGoToStep: (stepId: ReviewSectionStepId) => void
  onLaunchSuccess: () => void
  onGoDashboard: () => void
}

function _abortError(): Error {
  const error = new Error('Aborted')
  error.name = 'AbortError'
  return error
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError'
}

function sleepWithAbort(ms: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) {
    return Promise.reject(_abortError())
  }

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      signal.removeEventListener('abort', onAbort)
      resolve()
    }, ms)

    function onAbort() {
      clearTimeout(timeout)
      signal.removeEventListener('abort', onAbort)
      reject(_abortError())
    }

    signal.addEventListener('abort', onAbort, { once: true })
  })
}

function toLaunchErrorMessage(error: unknown): string {
  if (isAPIError(error)) {
    if (error.errorCode) {
      return `${error.message} (${error.errorCode})`
    }
    return error.message
  }
  if (error instanceof Error && error.message.trim().length > 0) {
    return error.message
  }
  return 'Launch failed unexpectedly.'
}

interface WaitForPipelineStartResult {
  started: boolean
  cancelled: boolean
  error: string | null
}

async function waitForPipelineStart(
  signal: AbortSignal,
): Promise<WaitForPipelineStartResult> {
  const deadline = Date.now() + LAUNCH_POLL_TIMEOUT_MS
  let lastPollError: unknown = null

  while (Date.now() < deadline) {
    if (signal.aborted) {
      return { started: false, cancelled: true, error: null }
    }

    try {
      const status = await apiClient.getSetupStatus({ signal })
      if (status.pipeline_running) {
        return { started: true, cancelled: false, error: null }
      }
    } catch (error) {
      if (isAbortError(error)) {
        return { started: false, cancelled: true, error: null }
      }
      lastPollError = error
    }

    try {
      await sleepWithAbort(LAUNCH_POLL_INTERVAL_MS, signal)
    } catch (error) {
      if (isAbortError(error)) {
        return { started: false, cancelled: true, error: null }
      }
      throw error
    }
  }

  if (lastPollError) {
    return {
      started: false,
      cancelled: false,
      error: `Timed out waiting for pipeline startup. Last check error: ${toLaunchErrorMessage(lastPollError)}`,
    }
  }
  return {
    started: false,
    cancelled: false,
    error: 'Timed out waiting for pipeline startup.',
  }
}

export function ReviewStep({
  wizardData,
  skippedSteps,
  onGoToStep,
  onLaunchSuccess,
  onGoDashboard,
}: ReviewStepProps) {
  const finalizeMutation = useFinalizeMutation()
  const summaries = useMemo(
    () => buildReviewSectionSummaries(wizardData, skippedSteps),
    [wizardData, skippedSteps],
  )
  const [launchStatus, setLaunchStatus] = useState<LaunchProgressStatus | null>(null)
  const [launchError, setLaunchError] = useState<string | null>(null)
  const isMountedRef = useRef(true)
  const launchAbortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    return () => {
      isMountedRef.current = false
      launchAbortRef.current?.abort()
      launchAbortRef.current = null
    }
  }, [])

  async function handleLaunch(): Promise<void> {
    launchAbortRef.current?.abort()
    const launchAbortController = new AbortController()
    launchAbortRef.current = launchAbortController

    setLaunchError(null)
    setLaunchStatus('launching')

    try {
      const payload = buildFinalizeRequestFromDrafts(wizardData)
      const precheckPayload: FinalizeRequest = { ...payload, validate_only: true }
      const precheckResult = await finalizeMutation.mutateAsync({
        payload: precheckPayload,
        signal: launchAbortController.signal,
      })
      if (!precheckResult.success) {
        const combinedError = precheckResult.errors.length > 0
          ? precheckResult.errors.join('; ')
          : 'Setup finalize failed.'
        if (!isMountedRef.current) {
          return
        }
        setLaunchStatus('failed')
        setLaunchError(combinedError)
        return
      }

      const result = await finalizeMutation.mutateAsync({
        payload: { ...payload, validate_only: false },
        signal: launchAbortController.signal,
      })
      if (!result.success) {
        const combinedError = result.errors.length > 0
          ? result.errors.join('; ')
          : 'Setup finalize failed.'
        if (!isMountedRef.current) {
          return
        }
        setLaunchStatus('failed')
        setLaunchError(combinedError)
        return
      }

      const pollResult = await waitForPipelineStart(launchAbortController.signal)
      if (!isMountedRef.current) {
        return
      }
      if (pollResult.cancelled) {
        return
      }
      if (pollResult.started) {
        setLaunchStatus('started')
        setLaunchError(null)
        onLaunchSuccess()
        return
      }

      setLaunchStatus('failed')
      setLaunchError(pollResult.error ?? 'Pipeline failed to start.')
    } catch (error) {
      if (isAbortError(error)) {
        return
      }
      if (!isMountedRef.current) {
        return
      }
      setLaunchStatus('failed')
      setLaunchError(toLaunchErrorMessage(error))
    } finally {
      if (launchAbortRef.current === launchAbortController) {
        launchAbortRef.current = null
      }
    }
  }

  const launchPending = finalizeMutation.isPending || launchStatus === 'launching'
  const canEdit = !launchPending

  return (
    <section className="wizard-step-card review-step">
      <header className="review-step__header">
        <h2 className="review-step__title">Review configuration before launch</h2>
        <p className="subtle">
          Inspect each section and jump back to edit as needed. Launch writes config and restarts
          HomeSec.
        </p>
      </header>

      <div className="review-step__grid">
        {summaries.map((summary) => (
          <ReviewSummaryCard
            key={summary.stepId}
            title={summary.title}
            status={summary.status}
            items={summary.items}
            emptyMessage={summary.emptyMessage}
            editDisabled={!canEdit}
            onEdit={() => {
              onGoToStep(summary.stepId)
            }}
          />
        ))}
      </div>

      {launchStatus !== null ? (
        <LaunchProgress status={launchStatus} error={launchError} />
      ) : null}

      <div className="inline-form__actions">
        {launchStatus === 'started' ? (
          <Button onClick={onGoDashboard}>Go to Dashboard</Button>
        ) : (
          <Button onClick={() => void handleLaunch()} disabled={launchPending}>
            {launchPending
              ? 'Launching...'
              : launchStatus === 'failed'
                ? 'Retry launch'
                : 'Launch pipeline'}
          </Button>
        )}
      </div>
    </section>
  )
}
