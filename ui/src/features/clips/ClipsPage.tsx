import { useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'

import { clearApiKey, isUnauthorizedAPIError, saveApiKey } from '../../api/client'
import { useCamerasQuery } from '../../api/hooks/useCamerasQuery'
import { useClipsQuery } from '../../api/hooks/useClipsQuery'
import { ApiKeyGate } from '../../components/ui/ApiKeyGate'
import { Button } from '../../components/ui/Button'
import { Card } from '../../components/ui/Card'
import {
  CLIP_LIMIT_OPTIONS,
  CLIP_STATUS_OPTIONS,
  DEFAULT_CLIPS_LIMIT,
  formStateToQuery,
  parseClipsQuery,
  queryToFormState,
  queryToSearchParams,
  queryWithoutCursor,
  type ClipsFilterFormState,
} from './queryParams'
import {
  historyForSignature,
  popCursorHistory,
  pushCursorHistory,
  resetCursorHistory,
  type CursorHistoryState,
} from './cursorState'
import { describeClipError, formatTimestamp, renderDetectedObjects } from './presentation'

interface ClipsFilterPanelProps {
  cameraOptions: string[]
  initialFormState: ClipsFilterFormState
  onApply: (form: ClipsFilterFormState) => void
  onReset: () => void
}

function ClipsFilterPanel({
  cameraOptions,
  initialFormState,
  onApply,
  onReset,
}: ClipsFilterPanelProps) {
  const [formState, setFormState] = useState<ClipsFilterFormState>(initialFormState)

  function updateFormState<Key extends keyof ClipsFilterFormState>(
    key: Key,
    value: ClipsFilterFormState[Key],
  ): void {
    setFormState((current) => ({
      ...current,
      [key]: value,
    }))
  }

  return (
    <Card title="Filters">
      <div className="clips-filter-grid">
        <label className="field-label">
          Camera
          <select
            className="input"
            value={formState.camera}
            onChange={(event) => updateFormState('camera', event.target.value)}
          >
            <option value="">Any</option>
            {cameraOptions.map((cameraName) => (
              <option key={cameraName} value={cameraName}>
                {cameraName}
              </option>
            ))}
          </select>
        </label>

        <label className="field-label">
          Status
          <select
            className="input"
            value={formState.status}
            onChange={(event) =>
              updateFormState('status', event.target.value as ClipsFilterFormState['status'])
            }
          >
            <option value="">Any</option>
            {CLIP_STATUS_OPTIONS.map((status) => (
              <option key={status} value={status}>
                {status}
              </option>
            ))}
          </select>
        </label>

        <label className="field-label">
          Alerted
          <select
            className="input"
            value={formState.alerted}
            onChange={(event) =>
              updateFormState('alerted', event.target.value as ClipsFilterFormState['alerted'])
            }
          >
            <option value="any">Any</option>
            <option value="true">true</option>
            <option value="false">false</option>
          </select>
        </label>

        <label className="field-label">
          Detected
          <select
            className="input"
            value={formState.detected}
            onChange={(event) =>
              updateFormState('detected', event.target.value as ClipsFilterFormState['detected'])
            }
          >
            <option value="any">Any</option>
            <option value="true">true</option>
            <option value="false">false</option>
          </select>
        </label>

        <label className="field-label">
          Risk level
          <input
            className="input"
            value={formState.riskLevel}
            onChange={(event) => updateFormState('riskLevel', event.target.value)}
            placeholder="high"
          />
        </label>

        <label className="field-label">
          Activity type
          <input
            className="input"
            value={formState.activityType}
            onChange={(event) => updateFormState('activityType', event.target.value)}
            placeholder="package"
          />
        </label>

        <label className="field-label">
          Since
          <input
            className="input"
            type="datetime-local"
            value={formState.sinceLocal}
            onChange={(event) => updateFormState('sinceLocal', event.target.value)}
          />
        </label>

        <label className="field-label">
          Until
          <input
            className="input"
            type="datetime-local"
            value={formState.untilLocal}
            onChange={(event) => updateFormState('untilLocal', event.target.value)}
          />
        </label>

        <label className="field-label">
          Limit
          <select
            className="input"
            value={String(formState.limit)}
            onChange={(event) => updateFormState('limit', Number.parseInt(event.target.value, 10))}
          >
            {CLIP_LIMIT_OPTIONS.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="inline-form__actions">
        <Button onClick={() => onApply(formState)}>Apply filters</Button>
        <Button variant="ghost" onClick={onReset}>
          Reset
        </Button>
      </div>
    </Card>
  )
}

export function ClipsPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const query = useMemo(() => parseClipsQuery(searchParams), [searchParams])
  const filterQuery = useMemo(() => queryWithoutCursor(query), [query])
  const filterSignature = useMemo(
    () => queryToSearchParams(filterQuery).toString(),
    [filterQuery],
  )
  const initialFormState = useMemo(() => queryToFormState(filterQuery), [filterQuery])
  const [cursorState, setCursorState] = useState<CursorHistoryState>(() =>
    resetCursorHistory(filterSignature),
  )
  const cursorHistory = historyForSignature(cursorState, filterSignature)

  const clipsQuery = useClipsQuery(query)
  const camerasQuery = useCamerasQuery()

  const unauthorized = isUnauthorizedAPIError(clipsQuery.error)
  const clipItems = useMemo(() => clipsQuery.data?.clips ?? [], [clipsQuery.data])
  const cameraOptions = useMemo(() => {
    const uniqueNames = new Set<string>()
    for (const camera of camerasQuery.data ?? []) {
      uniqueNames.add(camera.name)
    }
    for (const clip of clipItems) {
      uniqueNames.add(clip.camera)
    }
    if (query.camera) {
      uniqueNames.add(query.camera)
    }
    return [...uniqueNames].sort((left, right) => left.localeCompare(right))
  }, [camerasQuery.data, clipItems, query.camera])

  function clearCursorHistory(nextSignature: string): void {
    setCursorState(resetCursorHistory(nextSignature))
  }

  function applyFilters(formState: ClipsFilterFormState): void {
    const nextQuery = formStateToQuery(formState)
    const nextParams = queryToSearchParams(queryWithoutCursor(nextQuery))
    clearCursorHistory(nextParams.toString())
    setSearchParams(nextParams)
  }

  function resetFilters(): void {
    const resetQuery = {
      limit: DEFAULT_CLIPS_LIMIT,
    }
    const nextParams = queryToSearchParams(resetQuery)
    clearCursorHistory(nextParams.toString())
    setSearchParams(nextParams)
  }

  async function submitApiKey(apiKey: string): Promise<void> {
    saveApiKey(apiKey)
    await Promise.all([clipsQuery.refetch(), camerasQuery.refetch()])
  }

  async function clearStoredApiKey(): Promise<void> {
    clearApiKey()
    await Promise.all([clipsQuery.refetch(), camerasQuery.refetch()])
  }

  async function refreshClips(): Promise<void> {
    await Promise.all([clipsQuery.refetch(), camerasQuery.refetch()])
  }

  function goToNextCursor(): void {
    const nextCursor = clipsQuery.data?.next_cursor
    if (!nextCursor) {
      return
    }
    setCursorState((current) => pushCursorHistory(current, filterSignature, query.cursor))
    setSearchParams(
      queryToSearchParams({
        ...query,
        cursor: nextCursor,
      }),
    )
  }

  function goToPreviousCursor(): void {
    const popped = popCursorHistory(cursorState, filterSignature)
    if (popped.previousCursor === undefined && cursorHistory.length === 0) {
      return
    }
    setCursorState(popped.state)
    setSearchParams(
      queryToSearchParams({
        ...query,
        cursor: popped.previousCursor,
      }),
    )
  }

  return (
    <section className="page fade-in-up">
      <header className="page__header">
        <div>
          <h1 className="page__title">Clips</h1>
          <p className="page__lead">Filter and page through clips with URL-synced state.</p>
        </div>
        <Button variant="ghost" onClick={refreshClips} disabled={clipsQuery.isFetching}>
          {clipsQuery.isFetching ? 'Refreshing...' : 'Refresh'}
        </Button>
      </header>

      <ClipsFilterPanel
        key={filterSignature}
        cameraOptions={cameraOptions}
        initialFormState={initialFormState}
        onApply={applyFilters}
        onReset={resetFilters}
      />

      <Card title="Results" subtitle={clipItems.length > 0 ? `${clipItems.length} clip(s)` : undefined}>
        {clipsQuery.isPending ? (
          <p className="muted">Loading clips...</p>
        ) : null}

        {unauthorized ? (
          <ApiKeyGate
            busy={clipsQuery.isFetching}
            onSubmit={submitApiKey}
            onClear={clearStoredApiKey}
          />
        ) : null}

        {clipsQuery.error && !unauthorized ? (
          <p className="error-text">{describeClipError(clipsQuery.error)}</p>
        ) : null}

        {!clipsQuery.isPending && !clipsQuery.error && clipItems.length === 0 ? (
          <p className="muted">No clips match the current filters.</p>
        ) : null}

        {clipItems.length > 0 ? (
          <>
            <div className="clips-table-wrap">
              <table className="clips-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Camera</th>
                    <th>Status</th>
                    <th>Created</th>
                    <th>Activity</th>
                    <th>Risk</th>
                    <th>Alerted</th>
                    <th>Detected</th>
                  </tr>
                </thead>
                <tbody>
                  {clipItems.map((clip) => (
                    <tr key={clip.id}>
                      <td>
                        <Link to={`/clips/${clip.id}`}>{clip.id}</Link>
                      </td>
                      <td>{clip.camera}</td>
                      <td>
                        <span className="clips-chip">{clip.status}</span>
                      </td>
                      <td>{formatTimestamp(clip.created_at)}</td>
                      <td>{clip.activity_type ?? '-'}</td>
                      <td>{clip.risk_level ?? '-'}</td>
                      <td>{clip.alerted ? 'true' : 'false'}</td>
                      <td>{renderDetectedObjects(clip)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <ul className="clips-mobile-list">
              {clipItems.map((clip) => (
                <li key={clip.id} className="clips-mobile-item">
                  <Link to={`/clips/${clip.id}`} className="clips-mobile-id">
                    {clip.id}
                  </Link>
                  <p className="muted">{clip.camera}</p>
                  <p className="muted">{formatTimestamp(clip.created_at)}</p>
                  <div className="clips-mobile-meta">
                    <span className="clips-chip">{clip.status}</span>
                    <span className="clips-chip">{clip.activity_type ?? 'n/a'}</span>
                    <span className="clips-chip">{clip.risk_level ?? 'n/a'}</span>
                    <span className="clips-chip">{clip.alerted ? 'alerted' : 'no alert'}</span>
                  </div>
                </li>
              ))}
            </ul>

            <div className="clips-pagination">
              <Button
                variant="ghost"
                onClick={goToPreviousCursor}
                disabled={cursorHistory.length === 0 || clipsQuery.isFetching}
              >
                Previous
              </Button>
              <p className="subtle">
                {clipsQuery.data?.has_more ? 'More results available' : 'End of results'}
              </p>
              <Button
                variant="ghost"
                onClick={goToNextCursor}
                disabled={!clipsQuery.data?.next_cursor || clipsQuery.isFetching}
              >
                Next
              </Button>
            </div>
          </>
        ) : null}
      </Card>
    </section>
  )
}
