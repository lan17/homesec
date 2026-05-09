import { useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'

import { clearApiKey, isUnauthorizedAPIError, saveApiKey } from '../../api/client'
import { useCamerasQuery } from '../../api/hooks/useCamerasQuery'
import { useClipsQuery } from '../../api/hooks/useClipsQuery'
import { ApiKeyGate } from '../../components/ui/ApiKeyGate'
import { Button } from '../../components/ui/Button'
import { EmptyState } from '../../components/ui/EmptyState'
import { EventCard } from '../../components/ui/EventCard'
import { FilterChips, type FilterChip } from '../../components/ui/FilterChips'
import { MediaPanel } from '../../components/ui/MediaPanel'
import { RiskBadge } from '../../components/ui/RiskBadge'
import { StatusBadge } from '../../components/ui/StatusBadge'
import {
  CLIP_LIMIT_OPTIONS,
  CLIP_STATUS_OPTIONS,
  DEFAULT_CLIPS_LIMIT,
  DEFAULT_DETECTED_FILTER,
  formStateToSearchParams,
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
import {
  describeClipError,
  formatActivityType,
  formatDetectedObjects,
  formatEventTime,
  groupClipsByDate,
} from './presentation'

interface ClipsFilterPanelProps {
  cameraOptions: string[]
  initialFormState: ClipsFilterFormState
  onApply: (form: ClipsFilterFormState) => void
  onClose: () => void
  onReset: () => void
}

function ClipsFilterPanel({
  cameraOptions,
  initialFormState,
  onApply,
  onClose,
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
    <section className="advanced-filters" aria-label="More event filters">
      <header className="advanced-filters__header">
        <div>
          <h2 className="section-title">More filters</h2>
          <p className="muted">Use detailed filters when the quick chips are not enough.</p>
        </div>
        <Button variant="ghost" onClick={onClose}>Close</Button>
      </header>
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
          Processing status
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
          Alert status
          <select
            className="input"
            value={formState.alerted}
            onChange={(event) =>
              updateFormState('alerted', event.target.value as ClipsFilterFormState['alerted'])
            }
          >
            <option value="any">Any</option>
            <option value="true">Alert sent</option>
            <option value="false">No alert</option>
          </select>
        </label>

        <label className="field-label">
          Detection
          <select
            className="input"
            value={formState.detected}
            onChange={(event) =>
              updateFormState('detected', event.target.value as ClipsFilterFormState['detected'])
            }
          >
            <option value="any">Any</option>
            <option value="true">Something detected</option>
            <option value="false">No detection</option>
          </select>
        </label>

        <label className="field-label">
          Risk
          <input
            className="input"
            value={formState.riskLevel}
            onChange={(event) => updateFormState('riskLevel', event.target.value)}
            placeholder="High, medium, critical"
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
          Results per page
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
    </section>
  )
}

function eventDetailPath(clipId: string, routeSearch: string): string {
  const suffix = routeSearch ? `?${routeSearch}` : ''
  return `/events/${encodeURIComponent(clipId)}${suffix}`
}

function startOfTodayIso(): string {
  const date = new Date()
  date.setHours(0, 0, 0, 0)
  return date.toISOString()
}

function isTodayFilterActive(since: string | null | undefined): boolean {
  if (!since) {
    return false
  }
  const sinceDate = new Date(since)
  if (Number.isNaN(sinceDate.valueOf())) {
    return false
  }
  const today = new Date()
  return (
    sinceDate.getFullYear() === today.getFullYear()
    && sinceDate.getMonth() === today.getMonth()
    && sinceDate.getDate() === today.getDate()
  )
}

export function ClipsPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [showAdvancedFilters, setShowAdvancedFilters] = useState(false)
  const routeSearch = searchParams.toString()
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
  const eventGroups = useMemo(() => groupClipsByDate(clipItems), [clipItems])
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
    const nextParams = formStateToSearchParams(formState)
    clearCursorHistory(queryToSearchParams(queryWithoutCursor(nextQuery)).toString())
    setSearchParams(nextParams)
    setShowAdvancedFilters(false)
  }

  function resetFilters(): void {
    const resetQuery = {
      detected: DEFAULT_DETECTED_FILTER,
      limit: DEFAULT_CLIPS_LIMIT,
    }
    const nextParams = queryToSearchParams(resetQuery)
    clearCursorHistory(nextParams.toString())
    setSearchParams(nextParams)
    setShowAdvancedFilters(false)
  }

  function applyQuery(nextQuery: ReturnType<typeof queryWithoutCursor>): void {
    const nextParams = queryToRouteSearchParams(queryWithoutCursor(nextQuery))
    clearCursorHistory(nextParams.toString())
    setSearchParams(nextParams)
  }

  function toggleQueryValue(
    field: 'alerted' | 'activity_type' | 'risk_level',
    value: boolean | string,
  ): void {
    const currentValue = query[field]
    applyQuery({
      ...query,
      cursor: undefined,
      [field]: currentValue === value ? undefined : value,
    })
  }

  function toggleTodayFilter(): void {
    applyQuery({
      ...query,
      cursor: undefined,
      since: isTodayFilterActive(query.since) ? undefined : startOfTodayIso(),
      until: undefined,
    })
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

  function queryToRouteSearchParams(nextQuery: ReturnType<typeof queryWithoutCursor>): URLSearchParams {
    const params = queryToSearchParams(nextQuery)
    if (query.detected === undefined && searchParams.get('detected') === 'any') {
      params.set('detected', 'any')
    }
    return params
  }

  function goToNextCursor(): void {
    const nextCursor = clipsQuery.data?.next_cursor
    if (!nextCursor) {
      return
    }
    setCursorState((current) => pushCursorHistory(current, filterSignature, query.cursor))
    setSearchParams(queryToRouteSearchParams({ ...query, cursor: nextCursor }))
  }

  function goToPreviousCursor(): void {
    const popped = popCursorHistory(cursorState, filterSignature)
    if (popped.previousCursor === undefined && cursorHistory.length === 0) {
      return
    }
    setCursorState(popped.state)
    setSearchParams(queryToRouteSearchParams({ ...query, cursor: popped.previousCursor }))
  }

  const quickFilterChips: FilterChip[] = [
    { id: 'today', label: 'Today', active: isTodayFilterActive(query.since) },
    { id: 'alerts', label: 'Alerts', active: query.alerted === true },
    { id: 'people', label: 'People', active: query.activity_type === 'person' },
    { id: 'vehicles', label: 'Vehicles', active: query.activity_type === 'vehicle' },
    { id: 'packages', label: 'Packages', active: query.activity_type === 'package' },
    { id: 'high-risk', label: 'High risk', active: query.risk_level === 'high' },
    { id: 'more', label: 'More filters', active: showAdvancedFilters },
  ]

  function selectQuickFilter(id: string): void {
    switch (id) {
      case 'today':
        toggleTodayFilter()
        break
      case 'alerts':
        toggleQueryValue('alerted', true)
        break
      case 'people':
        toggleQueryValue('activity_type', 'person')
        break
      case 'vehicles':
        toggleQueryValue('activity_type', 'vehicle')
        break
      case 'packages':
        toggleQueryValue('activity_type', 'package')
        break
      case 'high-risk':
        toggleQueryValue('risk_level', 'high')
        break
      case 'more':
        setShowAdvancedFilters((current) => !current)
        break
      default:
        break
    }
  }

  return (
    <section className="page fade-in-up">
      <header className="page__header">
        <div>
          <h1 className="page__title">Events</h1>
          <p className="page__lead">Filter and page through recorded security events.</p>
        </div>
        <Button variant="ghost" onClick={refreshClips} disabled={clipsQuery.isFetching}>
          {clipsQuery.isFetching ? 'Refreshing...' : 'Refresh'}
        </Button>
      </header>

      <section className="events-filter-bar" aria-label="Event filters">
        <FilterChips chips={quickFilterChips} onSelect={selectQuickFilter} />
      </section>

      {showAdvancedFilters ? (
        <div className="filters-sheet" role="dialog" aria-modal="false" aria-label="More filters">
          <ClipsFilterPanel
            key={filterSignature}
            cameraOptions={cameraOptions}
            initialFormState={initialFormState}
            onApply={applyFilters}
            onClose={() => setShowAdvancedFilters(false)}
            onReset={resetFilters}
          />
        </div>
      ) : null}

      <section className="events-results" aria-labelledby="events-results-title">
        <header className="events-results__header">
          <div>
            <h2 id="events-results-title" className="section-title">Results</h2>
            {clipItems.length > 0 ? (
              <p className="muted">{clipItems.length} event(s)</p>
            ) : null}
          </div>
        </header>

        {unauthorized ? (
          <ApiKeyGate
            busy={clipsQuery.isFetching}
            onSubmit={submitApiKey}
            onClear={clearStoredApiKey}
          />
        ) : null}

        {clipsQuery.isPending ? (
          <EmptyState
            title="Loading events"
            description="Fetching recorded security events with the current filters."
            tone="loading"
          />
        ) : null}

        {clipsQuery.error && !unauthorized ? (
          <EmptyState
            title="Events could not load"
            description={describeClipError(clipsQuery.error)}
            tone="error"
          />
        ) : null}

        {!clipsQuery.isPending && !clipsQuery.error && clipItems.length === 0 ? (
          <EmptyState
            title="No events found"
            description="No recorded security events match the current filters."
          />
        ) : null}

        {clipItems.length > 0 ? (
          <>
            <div className="event-groups">
              {eventGroups.map((group) => (
                <section
                  key={group.key}
                  className="event-group"
                  aria-labelledby={`events-${group.key}`}
                >
                  <h3 id={`events-${group.key}`} className="event-group__title">
                    {group.label}
                  </h3>
                  <div className="event-list">
                    {group.clips.map((clip) => (
                      <EventCard
                        key={clip.id}
                        camera={clip.camera}
                        time={formatEventTime(clip.created_at)}
                        title={formatActivityType(clip.activity_type)}
                        summary={clip.summary?.trim() || 'No summary available yet.'}
                        media={(
                          <MediaPanel
                            aspect="video"
                            className="event-card__media-panel"
                            placeholder="Open event to view video"
                          />
                        )}
                        risk={<RiskBadge level={clip.risk_level} />}
                        status={(
                          <StatusBadge tone={clip.alerted ? 'unhealthy' : 'unknown'}>
                            {clip.alerted ? 'Alert sent' : 'No alert'}
                          </StatusBadge>
                        )}
                        meta={[
                          { label: 'Objects', value: formatDetectedObjects(clip) },
                        ]}
                        actions={(
                          <Link
                            to={eventDetailPath(clip.id, routeSearch)}
                            className="button button--primary"
                          >
                            View event
                          </Link>
                        )}
                      />
                    ))}
                  </div>
                </section>
              ))}
            </div>

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
      </section>
    </section>
  )
}
