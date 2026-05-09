import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'

import { clearApiKey, isUnauthorizedAPIError, saveApiKey } from '../../api/client'
import type { ClipResponse } from '../../api/generated/types'
import { useCamerasQuery } from '../../api/hooks/useCamerasQuery'
import { useClipMediaUrl } from '../../api/hooks/useClipMediaUrl'
import { useClipsQuery } from '../../api/hooks/useClipsQuery'
import { ApiKeyGate } from '../../components/ui/ApiKeyGate'
import { Button } from '../../components/ui/Button'
import { EmptyState } from '../../components/ui/EmptyState'
import { EventCard } from '../../components/ui/EventCard'
import { FilterChips, type FilterChip } from '../../components/ui/FilterChips'
import { MediaPanel } from '../../components/ui/MediaPanel'
import { RiskBadge } from '../../components/ui/RiskBadge'
import { riskLabelForLevel } from '../../components/ui/riskTone'
import { StatusBadge } from '../../components/ui/StatusBadge'
import { TechnicalDetailsDisclosure } from '../../components/ui/TechnicalDetailsDisclosure'
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
  formatAlertStatus,
  formatActivityType,
  formatDetectedObjects,
  formatEventCount,
  formatEventTime,
  formatTimestamp,
  groupClipsByDate,
  resolveClipExternalLink,
  resolveClipViewLink,
} from './presentation'
import {
  nextAttemptsAfterMediaSourceChange,
  shouldRefreshPlaybackSource,
} from './playbackRetry'

interface ClipsFilterPanelProps {
  initialFormState: ClipsFilterFormState
  onApply: (form: ClipsFilterFormState) => void
  onClose: () => void
  onReset: () => void
}

function ClipsFilterPanel({
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
                {formatActivityType(status)}
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
            <option value="false">{formatAlertStatus(false)}</option>
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

function hasActiveEventFilters(query: ReturnType<typeof queryWithoutCursor>): boolean {
  return Boolean(
    query.camera
    || query.status
    || typeof query.alerted === 'boolean'
    || query.detected !== DEFAULT_DETECTED_FILTER
    || query.risk_level
    || query.activity_type
    || query.since
    || query.until
    || query.limit !== DEFAULT_CLIPS_LIMIT,
  )
}

interface EventListCardProps {
  clip: ClipResponse
}

function EventListCard({ clip }: EventListCardProps) {
  const [showDetails, setShowDetails] = useState(false)
  const mediaQuery = useClipMediaUrl(clip.id)
  const playbackRefreshAttempts = useRef(0)
  const previousMediaUrl = useRef<string | null>(null)
  const externalLink = resolveClipExternalLink(clip)
  const viewUrlLink = resolveClipViewLink(clip)
  const externalStorageLink = externalLink && externalLink !== viewUrlLink ? externalLink : null

  useEffect(() => {
    playbackRefreshAttempts.current = nextAttemptsAfterMediaSourceChange(
      previousMediaUrl.current,
      mediaQuery.mediaUrl,
      playbackRefreshAttempts.current,
    )
    previousMediaUrl.current = mediaQuery.mediaUrl
  }, [mediaQuery.mediaUrl])

  async function refreshPlaybackSourceAfterError(): Promise<void> {
    const source = mediaQuery.mediaUrl
    if (!shouldRefreshPlaybackSource(source, mediaQuery.usesToken, playbackRefreshAttempts.current)) {
      return
    }
    playbackRefreshAttempts.current += 1

    await mediaQuery.refresh()
  }

  return (
    <EventCard
      camera={clip.camera}
      time={formatEventTime(clip.created_at)}
      title={formatActivityType(clip.activity_type)}
      summary={clip.summary?.trim() || 'No summary available yet.'}
      media={(
        <MediaPanel
          aspect="video"
          className="event-card__media-panel"
          placeholder="Event video is not available for playback."
        >
          <div className="event-card__video-shell">
            {mediaQuery.isPending ? (
              <p className="muted">Preparing event video.</p>
            ) : mediaQuery.mediaUrl ? (
              <video
                className="event-card__video"
                controls
                preload="metadata"
                src={mediaQuery.mediaUrl}
                onError={() => {
                  void refreshPlaybackSourceAfterError()
                }}
              >
                Your browser does not support video playback.
              </video>
            ) : (
              <p className="muted">Event video is not available for playback.</p>
            )}
          </div>
        </MediaPanel>
      )}
      risk={<RiskBadge level={clip.risk_level} />}
      status={(
        <StatusBadge tone={clip.alerted ? 'unhealthy' : 'healthy'}>
          {formatAlertStatus(clip.alerted)}
        </StatusBadge>
      )}
      meta={[
        { label: 'Objects', value: formatDetectedObjects(clip) },
      ]}
      technicalDetails={showDetails ? (
        <section className="event-inline-details" aria-label="Event details">
          {mediaQuery.error ? (
            <p className="error-text">{describeClipError(mediaQuery.error)}</p>
          ) : null}
          <dl className="clip-detail-kv">
            <div className="clip-detail-kv-row">
              <dt>Activity</dt>
              <dd>{formatActivityType(clip.activity_type)}</dd>
            </div>
            <div className="clip-detail-kv-row">
              <dt>Risk</dt>
              <dd>{riskLabelForLevel(clip.risk_level)}</dd>
            </div>
            <div className="clip-detail-kv-row">
              <dt>Detected Objects</dt>
              <dd>{formatDetectedObjects(clip)}</dd>
            </div>
            <div className="clip-detail-kv-row">
              <dt>Time</dt>
              <dd>{formatTimestamp(clip.created_at)}</dd>
            </div>
          </dl>
          <TechnicalDetailsDisclosure summary="Advanced event details">
            <dl className="clip-detail-kv">
              <div className="clip-detail-kv-row">
                <dt>Event ID</dt>
                <dd className="clip-detail-mono">{clip.id}</dd>
              </div>
              <div className="clip-detail-kv-row">
                <dt>Processing Status</dt>
                <dd>{clip.status}</dd>
              </div>
              <div className="clip-detail-kv-row">
                <dt>Storage URI</dt>
                <dd className="clip-detail-mono">{clip.storage_uri ?? 'not available'}</dd>
              </div>
              <div className="clip-detail-kv-row">
                <dt>View URL</dt>
                <dd>
                  {viewUrlLink ? (
                    <a href={viewUrlLink} target="_blank" rel="noreferrer noopener">
                      Open external view
                    </a>
                  ) : (
                    'not available'
                  )}
                </dd>
              </div>
              <div className="clip-detail-kv-row">
                <dt>Playback Source</dt>
                <dd>{mediaQuery.usesToken ? 'Tokenized media URL' : 'Direct media endpoint'}</dd>
              </div>
              <div className="clip-detail-kv-row">
                <dt>External Storage Link</dt>
                <dd>
                  {externalStorageLink ? (
                    <a href={externalStorageLink} target="_blank" rel="noreferrer noopener">
                      Open external link
                    </a>
                  ) : (
                    'not available'
                  )}
                </dd>
              </div>
            </dl>
          </TechnicalDetailsDisclosure>
        </section>
      ) : null}
      actions={(
        <Button
          variant={showDetails ? 'ghost' : 'primary'}
          onClick={() => setShowDetails((current) => !current)}
        >
          {showDetails ? 'Hide details' : 'Details'}
        </Button>
      )}
    />
  )
}

export function ClipsPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [showAdvancedFilters, setShowAdvancedFilters] = useState(false)
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
  const activeEventFilters = hasActiveEventFilters(filterQuery)
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

  function selectCameraFilter(cameraName: string): void {
    applyQuery({
      ...query,
      cursor: undefined,
      camera: cameraName || undefined,
    })
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
          <p className="page__lead">Review recorded security events and quickly filter what matters.</p>
        </div>
        <Button variant="ghost" onClick={refreshClips} disabled={clipsQuery.isFetching}>
          {clipsQuery.isFetching ? 'Refreshing...' : 'Refresh'}
        </Button>
      </header>

      <section className="events-filter-bar" aria-label="Event filters">
        <div className="events-primary-filters">
          <label className="field-label events-camera-filter">
            Camera
            <select
              className="input"
              value={query.camera ?? ''}
              onChange={(event) => selectCameraFilter(event.target.value)}
            >
              <option value="">All cameras</option>
              {cameraOptions.map((cameraName) => (
                <option key={cameraName} value={cameraName}>
                  {cameraName}
                </option>
              ))}
            </select>
          </label>
          <FilterChips chips={quickFilterChips} onSelect={selectQuickFilter} />
        </div>
      </section>

      {showAdvancedFilters ? (
        <div className="filters-sheet" role="dialog" aria-modal="false" aria-label="More filters">
          <ClipsFilterPanel
            key={filterSignature}
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
              <p className="muted">{formatEventCount(clipItems.length)}</p>
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
            description={
              activeEventFilters
                ? 'No recorded security events match the current filters.'
                : 'Recorded security events will appear here after a camera captures activity.'
            }
            action={
              activeEventFilters ? (
                <Button variant="ghost" onClick={resetFilters}>
                  Clear filters
                </Button>
              ) : (
                <Link className="button button--primary" to="/live">
                  Open Live
                </Link>
              )
            }
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
                      <EventListCard key={clip.id} clip={clip} />
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
