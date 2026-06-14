import { useEffect, useMemo, useRef } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { Link, useLocation, useParams } from 'react-router-dom'

import {
  clearApiKey,
  isAPIError,
  isUnauthorizedAPIError,
  saveApiKey,
  type ClipListSnapshot,
} from '../../api/client'
import type { ClipResponse, ListClipsQuery } from '../../api/generated/types'
import { useClipMediaUrl } from '../../api/hooks/useClipMediaUrl'
import { useClipQuery } from '../../api/hooks/useClipQuery'
import { QUERY_KEYS } from '../../api/hooks/queryKeys'
import { ApiKeyGate } from '../../components/ui/ApiKeyGate'
import { Button } from '../../components/ui/Button'
import { EmptyState } from '../../components/ui/EmptyState'
import { MediaPanel } from '../../components/ui/MediaPanel'
import { RiskBadge } from '../../components/ui/RiskBadge'
import { riskLabelForLevel } from '../../components/ui/riskTone'
import { StatusBadge } from '../../components/ui/StatusBadge'
import { TechnicalDetailsDisclosure } from '../../components/ui/TechnicalDetailsDisclosure'
import {
  describeClipError,
  formatAlertStatus,
  formatActivityType,
  formatDetectedObjects,
  formatEventTime,
  formatTimestamp,
  resolveClipExternalLink,
  resolveClipViewLink,
} from './presentation'
import {
  nextAttemptsAfterMediaSourceChange,
  shouldRefreshPlaybackSource,
} from './playbackRetry'
import { parseClipsQuery } from './queryParams'

interface NeighborEvents {
  previous: ClipResponse | null
  next: ClipResponse | null
}

function eventDetailPath(clipId: string, routeSearch: string): string {
  return `/events/${encodeURIComponent(clipId)}${eventNavigationSearch(routeSearch)}`
}

function eventNavigationSearch(routeSearch: string): string {
  const params = new URLSearchParams(routeSearch)
  params.delete('from')
  const search = params.toString()
  return search ? `?${search}` : ''
}

function eventListPath(routeSearch: string): string {
  return `/events${eventNavigationSearch(routeSearch)}`
}

function isNotificationOpen(searchParams: URLSearchParams): boolean {
  return searchParams.get('from') === 'notification'
}

function isMissingEventError(error: unknown): boolean {
  return isAPIError(error) && error.status === 404
}

function describeClipLoadError(error: unknown, openedFromNotification: boolean): string {
  if (isMissingEventError(error)) {
    return openedFromNotification
      ? 'The event opened from this notification is no longer available. It may have been deleted or cleaned up.'
      : 'This event is no longer available. It may have been deleted or cleaned up.'
  }

  return describeClipError(error)
}

function clipLoadErrorTitle(error: unknown, openedFromNotification: boolean): string {
  if (isMissingEventError(error)) {
    return 'Event no longer available'
  }
  return openedFromNotification ? 'Notification event could not load' : 'Event could not load'
}

function findClipWindow(
  queryClient: ReturnType<typeof useQueryClient>,
  currentClipId: string | undefined,
  query: ListClipsQuery,
): ClipResponse[] {
  if (!currentClipId) {
    return []
  }

  const preferredWindow = queryClient.getQueryData<ClipListSnapshot>(
    QUERY_KEYS.clips(query),
  )?.clips
  if (preferredWindow?.some((clip) => clip.id === currentClipId)) {
    return preferredWindow
  }

  const cachedWindows = queryClient.getQueriesData<ClipListSnapshot>({ queryKey: ['clips'] })
  for (const [, data] of cachedWindows) {
    if (data?.clips.some((clip) => clip.id === currentClipId)) {
      return data.clips
    }
  }

  return []
}

function findNeighbors(clips: readonly ClipResponse[], currentClipId: string | undefined): NeighborEvents {
  const currentIndex = clips.findIndex((clip) => clip.id === currentClipId)
  if (currentIndex < 0) {
    return { previous: null, next: null }
  }

  return {
    previous: clips[currentIndex - 1] ?? null,
    next: clips[currentIndex + 1] ?? null,
  }
}

export function ClipDetailPage() {
  const { clipId } = useParams<{ clipId: string }>()
  const location = useLocation()
  const queryClient = useQueryClient()
  const clipQuery = useClipQuery(clipId)
  const missingEvent = isMissingEventError(clipQuery.error)
  const clip = missingEvent ? undefined : clipQuery.data
  const mediaQuery = useClipMediaUrl(clip?.id)
  const searchParams = useMemo(() => new URLSearchParams(location.search), [location.search])
  const openedFromNotification = isNotificationOpen(searchParams)
  const unauthorized = isUnauthorizedAPIError(clipQuery.error)
  const externalLink = clip ? resolveClipExternalLink(clip) : null
  const viewUrlLink = clip ? resolveClipViewLink(clip) : null
  const externalStorageLink = externalLink && externalLink !== viewUrlLink ? externalLink : null
  const backToEventsPath = eventListPath(location.search)
  const listQuery = useMemo(
    () => parseClipsQuery(searchParams),
    [searchParams],
  )
  const clipWindow = useMemo(
    () => findClipWindow(queryClient, clipId, listQuery),
    [clipId, listQuery, queryClient],
  )
  const neighbors = useMemo(
    () => findNeighbors(clipWindow, clipId),
    [clipId, clipWindow],
  )
  const playbackRefreshAttempts = useRef(0)
  const previousMediaUrl = useRef<string | null>(null)

  useEffect(() => {
    playbackRefreshAttempts.current = nextAttemptsAfterMediaSourceChange(
      previousMediaUrl.current,
      mediaQuery.mediaUrl,
      playbackRefreshAttempts.current,
    )
    previousMediaUrl.current = mediaQuery.mediaUrl
  }, [mediaQuery.mediaUrl])

  async function submitApiKey(apiKey: string): Promise<void> {
    await saveApiKey(apiKey)
    playbackRefreshAttempts.current = 0
    await clipQuery.refetch()
    await mediaQuery.refresh()
  }

  async function clearStoredApiKey(): Promise<void> {
    await clearApiKey()
    await clipQuery.refetch()
  }

  async function refreshClip(): Promise<void> {
    playbackRefreshAttempts.current = 0
    await clipQuery.refetch()
    await mediaQuery.refresh()
  }

  async function refreshPlaybackSourceAfterError(): Promise<void> {
    const source = mediaQuery.mediaUrl
    if (!shouldRefreshPlaybackSource(source, mediaQuery.usesToken, playbackRefreshAttempts.current)) {
      return
    }
    playbackRefreshAttempts.current += 1

    await mediaQuery.refresh()
  }

  return (
    <section className="page fade-in-up">
      <header className="page__header">
        <div>
          <h1 className="page__title">Event</h1>
          <p className="page__lead">
            {clip ? `${clip.camera} - ${formatTimestamp(clip.created_at)}` : 'Recorded security event'}
          </p>
        </div>
        <div className="clip-detail-header-actions">
          <Link className="button button--ghost" to={backToEventsPath}>
            Back to Events
          </Link>
          <Button variant="ghost" onClick={refreshClip} disabled={clipQuery.isFetching || !clipId}>
            {clipQuery.isFetching ? 'Refreshing...' : 'Refresh'}
          </Button>
        </div>
      </header>

      {!clipId ? (
        <EmptyState
          title="Invalid event request"
          description={(
            <>
              Missing event ID. Return to <Link to={backToEventsPath}>events</Link>.
            </>
          )}
          tone="error"
        />
      ) : null}

      {clipQuery.isPending ? (
        <EmptyState
          title="Loading event"
          description="Fetching event video and metadata."
          tone="loading"
        />
      ) : null}

      {unauthorized ? (
        <section className="auth-panel" aria-label="Authentication required">
          <ApiKeyGate
            busy={clipQuery.isFetching}
            onSubmit={submitApiKey}
            onClear={clearStoredApiKey}
          />
        </section>
      ) : null}

      {clipQuery.error && !unauthorized ? (
        <EmptyState
          title={clipLoadErrorTitle(clipQuery.error, openedFromNotification)}
          description={describeClipLoadError(clipQuery.error, openedFromNotification)}
          tone="error"
        />
      ) : null}

      {clip ? (
        <div className="clip-detail-layout">
          <MediaPanel
            title="Event video"
            subtitle={`${clip.camera} at ${formatEventTime(clip.created_at)}`}
            status={<RiskBadge level={clip.risk_level} />}
            aspect="video"
            className="clip-detail-media"
          >
            <div className="clip-detail-video-shell">
              {mediaQuery.isPending ? (
                <p className="muted">Preparing event video.</p>
              ) : mediaQuery.mediaUrl ? (
                <video
                  className="clip-detail-video"
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

          <section className="clip-detail-summary" aria-label="Event summary">
            {mediaQuery.error ? (
              <p className="error-text">{describeClipError(mediaQuery.error)}</p>
            ) : null}
            <div className="clip-detail-summary__header">
              <div>
                <p className="clip-detail-eyebrow">{clip.camera}</p>
                <h2 className="clip-detail-heading">
                  {formatActivityType(clip.activity_type)}
                </h2>
              </div>
              <StatusBadge tone={clip.alerted ? 'unhealthy' : 'healthy'}>
                {formatAlertStatus(clip.alerted)}
              </StatusBadge>
            </div>
            <p className="clip-detail-summary__text">
              {clip.summary?.trim() || 'No summary is available yet.'}
            </p>

            <div className="clip-detail-actions">
              {neighbors.previous ? (
                <Link
                  className="button button--ghost"
                  to={eventDetailPath(neighbors.previous.id, location.search)}
                >
                  Previous event
                </Link>
              ) : (
                <span className="button button--ghost button--disabled" aria-disabled="true">
                  Previous event
                </span>
              )}
              {neighbors.next ? (
                <Link
                  className="button button--primary"
                  to={eventDetailPath(neighbors.next.id, location.search)}
                >
                  Next event
                </Link>
              ) : (
                <span className="button button--primary button--disabled" aria-disabled="true">
                  Next event
                </span>
              )}
            </div>

            <section className="clip-detail-section" aria-labelledby="event-ai-metadata">
              <h3 id="event-ai-metadata" className="section-title">AI metadata</h3>
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
            </section>

            <TechnicalDetailsDisclosure summary="Technical event details">
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
        </div>
      ) : null}
    </section>
  )
}
