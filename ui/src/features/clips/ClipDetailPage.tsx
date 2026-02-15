import { useEffect, useRef } from 'react'
import { Link, useParams } from 'react-router-dom'

import { clearApiKey, isUnauthorizedAPIError, saveApiKey } from '../../api/client'
import { useClipMediaUrl } from '../../api/hooks/useClipMediaUrl'
import { useClipQuery } from '../../api/hooks/useClipQuery'
import { ApiKeyGate } from '../../components/ui/ApiKeyGate'
import { Button } from '../../components/ui/Button'
import { Card } from '../../components/ui/Card'
import {
  describeClipError,
  formatTimestamp,
  renderDetectedObjects,
  resolveClipExternalLink,
  resolveClipViewLink,
} from './presentation'
import {
  nextAttemptsAfterMediaSourceChange,
  shouldRefreshPlaybackSource,
} from './playbackRetry'

export function ClipDetailPage() {
  const { clipId } = useParams<{ clipId: string }>()
  const clipQuery = useClipQuery(clipId)
  const mediaQuery = useClipMediaUrl(clipId)
  const unauthorized = isUnauthorizedAPIError(clipQuery.error)
  const clip = clipQuery.data
  const externalLink = clip ? resolveClipExternalLink(clip) : null
  const viewUrlLink = clip ? resolveClipViewLink(clip) : null
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
    saveApiKey(apiKey)
    playbackRefreshAttempts.current = 0
    await clipQuery.refetch()
    await mediaQuery.refresh()
  }

  async function clearStoredApiKey(): Promise<void> {
    clearApiKey()
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
          <h1 className="page__title">Clip Detail</h1>
          <p className="page__lead">Clip ID: {clipId ?? 'unknown'}</p>
        </div>
        <Button variant="ghost" onClick={refreshClip} disabled={clipQuery.isFetching || !clipId}>
          {clipQuery.isFetching ? 'Refreshing...' : 'Refresh'}
        </Button>
      </header>

      {!clipId ? (
        <Card title="Invalid clip request">
          <p className="error-text">Missing clip ID in route.</p>
          <p className="muted">
            Return to <Link to="/clips">clips list</Link>.
          </p>
        </Card>
      ) : null}

      {clipQuery.isPending ? (
        <Card title="Loading clip">
          <p className="muted">Fetching clip metadata...</p>
        </Card>
      ) : null}

      {unauthorized ? (
        <Card title="Authentication required">
          <ApiKeyGate
            busy={clipQuery.isFetching}
            onSubmit={submitApiKey}
            onClear={clearStoredApiKey}
          />
        </Card>
      ) : null}

      {clipQuery.error && !unauthorized ? (
        <Card title="Clip query failed">
          <p className="error-text">{describeClipError(clipQuery.error)}</p>
        </Card>
      ) : null}

      {clip ? (
        <>
          <Card title="Playback + Storage" subtitle="Primary playback is served by HomeSec /media">
            <div className="clip-detail-video-shell">
              {mediaQuery.isPending ? (
                <p className="muted">Preparing secure playback URL...</p>
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
                <p className="muted">Clip media is not available for playback.</p>
              )}
            </div>

            {mediaQuery.error ? (
              <p className="error-text">{describeClipError(mediaQuery.error)}</p>
            ) : null}

            <div className="clip-detail-actions">
              {externalLink ? (
                <a
                  className="button button--primary"
                  href={externalLink}
                  target="_blank"
                  rel="noreferrer noopener"
                >
                  Open in storage
                </a>
              ) : null}
              <Link className="button button--ghost" to="/clips">
                Back to clips
              </Link>
            </div>
            <div className="clip-detail-link-list">
              <p className="muted">
                <strong>view_url:</strong>{' '}
                {viewUrlLink ? (
                  <a href={viewUrlLink} target="_blank" rel="noreferrer noopener">
                    {viewUrlLink}
                  </a>
                ) : (
                  'not available'
                )}
              </p>
              <p className="muted clip-detail-mono">
                <strong>storage_uri:</strong> {clip.storage_uri ?? 'not available'}
              </p>
            </div>
          </Card>

          <div className="clip-detail-grid">
            <Card title="Analysis" subtitle="VLM + detection fields from API">
              <dl className="clip-detail-kv">
                <div className="clip-detail-kv-row">
                  <dt>Summary</dt>
                  <dd>{clip.summary ?? 'not available'}</dd>
                </div>
                <div className="clip-detail-kv-row">
                  <dt>Activity Type</dt>
                  <dd>{clip.activity_type ?? 'not available'}</dd>
                </div>
                <div className="clip-detail-kv-row">
                  <dt>Risk Level</dt>
                  <dd>{clip.risk_level ?? 'not available'}</dd>
                </div>
                <div className="clip-detail-kv-row">
                  <dt>Detected Objects</dt>
                  <dd>{renderDetectedObjects(clip)}</dd>
                </div>
                <div className="clip-detail-kv-row">
                  <dt>Alerted</dt>
                  <dd>{clip.alerted ? 'true' : 'false'}</dd>
                </div>
              </dl>
            </Card>

            <Card title="Metadata">
              <dl className="clip-detail-kv">
                <div className="clip-detail-kv-row">
                  <dt>ID</dt>
                  <dd className="clip-detail-mono">{clip.id}</dd>
                </div>
                <div className="clip-detail-kv-row">
                  <dt>Camera</dt>
                  <dd>{clip.camera}</dd>
                </div>
                <div className="clip-detail-kv-row">
                  <dt>Status</dt>
                  <dd>{clip.status}</dd>
                </div>
                <div className="clip-detail-kv-row">
                  <dt>Created At</dt>
                  <dd>{formatTimestamp(clip.created_at)}</dd>
                </div>
              </dl>
            </Card>
          </div>
        </>
      ) : null}
    </section>
  )
}
