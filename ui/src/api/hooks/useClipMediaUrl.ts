import { useCallback, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'

import { apiClient, hasStoredApiKey } from '../client'
import type { ClipMediaTokenSnapshot } from '../client'

const MEDIA_TOKEN_REFRESH_LEAD_MS = 60_000

interface ClipMediaUrlState {
  mediaUrl: string | null
  expiresAt: Date | null
  usesToken: boolean
  isPending: boolean
  error: unknown
  refresh: () => Promise<string | null>
}

export function buildDirectMediaPath(clipId: string): string {
  return `/api/v1/clips/${encodeURIComponent(clipId)}/media`
}

export function toDateOrNull(value: string | null | undefined): Date | null {
  if (!value) {
    return null
  }
  const parsed = new Date(value)
  if (Number.isNaN(parsed.valueOf())) {
    return null
  }
  return parsed
}

export function computeTokenRefreshDelayMs(
  expiresAt: string,
  nowMs: number,
  refreshLeadMs = MEDIA_TOKEN_REFRESH_LEAD_MS,
): number | null {
  const refreshAtMs = Date.parse(expiresAt) - refreshLeadMs
  if (Number.isNaN(refreshAtMs)) {
    return null
  }
  return Math.max(refreshAtMs - nowMs, 0)
}

export function useClipMediaUrl(clipId: string | undefined): ClipMediaUrlState {
  const directMediaUrl = clipId ? apiClient.resolvePath(buildDirectMediaPath(clipId)) : null
  const shouldRequestToken = Boolean(clipId) && hasStoredApiKey()

  const tokenQuery = useQuery<ClipMediaTokenSnapshot>({
    queryKey: ['clip-media-token', clipId],
    queryFn: ({ signal }) => apiClient.createClipMediaToken(clipId ?? '', { signal }),
    enabled: Boolean(clipId) && shouldRequestToken,
    staleTime: 0,
  })
  const refetchToken = tokenQuery.refetch

  const expiresAtRaw = tokenQuery.data?.expires_at

  useEffect(() => {
    if (!shouldRequestToken || typeof window === 'undefined') {
      return
    }

    if (!expiresAtRaw) {
      return
    }

    const timeoutMs = computeTokenRefreshDelayMs(expiresAtRaw, Date.now())
    if (timeoutMs === null) {
      return
    }
    const timeoutId = window.setTimeout(() => {
      void refetchToken()
    }, timeoutMs)

    return () => {
      window.clearTimeout(timeoutId)
    }
  }, [expiresAtRaw, refetchToken, shouldRequestToken])

  const refresh = useCallback(async (): Promise<string | null> => {
    if (!clipId) {
      return null
    }
    if (!shouldRequestToken) {
      return directMediaUrl
    }

    const refreshed = await refetchToken()
    const tokenizedPath = refreshed.data?.media_url
    if (!tokenizedPath) {
      return null
    }
    return apiClient.resolvePath(tokenizedPath)
  }, [clipId, directMediaUrl, refetchToken, shouldRequestToken])

  const tokenizedUrl = tokenQuery.data ? apiClient.resolvePath(tokenQuery.data.media_url) : null
  const mediaUrl = shouldRequestToken ? tokenizedUrl : directMediaUrl
  const expiresAt = shouldRequestToken ? toDateOrNull(expiresAtRaw) : null

  return {
    mediaUrl,
    expiresAt,
    usesToken: Boolean(expiresAt),
    isPending: shouldRequestToken ? tokenQuery.isPending : false,
    error: shouldRequestToken ? tokenQuery.error : null,
    refresh,
  }
}
