export const MAX_PLAYBACK_REFRESH_RETRIES = 1

export function shouldRefreshPlaybackSource(
  mediaUrl: string | null,
  usesToken: boolean,
  attempts: number,
): boolean {
  if (!mediaUrl) {
    return false
  }
  if (!usesToken) {
    return false
  }
  return attempts < MAX_PLAYBACK_REFRESH_RETRIES
}

export function nextAttemptsAfterMediaSourceChange(
  previousMediaUrl: string | null,
  currentMediaUrl: string | null,
  attempts: number,
): number {
  if (previousMediaUrl !== currentMediaUrl) {
    return 0
  }
  return attempts
}
