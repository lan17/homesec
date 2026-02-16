import type { ListClipsQuery } from '../generated/types'

export const QUERY_KEYS = {
  cameras: ['cameras'] as const,
  runtimeStatus: ['runtime-status'] as const,
  health: ['health'] as const,
  stats: ['stats'] as const,
  clips: (query: ListClipsQuery) => ['clips', query] as const,
  clip: (clipId: string | undefined) => ['clip', clipId] as const,
  clipMediaToken: (clipId: string | undefined) => ['clip-media-token', clipId] as const,
}
