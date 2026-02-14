import { useQuery } from '@tanstack/react-query'

import { apiClient } from '../client'
import type { ClipListSnapshot } from '../client'
import type { ListClipsQuery } from '../generated/types'

const CLIPS_REFRESH_MS = 10_000

export function useClipsQuery(query: ListClipsQuery) {
  return useQuery<ClipListSnapshot>({
    queryKey: ['clips', query],
    queryFn: ({ signal }) => apiClient.getClips(query, { signal }),
    staleTime: CLIPS_REFRESH_MS,
    refetchInterval: CLIPS_REFRESH_MS,
  })
}
