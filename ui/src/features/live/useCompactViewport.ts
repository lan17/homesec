import { useCallback, useSyncExternalStore } from 'react'

export const COMPACT_VIEWPORT_QUERY = '(max-width: 620px)'

function matchesQuery(query: string): boolean {
  if (typeof window === 'undefined' || !window.matchMedia) {
    return false
  }
  return window.matchMedia(query).matches
}

export function useCompactViewport(query = COMPACT_VIEWPORT_QUERY): boolean {
  const subscribe = useCallback((onStoreChange: () => void) => {
    if (typeof window === 'undefined' || !window.matchMedia) {
      return () => {}
    }

    const mediaQuery = window.matchMedia(query)
    const handleChange = (): void => {
      onStoreChange()
    }

    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener('change', handleChange)
      return () => {
        mediaQuery.removeEventListener('change', handleChange)
      }
    }

    mediaQuery.addListener(handleChange)
    return () => {
      mediaQuery.removeListener(handleChange)
    }
  }, [query])

  const getSnapshot = useCallback(() => matchesQuery(query), [query])
  const getServerSnapshot = useCallback(() => false, [])

  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot)
}
