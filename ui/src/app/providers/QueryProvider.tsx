import type { PropsWithChildren } from 'react'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      gcTime: 5 * 60 * 1000,
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
})

export function QueryProvider({ children }: PropsWithChildren) {
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
}
