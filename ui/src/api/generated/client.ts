import type { HealthResponse } from './types'

export interface ApiRequestOptions {
  signal?: AbortSignal
  apiKey?: string | null
}

export interface GeneratedHomeSecClient {
  getHealth(options?: ApiRequestOptions): Promise<HealthResponse>
}
