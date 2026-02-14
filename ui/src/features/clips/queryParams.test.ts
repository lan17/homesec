import { describe, expect, it } from 'vitest'

import {
  DEFAULT_CLIPS_LIMIT,
  formStateToQuery,
  parseClipsQuery,
  queryToFormState,
  queryToSearchParams,
} from './queryParams'

describe('parseClipsQuery', () => {
  it('parses supported filters and normalizes activity_type to lowercase', () => {
    // Given: URL search params with mixed-case activity type and explicit filters
    const params = new URLSearchParams({
      camera: 'front-door',
      status: 'done',
      alerted: 'false',
      risk_level: 'high',
      activity_type: 'PackageDrop',
      since: '2026-02-14T00:00:00.000Z',
      until: '2026-02-14T01:00:00.000Z',
      limit: '50',
      cursor: 'abc123',
    })

    // When: Parsing into query object
    const query = parseClipsQuery(params)

    // Then: Fields should be typed and activity_type normalized
    expect(query).toEqual({
      camera: 'front-door',
      status: 'done',
      alerted: false,
      risk_level: 'high',
      activity_type: 'packagedrop',
      since: '2026-02-14T00:00:00.000Z',
      until: '2026-02-14T01:00:00.000Z',
      limit: 50,
      cursor: 'abc123',
    })
  })

  it('falls back to defaults for invalid or missing values', () => {
    // Given: Params with unsupported status and invalid limit
    const params = new URLSearchParams({
      status: 'unknown',
      limit: '-10',
      alerted: 'maybe',
    })

    // When: Parsing into query object
    const query = parseClipsQuery(params)

    // Then: Unsupported values should be dropped or clamped to defaults
    expect(query.status).toBeUndefined()
    expect(query.alerted).toBeUndefined()
    expect(query.limit).toBe(1)
    expect(query.cursor).toBeUndefined()
  })
})

describe('form/query conversion', () => {
  it('round-trips form state through query and URL params', () => {
    // Given: Form state with local datetime values and activity type mixed case
    const formState = {
      camera: 'garage',
      status: 'uploaded',
      alerted: 'true',
      riskLevel: 'medium',
      activityType: 'Vehicle',
      sinceLocal: '2026-02-14T10:15',
      untilLocal: '2026-02-14T11:30',
      limit: DEFAULT_CLIPS_LIMIT,
    } as const

    // When: Converting form -> query -> URL params -> parsed query -> form
    const query = formStateToQuery(formState)
    const params = queryToSearchParams(query)
    const parsed = parseClipsQuery(params)
    const hydratedForm = queryToFormState(parsed)

    // Then: Key filter fields should remain stable across conversion cycle
    expect(parsed.camera).toBe('garage')
    expect(parsed.status).toBe('uploaded')
    expect(parsed.alerted).toBe(true)
    expect(parsed.activity_type).toBe('vehicle')
    expect(hydratedForm.camera).toBe('garage')
    expect(hydratedForm.status).toBe('uploaded')
    expect(hydratedForm.alerted).toBe('true')
    expect(hydratedForm.activityType).toBe('vehicle')
    expect(hydratedForm.limit).toBe(DEFAULT_CLIPS_LIMIT)
  })
})
