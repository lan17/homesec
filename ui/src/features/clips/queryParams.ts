import type { ClipStatus, ListClipsQuery } from '../../api/generated/types'

export const DEFAULT_CLIPS_LIMIT = 25
const MIN_CLIPS_LIMIT = 1
const MAX_CLIPS_LIMIT = 100

export const CLIP_LIMIT_OPTIONS = [10, 25, 50, 100] as const

export const CLIP_STATUS_OPTIONS = [
  'queued_local',
  'uploaded',
  'analyzed',
  'done',
  'error',
  'deleted',
] as const satisfies readonly ClipStatus[]

export type AlertedFilter = 'any' | 'true' | 'false'

export interface ClipsFilterFormState {
  camera: string
  status: ClipStatus | ''
  alerted: AlertedFilter
  riskLevel: string
  activityType: string
  sinceLocal: string
  untilLocal: string
  limit: number
}

function parseLimit(value: string | null): number {
  if (value === null) {
    return DEFAULT_CLIPS_LIMIT
  }

  const parsed = Number.parseInt(value, 10)
  if (!Number.isFinite(parsed)) {
    return DEFAULT_CLIPS_LIMIT
  }

  return Math.min(MAX_CLIPS_LIMIT, Math.max(MIN_CLIPS_LIMIT, parsed))
}

function isClipStatus(value: string | null): value is ClipStatus {
  if (value === null) {
    return false
  }
  return (CLIP_STATUS_OPTIONS as readonly string[]).includes(value)
}

function toIsoUtc(localValue: string): string | undefined {
  if (!localValue.trim()) {
    return undefined
  }
  const date = new Date(localValue)
  if (Number.isNaN(date.valueOf())) {
    return undefined
  }
  return date.toISOString()
}

function toLocalDateTimeInput(isoValue: string | null | undefined): string {
  if (!isoValue) {
    return ''
  }
  const date = new Date(isoValue)
  if (Number.isNaN(date.valueOf())) {
    return ''
  }
  const year = date.getFullYear()
  const month = `${date.getMonth() + 1}`.padStart(2, '0')
  const day = `${date.getDate()}`.padStart(2, '0')
  const hours = `${date.getHours()}`.padStart(2, '0')
  const minutes = `${date.getMinutes()}`.padStart(2, '0')
  return `${year}-${month}-${day}T${hours}:${minutes}`
}

export function parseClipsQuery(searchParams: URLSearchParams): ListClipsQuery {
  const alertedRaw = searchParams.get('alerted')
  const alerted = alertedRaw === 'true' ? true : alertedRaw === 'false' ? false : undefined
  const statusRaw = searchParams.get('status')

  return {
    camera: searchParams.get('camera')?.trim() || undefined,
    status: isClipStatus(statusRaw) ? statusRaw : undefined,
    alerted,
    risk_level: searchParams.get('risk_level')?.trim() || undefined,
    activity_type: searchParams.get('activity_type')?.trim().toLowerCase() || undefined,
    since: searchParams.get('since')?.trim() || undefined,
    until: searchParams.get('until')?.trim() || undefined,
    limit: parseLimit(searchParams.get('limit')),
    cursor: searchParams.get('cursor')?.trim() || undefined,
  }
}

export function queryToFormState(query: ListClipsQuery): ClipsFilterFormState {
  return {
    camera: query.camera ?? '',
    status: query.status ?? '',
    alerted: query.alerted === true ? 'true' : query.alerted === false ? 'false' : 'any',
    riskLevel: query.risk_level ?? '',
    activityType: query.activity_type ?? '',
    sinceLocal: toLocalDateTimeInput(query.since),
    untilLocal: toLocalDateTimeInput(query.until),
    limit: query.limit ?? DEFAULT_CLIPS_LIMIT,
  }
}

export function formStateToQuery(form: ClipsFilterFormState): ListClipsQuery {
  return {
    camera: form.camera.trim() || undefined,
    status: form.status || undefined,
    alerted: form.alerted === 'any' ? undefined : form.alerted === 'true',
    risk_level: form.riskLevel.trim() || undefined,
    activity_type: form.activityType.trim().toLowerCase() || undefined,
    since: toIsoUtc(form.sinceLocal),
    until: toIsoUtc(form.untilLocal),
    limit: form.limit,
  }
}

export function queryToSearchParams(query: ListClipsQuery): URLSearchParams {
  const params = new URLSearchParams()

  if (query.camera) {
    params.set('camera', query.camera)
  }
  if (query.status) {
    params.set('status', query.status)
  }
  if (typeof query.alerted === 'boolean') {
    params.set('alerted', query.alerted ? 'true' : 'false')
  }
  if (query.risk_level) {
    params.set('risk_level', query.risk_level)
  }
  if (query.activity_type) {
    params.set('activity_type', query.activity_type)
  }
  if (query.since) {
    params.set('since', query.since)
  }
  if (query.until) {
    params.set('until', query.until)
  }
  if (query.limit) {
    params.set('limit', String(query.limit))
  }
  if (query.cursor) {
    params.set('cursor', query.cursor)
  }

  return params
}

export function queryWithoutCursor(query: ListClipsQuery): ListClipsQuery {
  return {
    ...query,
    cursor: undefined,
  }
}
