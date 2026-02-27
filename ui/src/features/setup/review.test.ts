import { describe, expect, it } from 'vitest'

import { buildFinalizeRequestFromDrafts, buildReviewSectionSummaries } from './review'

describe('review helpers', () => {
  it('builds finalize payload from available wizard drafts', () => {
    // Given: Wizard drafts with all major setup sections configured
    const drafts = {
      camera: {
        name: 'front_door',
        enabled: true,
        source_backend: 'rtsp',
        source_config: { rtsp_url: 'rtsp://front-door' },
      },
      storage: {
        backend: 'local' as const,
        config: { root: './storage' },
      },
      detection: {
        filter: {
          backend: 'yolo',
          config: { classes: ['person'] },
        },
        vlm: null,
      },
      notifications: {
        notifiers: [
          {
            backend: 'mqtt',
            enabled: true,
            config: { topic_prefix: 'homesec' },
          },
        ],
        alert_policy: {
          backend: 'default',
          enabled: true,
          config: { min_risk_level: 'high' },
        },
      },
    }

    // When: Building finalize request from drafts
    const payload = buildFinalizeRequestFromDrafts(drafts)

    // Then: Payload includes normalized section contracts expected by finalize endpoint
    expect(payload.cameras).toEqual([
      {
        name: 'front_door',
        enabled: true,
        source: {
          backend: 'rtsp',
          config: { rtsp_url: 'rtsp://front-door' },
        },
      },
    ])
    expect(payload.storage).toEqual({
      backend: 'local',
      config: { root: './storage' },
    })
    expect(payload.filter?.backend).toBe('yolo')
    expect(payload.notifiers?.[0]?.backend).toBe('mqtt')
  })

  it('summarizes section status for configured, skipped, and defaults states', () => {
    // Given: Camera configured, storage skipped, and other sections left empty
    const summaries = buildReviewSectionSummaries(
      {
        camera: {
          name: 'front_door',
          enabled: true,
          source_backend: 'rtsp',
          source_config: { rtsp_url: 'rtsp://front-door' },
        },
        storage: null,
        detection: null,
        notifications: null,
      },
      new Set(['storage']),
    )

    // When: Reading section summary status map
    const byId = new Map(summaries.map((summary) => [summary.stepId, summary.status]))

    // Then: Statuses reflect configured, skipped, and defaults behavior
    expect(byId.get('camera')).toBe('configured')
    expect(byId.get('storage')).toBe('skipped')
    expect(byId.get('detection')).toBe('defaults')
    expect(byId.get('notifications')).toBe('defaults')
  })
})
