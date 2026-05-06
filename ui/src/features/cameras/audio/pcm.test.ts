import { describe, expect, it } from 'vitest'

import { expectedPcmFrameBytes, frameSampleCount, PcmFrameEncoder } from './pcm'

describe('PcmFrameEncoder', () => {
  it('emits exact PCM frame sizes without resampling', () => {
    const encoder = new PcmFrameEncoder(16_000, 16_000, frameSampleCount(16_000, 20))

    const frames = encoder.push(new Float32Array(320).fill(0.5))

    expect(frames).toHaveLength(1)
    expect(frames[0]).toHaveProperty('byteLength', expectedPcmFrameBytes(16_000, 20, 1))
  })

  it('resamples 48 kHz browser input to 16 kHz talk frames', () => {
    const encoder = new PcmFrameEncoder(48_000, 16_000, frameSampleCount(16_000, 20))

    const firstFrames = encoder.push(new Float32Array(480).fill(0.25))
    const secondFrames = encoder.push(new Float32Array(480).fill(0.25))

    expect(firstFrames).toHaveLength(0)
    expect(secondFrames).toHaveLength(1)
    expect(secondFrames[0]).toHaveProperty('byteLength', 320 * Int16Array.BYTES_PER_ELEMENT)
  })

  it('does not drift across consecutive resampled frames', () => {
    const encoder = new PcmFrameEncoder(48_000, 16_000, frameSampleCount(16_000, 20))

    const frames = encoder.push(new Float32Array(48_000).fill(0.1))

    expect(frames).toHaveLength(50)
    expect(frames.every((frame) => frame.byteLength === 640)).toBe(true)
  })
})
