import { afterEach, describe, expect, it, vi } from 'vitest'

type AudioWorkletProcessorClass = new (options?: AudioWorkletNodeOptions) => {
  port: { postMessage: ReturnType<typeof vi.fn> }
  process: (inputs: Float32Array[][]) => boolean
}

describe('talkPcmProcessor AudioWorklet', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.resetModules()
  })

  it('emits exact PCM16 frames from the browser AudioWorklet path', async () => {
    const registeredProcessors = new Map<string, AudioWorkletProcessorClass>()

    class FakeAudioWorkletProcessor {
      port = { postMessage: vi.fn() }
    }

    vi.stubGlobal('sampleRate', 48_000)
    vi.stubGlobal('AudioWorkletProcessor', FakeAudioWorkletProcessor)
    vi.stubGlobal(
      'registerProcessor',
      vi.fn((name: string, processor: AudioWorkletProcessorClass) => {
        registeredProcessors.set(name, processor)
      }),
    )

    // Given: The production AudioWorklet processor is registered in a worklet-like runtime
    // @ts-expect-error AudioWorklet source is a plain JS module loaded by the browser.
    await import('./talkPcmProcessor.js')
    const Processor = registeredProcessors.get('talk-pcm-processor')
    expect(Processor).toBeDefined()
    if (!Processor) {
      throw new Error('talk-pcm-processor was not registered')
    }
    const processor = new Processor({
      processorOptions: {
        sourceSampleRate: 48_000,
        targetSampleRate: 16_000,
        frameSamples: 320,
      },
    })

    // When: 48 kHz browser audio supplies enough samples for one 16 kHz/20 ms frame
    const keepAlive = processor.process([[new Float32Array(960).fill(0.25)]])

    // Then: The worklet posts one exact-size transferable PCM16 frame
    expect(keepAlive).toBe(true)
    expect(processor.port.postMessage).toHaveBeenCalledTimes(1)
    const call = processor.port.postMessage.mock.calls[0]
    if (!call) {
      throw new Error('AudioWorklet did not post a PCM frame')
    }
    const [frame, transfer] = call
    expect(frame).toBeInstanceOf(ArrayBuffer)
    expect((frame as ArrayBuffer).byteLength).toBe(640)
    expect(transfer).toEqual([frame])
  })
})
