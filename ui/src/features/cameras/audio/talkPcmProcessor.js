class TalkPcmProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super()
    const processorOptions = options.processorOptions || {}
    this.frameSamples = Math.max(1, Number(processorOptions.frameSamples) || 320)
    this.buffer = []
  }

  process(inputs) {
    const input = inputs[0]
    const channel = input && input[0]
    if (!channel) {
      return true
    }

    for (let index = 0; index < channel.length; index += 1) {
      this.buffer.push(channel[index])
      if (this.buffer.length >= this.frameSamples) {
        const frame = this.buffer.splice(0, this.frameSamples)
        const pcm = new Int16Array(frame.length)
        for (let sampleIndex = 0; sampleIndex < frame.length; sampleIndex += 1) {
          const clamped = Math.max(-1, Math.min(1, frame[sampleIndex]))
          pcm[sampleIndex] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff
        }
        this.port.postMessage(pcm.buffer, [pcm.buffer])
      }
    }

    return true
  }
}

registerProcessor('talk-pcm-processor', TalkPcmProcessor)
