import type { BackendFormStepProps } from './types'

function readString(config: Record<string, unknown>, key: string, fallback: string): string {
  const value = config[key]
  return typeof value === 'string' ? value : fallback
}

export function RtspForm({ config, onChange }: BackendFormStepProps) {
  const rtspUrl = readString(config, 'rtsp_url', '')
  const outputDir = readString(config, 'output_dir', './recordings')

  return (
    <div className="inline-form">
      <label className="field-label" htmlFor="camera-rtsp-url">
        RTSP URL
        <input
          id="camera-rtsp-url"
          className="input"
          type="text"
          value={rtspUrl}
          placeholder="rtsp://username:password@camera.local/stream"
          onChange={(event) => {
            onChange({
              ...config,
              rtsp_url: event.target.value,
            })
          }}
        />
      </label>

      <label className="field-label" htmlFor="camera-rtsp-output-dir">
        Output directory
        <input
          id="camera-rtsp-output-dir"
          className="input"
          type="text"
          value={outputDir}
          onChange={(event) => {
            onChange({
              ...config,
              output_dir: event.target.value,
            })
          }}
        />
      </label>
    </div>
  )
}

