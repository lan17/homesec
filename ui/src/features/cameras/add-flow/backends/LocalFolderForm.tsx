import type { BackendFormStepProps } from './types'

function readString(config: Record<string, unknown>, key: string, fallback: string): string {
  const value = config[key]
  return typeof value === 'string' ? value : fallback
}

function readNumber(config: Record<string, unknown>, key: string, fallback: number): number {
  const value = config[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

export function LocalFolderForm({ config, onChange }: BackendFormStepProps) {
  const watchDir = readString(config, 'watch_dir', './recordings')
  const pollInterval = readNumber(config, 'poll_interval', 1.0)
  const stabilityThreshold = readNumber(config, 'stability_threshold_s', 3.0)

  return (
    <div className="inline-form">
      <label className="field-label" htmlFor="camera-local-folder-watch-dir">
        Watch directory
        <input
          id="camera-local-folder-watch-dir"
          className="input"
          type="text"
          value={watchDir}
          onChange={(event) => {
            onChange({
              ...config,
              watch_dir: event.target.value,
            })
          }}
        />
      </label>

      <div className="camera-form-grid">
        <label className="field-label" htmlFor="camera-local-folder-poll-interval">
          Poll interval (seconds)
          <input
            id="camera-local-folder-poll-interval"
            className="input"
            type="number"
            min={0.1}
            step={0.1}
            value={pollInterval}
            onChange={(event) => {
              const nextValue = Number.parseFloat(event.target.value)
              if (Number.isNaN(nextValue)) {
                return
              }
              onChange({
                ...config,
                poll_interval: nextValue,
              })
            }}
          />
        </label>

        <label className="field-label" htmlFor="camera-local-folder-stability">
          Stability threshold (seconds)
          <input
            id="camera-local-folder-stability"
            className="input"
            type="number"
            min={0.1}
            step={0.1}
            value={stabilityThreshold}
            onChange={(event) => {
              const nextValue = Number.parseFloat(event.target.value)
              if (Number.isNaN(nextValue)) {
                return
              }
              onChange({
                ...config,
                stability_threshold_s: nextValue,
              })
            }}
          />
        </label>
      </div>
    </div>
  )
}

