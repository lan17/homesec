export interface FilterChip {
  id: string
  label: string
  active?: boolean
  disabled?: boolean
}

interface FilterChipsProps {
  chips: readonly FilterChip[]
  onSelect: (id: string) => void
  ariaLabel?: string
}

export function FilterChips({
  chips,
  onSelect,
  ariaLabel = 'Quick filters',
}: FilterChipsProps) {
  return (
    <div className="filter-chips" role="toolbar" aria-label={ariaLabel}>
      {chips.map((chip) => (
        <button
          key={chip.id}
          type="button"
          className={chip.active ? 'filter-chip filter-chip--active' : 'filter-chip'}
          aria-pressed={chip.active ?? false}
          disabled={chip.disabled}
          onClick={() => {
            onSelect(chip.id)
          }}
        >
          {chip.label}
        </button>
      ))}
    </div>
  )
}
