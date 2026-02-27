import { Button } from '../../../components/ui/Button'
import { StatusBadge } from '../../../components/ui/StatusBadge'
import type { ReviewSectionStatus, ReviewSummaryItem } from '../review'

interface ReviewSummaryCardProps {
  title: string
  status: ReviewSectionStatus
  items: readonly ReviewSummaryItem[]
  emptyMessage: string
  onEdit: () => void
  editDisabled?: boolean
}

function statusTone(status: ReviewSectionStatus): 'healthy' | 'degraded' | 'unknown' {
  switch (status) {
    case 'configured':
      return 'healthy'
    case 'skipped':
      return 'degraded'
    case 'defaults':
      return 'unknown'
  }
}

function statusLabel(status: ReviewSectionStatus): string {
  switch (status) {
    case 'configured':
      return 'Configured'
    case 'skipped':
      return 'Skipped'
    case 'defaults':
      return 'Defaults'
  }
}

export function ReviewSummaryCard({
  title,
  status,
  items,
  emptyMessage,
  onEdit,
  editDisabled = false,
}: ReviewSummaryCardProps) {
  return (
    <article className="review-summary-card">
      <header className="review-summary-card__header">
        <h3 className="review-summary-card__title">{title}</h3>
        <StatusBadge tone={statusTone(status)}>{statusLabel(status)}</StatusBadge>
      </header>

      {items.length > 0 ? (
        <dl className="review-summary-card__items">
          {items.map((item) => (
            <div key={`${item.label}:${item.value}`} className="review-summary-card__item">
              <dt>{item.label}</dt>
              <dd>{item.value}</dd>
            </div>
          ))}
        </dl>
      ) : (
        <p className="subtle">{emptyMessage}</p>
      )}

      <div className="review-summary-card__actions">
        <Button variant="ghost" onClick={onEdit} disabled={editDisabled}>
          Edit
        </Button>
      </div>
    </article>
  )
}
