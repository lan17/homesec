export type RiskBadgeTone = 'low' | 'medium' | 'high' | 'critical' | 'unknown'

export function riskToneForLevel(level: string | null | undefined): RiskBadgeTone {
  switch (level?.trim().toLowerCase()) {
    case 'low':
      return 'low'
    case 'medium':
    case 'moderate':
      return 'medium'
    case 'high':
      return 'high'
    case 'critical':
      return 'critical'
    default:
      return 'unknown'
  }
}

export function riskLabelForLevel(level: string | null | undefined): string {
  switch (level?.trim().toLowerCase()) {
    case 'low':
      return 'Low'
    case 'medium':
    case 'moderate':
      return 'Medium'
    case 'high':
      return 'High'
    case 'critical':
      return 'Critical'
    default:
      return 'Risk unavailable'
  }
}
