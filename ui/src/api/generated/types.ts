export interface HealthResponse {
  status: string
  pipeline: string
  postgres: string
  cameras_online: number
}
