import type { ReactNode } from 'react'

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'

import App from '../App'
import { initializeApiRuntimeConfig } from '../api/runtimeConfig'
import { QueryProvider } from './providers/QueryProvider'
import { ThemeProvider } from './providers/ThemeProvider'

export type RenderHomeSecApp = (rootElement: HTMLElement, app: ReactNode) => void

export interface BootstrapHomeSecAppOptions {
  rootElement: HTMLElement
  initializeRuntimeConfig?: () => Promise<void>
  render?: RenderHomeSecApp
}

function renderReactApp(rootElement: HTMLElement, app: ReactNode): void {
  createRoot(rootElement).render(app)
}

export function createHomeSecAppElement(): ReactNode {
  return (
    <StrictMode>
      <ThemeProvider>
        <QueryProvider>
          <BrowserRouter>
            <App />
          </BrowserRouter>
        </QueryProvider>
      </ThemeProvider>
    </StrictMode>
  )
}

export async function bootstrapHomeSecApp({
  rootElement,
  initializeRuntimeConfig = initializeApiRuntimeConfig,
  render = renderReactApp,
}: BootstrapHomeSecAppOptions): Promise<void> {
  await initializeRuntimeConfig()
  render(rootElement, createHomeSecAppElement())
}
