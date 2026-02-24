import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'

import App from './App'
import { QueryProvider } from './app/providers/QueryProvider'
import { ThemeProvider } from './app/providers/ThemeProvider'
import './styles/global.css'
import './styles/tokens.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider>
      <QueryProvider>
        <BrowserRouter>
          <App />
        </BrowserRouter>
      </QueryProvider>
    </ThemeProvider>
  </StrictMode>,
)
