import { bootstrapHomeSecApp } from './app/bootstrap'
import './styles/global.css'
import './styles/tokens.css'

void bootstrapHomeSecApp({
  rootElement: document.getElementById('root')!,
})
