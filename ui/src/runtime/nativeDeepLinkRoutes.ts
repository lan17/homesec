const HOMESEC_DEEP_LINK_SCHEME = 'homesec:'
const DEFAULT_DEEP_LINK_ROUTE = '/live'

const ALLOWED_DEEP_LINK_ROUTE_PREFIXES = [
  '/live',
  '/events',
  '/settings',
  '/system',
  '/cameras',
  '/clips',
  '/dashboard',
  '/home',
]

function routeIsAllowed(route: string): boolean {
  return ALLOWED_DEEP_LINK_ROUTE_PREFIXES.some((prefix) => {
    return route === prefix || route.startsWith(`${prefix}/`)
  })
}

function pathnameFromUrl(url: URL): string {
  const pathSegments = [url.hostname, url.pathname.replace(/^\/+/, '')].filter(Boolean)
  return `/${pathSegments.join('/')}`.replace(/\/{2,}/g, '/')
}

export function parseNativeDeepLinkRoute(rawUrl: string): string | null {
  let url: URL
  try {
    url = new URL(rawUrl)
  } catch {
    return null
  }

  if (url.protocol !== HOMESEC_DEEP_LINK_SCHEME) {
    return null
  }

  const pathname = pathnameFromUrl(url)
  if (!routeIsAllowed(pathname)) {
    return DEFAULT_DEEP_LINK_ROUTE
  }

  return `${pathname}${url.search}${url.hash}`
}
