import { expect, test, type Page } from '@playwright/test'

const MOBILE_VIEWPORT = { width: 320, height: 700 }
const DESKTOP_VIEWPORT = { width: 1280, height: 800 }
const SHELL_ROUTES = [
  '/live',
  '/events',
  '/events/test-id',
  '/settings',
  '/settings/cameras',
  '/system',
] as const

const SAFE_AREA_OVERRIDES = {
  top: '8px',
  right: '12px',
  bottom: '34px',
  left: '12px',
} as const

const camera = {
  name: 'front_door',
  enabled: true,
  healthy: true,
  last_heartbeat: 1_797_187_200,
  source_backend: 'rtsp',
  source_config: {},
}

const clip = {
  id: 'test-id',
  camera: 'front_door',
  status: 'done',
  created_at: '2026-06-14T02:30:00Z',
  activity_type: 'package',
  risk_level: 'medium',
  summary: 'Package delivery at the front door.',
  detected_objects: ['person', 'package'],
  storage_uri: 'dropbox:/clips/test-id.mp4',
  view_url: null,
  alerted: true,
}

async function mockHomeSecApi(page: Page): Promise<void> {
  await page.route('**/api/v1/**', async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    const path = url.pathname
    const method = request.method()

    const fulfillJson = (payload: unknown, status = 200) => route.fulfill({
      status,
      contentType: 'application/json',
      headers: {
        'access-control-allow-origin': '*',
        'access-control-allow-headers': '*',
        'access-control-allow-methods': 'GET,POST,DELETE,OPTIONS',
      },
      body: JSON.stringify(payload),
    })

    if (method === 'OPTIONS') {
      await route.fulfill({
        status: 204,
        headers: {
          'access-control-allow-origin': '*',
          'access-control-allow-headers': '*',
          'access-control-allow-methods': 'GET,POST,DELETE,OPTIONS',
        },
      })
      return
    }

    if (path === '/api/v1/setup/status') {
      await fulfillJson({
        state: 'complete',
        has_cameras: true,
        pipeline_running: true,
        auth_configured: false,
      })
      return
    }

    if (path === '/api/v1/health') {
      await fulfillJson({
        status: 'healthy',
        bootstrap_mode: false,
        pipeline: 'running',
        postgres: 'ok',
        cameras_online: 1,
      })
      return
    }

    if (path === '/api/v1/stats') {
      await fulfillJson({
        clips_today: 3,
        alerts_today: 1,
        cameras_total: 1,
        cameras_online: 1,
        uptime_seconds: 18_000,
      })
      return
    }

    if (path === '/api/v1/maintenance/postgres-backups/status') {
      await fulfillJson({
        enabled: true,
        available: true,
        running: false,
        last_attempted_at: '2026-06-14T02:00:00Z',
        last_success_at: '2026-06-14T02:00:00Z',
        last_error: null,
        last_local_path: '/backups/homesec.sql',
        last_uploaded_uri: null,
        next_run_at: '2026-06-15T02:00:00Z',
        pending_remote_delete_count: 0,
        unavailable_reason: null,
      })
      return
    }

    if (path === '/api/v1/cameras') {
      await fulfillJson([camera])
      return
    }

    if (path === '/api/v1/runtime/status') {
      await fulfillJson({
        state: 'idle',
        generation: 3,
        reload_in_progress: false,
        active_config_version: 'cfg-v3',
        last_reload_at: null,
        last_reload_error: null,
      })
      return
    }

    if (path === '/api/v1/preview/cameras/front_door') {
      await fulfillJson({
        camera_name: 'front_door',
        enabled: true,
        state: 'idle',
        viewer_count: 0,
        degraded_reason: null,
        last_error: null,
        idle_shutdown_at: null,
      })
      return
    }

    if (path === '/api/v1/talk/cameras/front_door') {
      await fulfillJson({
        camera_name: 'front_door',
        enabled: true,
        policy_enabled: true,
        capability: 'supported',
        state: 'idle',
        active_session_id: null,
        supported_codecs: ['pcm_s16le'],
        offered_codecs: ['pcm_s16le'],
        selected_codec: 'pcm_s16le',
        backend: 'rtsp',
        backend_reason: null,
        last_error: null,
      })
      return
    }

    if (path === '/api/v1/clips' && method === 'GET') {
      await fulfillJson({
        clips: [clip],
        limit: 25,
        next_cursor: null,
        has_more: false,
      })
      return
    }

    if (path === '/api/v1/clips/test-id/media-token' && method === 'POST') {
      await fulfillJson({
        media_url: '/event-video.mp4',
        tokenized: false,
        expires_at: null,
      })
      return
    }

    if (path === '/api/v1/clips/test-id' && method === 'GET') {
      await fulfillJson(clip)
      return
    }

    await fulfillJson({ detail: `Unhandled ${method} ${path}` }, 404)
  })
}

async function openApp(page: Page, path: string): Promise<void> {
  await page.goto(path)
  await page.getByRole('main').waitFor()
}

async function applySafeAreaOverrides(page: Page): Promise<void> {
  await page.evaluate((tokens) => {
    const root = document.documentElement
    root.style.setProperty('--safe-area-inset-top', tokens.top)
    root.style.setProperty('--safe-area-inset-right', tokens.right)
    root.style.setProperty('--safe-area-inset-bottom', tokens.bottom)
    root.style.setProperty('--safe-area-inset-left', tokens.left)
  }, SAFE_AREA_OVERRIDES)
}

async function expectNoHorizontalOverflow(page: Page): Promise<void> {
  const metrics = await page.evaluate(() => ({
    viewportWidth: window.innerWidth,
    htmlScrollWidth: document.documentElement.scrollWidth,
    bodyScrollWidth: document.body.scrollWidth,
  }))

  expect(metrics.htmlScrollWidth).toBeLessThanOrEqual(metrics.viewportWidth)
  expect(metrics.bodyScrollWidth).toBeLessThanOrEqual(metrics.viewportWidth)
}

async function expectMobileBottomNavClearance(page: Page): Promise<void> {
  const nav = page.locator('.mobile-bottom-nav')
  const navBox = await nav.boundingBox()
  const metrics = await page.evaluate(() => {
    const content = document.querySelector('.app-shell__content')
    const styles = content ? getComputedStyle(content) : null
    return {
      viewportWidth: window.innerWidth,
      viewportHeight: window.innerHeight,
      contentPaddingBottom: styles ? Number.parseFloat(styles.paddingBottom) : 0,
      scrollPaddingBottom: Number.parseFloat(getComputedStyle(document.documentElement).scrollPaddingBottom),
    }
  })

  expect(navBox).not.toBeNull()
  expect(navBox?.x ?? -1).toBeGreaterThanOrEqual(0)
  expect((navBox?.x ?? 0) + (navBox?.width ?? 0)).toBeLessThanOrEqual(metrics.viewportWidth)
  expect((navBox?.y ?? 0) + (navBox?.height ?? 0)).toBeLessThanOrEqual(metrics.viewportHeight)
  expect(metrics.contentPaddingBottom).toBeGreaterThan(80)
  expect(metrics.scrollPaddingBottom).toBeGreaterThan(80)
}

async function expectElementAboveMobileNav(page: Page, selector: string): Promise<void> {
  const target = page.locator(selector).first()
  await target.scrollIntoViewIfNeeded()
  const targetBox = await target.boundingBox()
  const navBox = await page.locator('.mobile-bottom-nav').boundingBox()

  expect(targetBox).not.toBeNull()
  expect(navBox).not.toBeNull()
  expect(targetBox?.y ?? 0).toBeGreaterThanOrEqual(0)
  expect((targetBox?.y ?? 0) + (targetBox?.height ?? 0)).toBeLessThanOrEqual(navBox?.y ?? 0)
}

test.beforeEach(async ({ page }) => {
  await mockHomeSecApi(page)
})

test.describe('iOS M1 mobile layout hardening', () => {
  test.use({ viewport: MOBILE_VIEWPORT })

  for (const route of SHELL_ROUTES) {
    test(`${route} has no horizontal overflow and keeps bottom nav clear`, async ({ page }) => {
      // Given: The HomeSec app is opened at iPhone width with API responses mocked
      await openApp(page, route)

      // When: The rendered route is measured in a real browser layout engine
      await expect(page.getByRole('heading').first()).toBeVisible()

      // Then: Page content stays within the viewport and reserves room for fixed nav
      await expectNoHorizontalOverflow(page)
      await expectMobileBottomNavClearance(page)
    })
  }

  test('honors nonzero safe-area insets for mobile shell chrome', async ({ page }) => {
    // Given: The shell renders with simulated iPhone notch and home-indicator insets
    await openApp(page, '/live')
    await applySafeAreaOverrides(page)

    // When: The topbar, content, and fixed bottom nav are measured after re-layout
    const metrics = await page.evaluate(() => {
      const topbar = document.querySelector('.app-shell__topbar')
      const content = document.querySelector('.app-shell__content')
      const nav = document.querySelector('.mobile-bottom-nav')
      const topbarStyles = topbar ? getComputedStyle(topbar) : null
      const contentStyles = content ? getComputedStyle(content) : null
      const navBox = nav?.getBoundingClientRect()

      return {
        viewportWidth: window.innerWidth,
        viewportHeight: window.innerHeight,
        topbarPaddingTop: topbarStyles ? Number.parseFloat(topbarStyles.paddingTop) : 0,
        contentPaddingBottom: contentStyles ? Number.parseFloat(contentStyles.paddingBottom) : 0,
        scrollPaddingBottom: Number.parseFloat(getComputedStyle(document.documentElement).scrollPaddingBottom),
        navLeft: navBox?.left ?? -1,
        navRight: navBox?.right ?? -1,
        navBottom: navBox?.bottom ?? -1,
      }
    })

    // Then: Safe-area tokens move chrome away from each unsafe viewport edge
    expect(metrics.topbarPaddingTop).toBeGreaterThanOrEqual(20)
    expect(metrics.contentPaddingBottom).toBeGreaterThanOrEqual(120)
    expect(metrics.scrollPaddingBottom).toBeGreaterThanOrEqual(120)
    expect(metrics.navLeft).toBeGreaterThanOrEqual(24)
    expect(metrics.viewportWidth - metrics.navRight).toBeGreaterThanOrEqual(24)
    expect(metrics.viewportHeight - metrics.navBottom).toBeGreaterThanOrEqual(46)
  })

  test('keeps live preview controls above the bottom nav', async ({ page }) => {
    // Given: Live view renders a camera preview at iPhone width
    await openApp(page, '/live')

    // When: The preview viewport is scrolled into view
    await expect(page.getByText('Start live view to watch this camera.')).toBeVisible()

    // Then: The preview surface remains above the fixed bottom nav
    await expectElementAboveMobileNav(page, '.camera-preview__viewport')
  })

  test('keeps event video controls above the bottom nav', async ({ page }) => {
    // Given: Event detail renders a video panel at iPhone width
    await openApp(page, '/events/test-id')

    // When: The event video panel is scrolled into view
    await expect(page.locator('.clip-detail-video')).toBeVisible()

    // Then: The media viewport remains above the fixed bottom nav
    await expectElementAboveMobileNav(page, '.clip-detail-media .media-panel__viewport')
  })

  test('hides bottom nav while a form control is focused', async ({ page }) => {
    // Given: Events exposes a mobile filter form control
    await openApp(page, '/events')

    // When: The Camera filter receives focus as it would with the iOS keyboard
    await page.getByRole('combobox', { name: 'Camera' }).focus()

    // Then: The fixed bottom nav is removed from the focus layout
    await expect(page.locator('.mobile-bottom-nav')).toHaveCSS('display', 'none')
  })

  test('keeps native setup inside safe-area-aware viewport padding', async ({ page }) => {
    // Given: Native setup bypasses AppShell and renders with simulated iPhone safe-area insets
    await page.goto('/native-setup')
    await applySafeAreaOverrides(page)

    // When: The setup page is measured at iPhone width
    await expect(page.getByRole('heading', { name: 'Connect to HomeSec' })).toBeVisible()
    const metrics = await page.locator('.native-setup-page').evaluate((element) => {
      const styles = getComputedStyle(element)
      return {
        minHeight: styles.minHeight,
        paddingTop: Number.parseFloat(styles.paddingTop),
        paddingBottom: Number.parseFloat(styles.paddingBottom),
        scrollPaddingBottom: Number.parseFloat(styles.scrollPaddingBottom),
        htmlScrollWidth: document.documentElement.scrollWidth,
        viewportWidth: window.innerWidth,
      }
    })

    // Then: Setup has dynamic viewport sizing and no mobile horizontal overflow
    expect(metrics.minHeight).toBe('700px')
    expect(metrics.paddingTop).toBeGreaterThanOrEqual(24)
    expect(metrics.paddingBottom).toBeGreaterThanOrEqual(50)
    expect(metrics.scrollPaddingBottom).toBeGreaterThanOrEqual(50)
    expect(metrics.htmlScrollWidth).toBeLessThanOrEqual(metrics.viewportWidth)
  })
})

test.describe('desktop layout regression guard', () => {
  test.use({ viewport: DESKTOP_VIEWPORT })

  for (const route of SHELL_ROUTES) {
    test(`${route} keeps desktop nav in the topbar`, async ({ page }) => {
      // Given: The app is opened at desktop width
      await openApp(page, route)

      // When: Navigation CSS is inspected in a real browser
      const desktopNavDisplay = await page.locator('.app-shell__nav').evaluate((element) =>
        getComputedStyle(element).display
      )
      const mobileNavDisplay = await page.locator('.mobile-bottom-nav').evaluate((element) =>
        getComputedStyle(element).display
      )

      // Then: Desktop keeps the topbar nav visible and the mobile nav hidden
      expect(desktopNavDisplay).toBe('flex')
      expect(mobileNavDisplay).toBe('none')
      await expectNoHorizontalOverflow(page)
    })
  }
})
