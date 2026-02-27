"""Helpers for API tests that construct stub FastAPI apps.

When UI serving is mandatory, test stubs that use the default `ui/dist` need a
minimal dist tree to avoid unrelated route tests failing before request logic
is exercised.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from homesec.models.config import FastAPIServerConfig

_DEFAULT_UI_DIST_DIR = Path("ui/dist")


def ensure_stub_ui_dist(server_config: FastAPIServerConfig) -> FastAPIServerConfig:
    """Return a server config with an existing ui_dist_dir for default test stubs.

    Only rewrites the path when it is the default `ui/dist` and missing. Custom
    paths are preserved so tests can intentionally exercise missing-dist failures.
    """
    configured = Path(server_config.ui_dist_dir)
    if configured.exists() or configured != _DEFAULT_UI_DIST_DIR:
        return server_config

    temp_root = Path(tempfile.mkdtemp(prefix="homesec-ui-dist-"))
    assets_dir = temp_root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    (temp_root / "index.html").write_text("<!doctype html><html><body>HomeSec UI</body></html>")
    (temp_root / "favicon.ico").write_bytes(b"ico")
    (assets_dir / "app.js").write_text("console.log('homesec ui');")
    return server_config.model_copy(update={"ui_dist_dir": str(temp_root)})
