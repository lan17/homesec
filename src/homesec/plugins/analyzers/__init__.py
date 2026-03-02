"""Analyzer plugins and registry."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

from homesec.interfaces import VLMAnalyzer
from homesec.models.enums import RiskLevel, VLMRunMode
from homesec.models.filter import FilterResult
from homesec.models.vlm import AnalysisResult, VLMConfig
from homesec.plugins.registry import PluginType, load_plugin

logger = logging.getLogger(__name__)


class _NoopVLM(VLMAnalyzer):
    """Placeholder VLM that returns a safe default when run_mode is 'never'."""

    async def analyze(
        self, video_path: Path, filter_result: FilterResult, config: VLMConfig
    ) -> AnalysisResult:
        return AnalysisResult(
            risk_level=RiskLevel.LOW,
            activity_type="skipped",
            summary="VLM analysis disabled (run_mode=never)",
        )

    async def ping(self) -> bool:
        return True

    async def shutdown(self, timeout: float | None = None) -> None:
        pass


def load_analyzer(config: VLMConfig) -> VLMAnalyzer:
    """Load and instantiate a VLM analyzer plugin.

    Args:
        config: VLM configuration

    Returns:
        Configured VLMAnalyzer instance

    Raises:
        ValueError: If backend not found in registry
        ValidationError: If config validation fails
    """
    if config.run_mode == VLMRunMode.NEVER:
        logger.info("VLM run_mode is 'never'; using no-op analyzer")
        return _NoopVLM()

    return cast(
        VLMAnalyzer,
        load_plugin(
            PluginType.ANALYZER,
            config.backend,
            config.config,
        ),
    )


__all__ = ["load_analyzer"]
