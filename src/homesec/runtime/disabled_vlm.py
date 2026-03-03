"""Disabled VLM analyzer implementation for run_mode=never."""

from __future__ import annotations

from pathlib import Path

from homesec.interfaces import VLMAnalyzer
from homesec.models.enums import RiskLevel
from homesec.models.filter import FilterResult
from homesec.models.vlm import AnalysisResult, VLMConfig


class DisabledVLMAnalyzer(VLMAnalyzer):
    """Runtime-only analyzer used when VLM is disabled."""

    async def analyze(
        self, video_path: Path, filter_result: FilterResult, config: VLMConfig
    ) -> AnalysisResult:
        _ = video_path
        _ = filter_result
        _ = config
        return AnalysisResult(
            risk_level=RiskLevel.LOW,
            activity_type="skipped",
            summary="VLM analysis disabled (run_mode=never)",
        )

    async def ping(self) -> bool:
        return True

    async def shutdown(self, timeout: float | None = None) -> None:
        _ = timeout
