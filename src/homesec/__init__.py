"""HomeSec camera pipeline."""

__version__ = "0.1.1"

# Export commonly used types
from homesec.errors import PipelineError
from homesec.models.alert import Alert
from homesec.models.clip import Clip, ClipStateData
from homesec.models.filter import FilterResult
from homesec.models.vlm import AnalysisResult

__all__ = [
    "Alert",
    "AnalysisResult",
    "Clip",
    "ClipStateData",
    "FilterResult",
    "PipelineError",
    "__version__",
]
