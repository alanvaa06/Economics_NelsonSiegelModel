"""
Nelson-Siegel Model Library

A Python library for yield curve modeling using the Nelson-Siegel methodology.
Supports both nominal Treasury and real TIPS yield curve analysis.
"""

__version__ = "1.0.0"
__author__ = "Economics Research Team"
__email__ = "research@example.com"

from .model import NelsonSiegelModel, TreasuryNelsonSiegelModel, TIPSNelsonSiegelModel
from .data import TreasuryDataDownloader, TIPSDataDownloader, DataManager
from .analysis import YieldCurveAnalyzer
from .plotting import YieldCurvePlotter

# Optional interactive components (requires ipywidgets)
try:
    from .interactive import InteractiveYieldCurveExplorer, HistoricalFactorExplorer, create_yield_curve_tutorial
    HAS_INTERACTIVE = True
except ImportError:
    HAS_INTERACTIVE = False

__all__ = [
    "NelsonSiegelModel",
    "TreasuryNelsonSiegelModel",
    "TIPSNelsonSiegelModel",
    "TreasuryDataDownloader", 
    "TIPSDataDownloader",
    "DataManager",
    "YieldCurveAnalyzer",
    "YieldCurvePlotter",
]

if HAS_INTERACTIVE:
    __all__.extend([
        "InteractiveYieldCurveExplorer",
        "HistoricalFactorExplorer", 
        "create_yield_curve_tutorial"
    ])
