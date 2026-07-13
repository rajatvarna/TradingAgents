"""A1 shared prompt partials (docs/PROMPT_STYLE_GUIDE.md) for the risk-debate agents.

See researchers/bull_researcher.py's _SHARED_BLOCKS for why this is passed
unconditionally regardless of which template version is selected.
"""

from __future__ import annotations

_SHARED_BLOCKS = {
    "data_integrity_block": ("_shared/data_integrity", "v1"),
    "calibration_block": ("_shared/calibration", "v1"),
}
