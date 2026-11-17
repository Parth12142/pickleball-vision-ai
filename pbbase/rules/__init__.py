"""
pbbase.rules - Phase 3: Official USAP Pickleball Rules Engine & Autonomous Referee.
Built incrementally day-wise in Week 4 on top of Weeks 1-3 zero-rule perception telemetry.
"""

from pbbase.rules.scoring_fsm import (
    ScoringFSM,
    ServeContext,
    SequenceEvent,
    TeamCourtState,
    SERVICE_QUADRANT_BOUNDS_FT,
)
from pbbase.rules.match_tracker import (
    FrameRuleSnapshot,
    MatchScoreTracker,
)
from pbbase.rules.two_bounce import (
    TwoBounceEnforcer,
    TwoBounceEvent,
    TwoBounceFrameState,
    TWO_BOUNCE_PHASES,
    infer_serve_start_from_telemetry,
)
from pbbase.rules.nvz_detector import (
    NVZFootFaultDetector,
    NVZFaultEvent,
    NVZ_FAR_LINE_Y_FT,
    NVZ_NEAR_LINE_Y_FT,
    compute_nvz_clearance_ft,
)

__all__ = [
    "ScoringFSM",
    "ServeContext",
    "SequenceEvent",
    "TeamCourtState",
    "SERVICE_QUADRANT_BOUNDS_FT",
    "FrameRuleSnapshot",
    "MatchScoreTracker",
    "TwoBounceEnforcer",
    "TwoBounceEvent",
    "TwoBounceFrameState",
    "TWO_BOUNCE_PHASES",
    "infer_serve_start_from_telemetry",
    "NVZFootFaultDetector",
    "NVZFaultEvent",
    "NVZ_FAR_LINE_Y_FT",
    "NVZ_NEAR_LINE_Y_FT",
    "compute_nvz_clearance_ft",
]
