"""
match_tracker.py - Week 4 (W4-02, Day 15): Side-Out Tracking & Telemetry Score Progression.

Bridges sequential perception telemetry frames & bounce/hit events with ScoringFSM so
individual rally clips and full multi-inning rally streams emit per-frame USAP scoreboard states,
diagonal serve verification status, and terminal side-out / point transitions.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pbbase.court.homography import CourtGeometry
from pbbase.rules.scoring_fsm import ScoringFSM, ServeContext, SequenceEvent


@dataclass
class FrameRuleSnapshot:
    """Frame-aligned USAP scoreboard and serving state overlay for video/HUD rendering."""
    frame_idx: int
    timestamp_s: float
    rally_state: str               # SERVE_IN_FLIGHT | RALLY_LIVE | RALLY_ENDED_SIDE_OUT | RALLY_ENDED_POINT
    score_call: str
    score_a: int
    score_b: int
    serving_team: str
    receiving_team: str
    server_number: int
    active_server_id: str
    active_receiver_id: str
    server_box: str
    target_receiver_box: str
    target_quadrant_ft: Tuple[float, float, float, float]
    side_out_count: int
    rally_count: int
    serve_valid_quadrant: Optional[bool]
    status_banner: str
    event_emitted: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "timestamp_s": round(self.timestamp_s, 4),
            "rally_state": self.rally_state,
            "score_call": self.score_call,
            "score_a": self.score_a,
            "score_b": self.score_b,
            "serving_team": self.serving_team,
            "receiving_team": self.receiving_team,
            "server_number": self.server_number,
            "active_server_id": self.active_server_id,
            "active_receiver_id": self.active_receiver_id,
            "server_box": self.server_box,
            "target_receiver_box": self.target_receiver_box,
            "target_quadrant_ft": [round(v, 2) for v in self.target_quadrant_ft],
            "side_out_count": self.side_out_count,
            "rally_count": self.rally_count,
            "serve_valid_quadrant": self.serve_valid_quadrant,
            "status_banner": self.status_banner,
            "event_emitted": self.event_emitted,
        }


def _point_in_bounds(x_ft: float, y_ft: float, bounds: Tuple[float, float, float, float]) -> bool:
    x_min, x_max, y_min, y_max = bounds
    return (x_min <= x_ft <= x_max) and (y_min <= y_ft <= y_max)


def _infer_rally_terminal_winner(
    telemetry_data: Dict[str, Any],
    fsm: ScoringFSM,
) -> Tuple[Optional[int], Optional[str], str]:
    """
    Determine whether the clip exhibits a terminal rally outcome from the final ground bounce.
    Note: Formal two-bounce (W4-03), kitchen foot-fault (W4-04), and millimeter line-calling (W4-05)
    modules arrive in Days 16-17. Here on Day 15 (W4-02), we detect a clean terminal court exit on
    the final bounce event to drive side-out / point progression on single-rally video clips.
    """
    bounces = telemetry_data.get("bounces", [])
    if not bounces:
        return None, None, ""

    ground_bounces = [b for b in bounces if b.get("is_ground_bounce", True)]
    if not ground_bounces:
        return None, None, ""

    last_gb = ground_bounces[-1]
    c_ft = last_gb.get("court_ft") or [10.0, 22.0]
    bx, by = float(c_ft[0]), float(c_ft[1])
    b_frame = int(last_gb["frame_idx"])

    in_court = (0.0 <= bx <= CourtGeometry.WIDTH_FT) and (0.0 <= by <= CourtGeometry.LENGTH_FT)
    if not in_court:
        # Ball went out of bounds landing on half `by < 22` -> Team A hit it out -> Team B wins rally (and vice versa)
        half_landed = "B" if by < CourtGeometry.NET_Y_FT else "A"
        winner = "B" if half_landed == "B" else "A"
        detail = f"Shot landed out-of-bounds @ ({bx:.1f}, {by:.1f})ft on f.{b_frame} -> Team {winner} wins rally"
        return b_frame, winner, detail

    return None, None, ""


class MatchScoreTracker:
    """
    Week 4 Day 2 (W4-02): Side-Out & Score Progression Tracker over telemetry streams.
    Maps sequential frames + bounce events onto USAP scoreboard progression and side-out ledger.
    """

    def __init__(self, fsm: Optional[ScoringFSM] = None) -> None:
        self.fsm: Optional[ScoringFSM] = fsm

    def analyze_clip_telemetry(
        self,
        telemetry_data: Dict[str, Any],
        forced_rally_winner: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a full perception telemetry dictionary (from BasePerceptionPipeline) and emit:
          - Per-frame `FrameRuleSnapshot` entries for video overlay rendering.
          - Initial 0-0-2 state, diagonal serve bounce validation, and side-out / score progression.
        """
        stream = telemetry_data.get("telemetry_stream", [])
        if not stream:
            raise ValueError("telemetry_stream is empty; cannot track match scoring.")

        frame0 = stream[0]
        f0_players = frame0.get("players", [])
        f0_ball = frame0.get("ball") or {}
        f0_ball_ft = tuple(f0_ball["court_ft"]) if f0_ball.get("court_ft") else None

        if self.fsm is None:
            self.fsm = ScoringFSM.seed_from_telemetry_players(
                f0_players,
                infer_initial_server=True,
                initial_ball_court_ft=f0_ball_ft,
            )

        bounces = telemetry_data.get("bounces", [])
        ground_bounces = [b for b in bounces if b.get("is_ground_bounce", True)]
        first_ground_bounce = ground_bounces[0] if ground_bounces else None
        serve_bounce_frame = int(first_ground_bounce["frame_idx"]) if first_ground_bounce else None

        serve_valid: Optional[bool] = None
        if first_ground_bounce is not None and first_ground_bounce.get("court_ft"):
            ctx0 = self.fsm.get_serve_context()
            sx, sy = float(first_ground_bounce["court_ft"][0]), float(first_ground_bounce["court_ft"][1])
            serve_valid = _point_in_bounds(sx, sy, ctx0.target_quadrant_ft)

        # Determine if clip contains a terminal rally resolution (e.g. out-of-bounds shot ending rally)
        terminal_frame, inferred_winner, terminal_reason = _infer_rally_terminal_winner(telemetry_data, self.fsm)
        if forced_rally_winner in ("A", "B"):
            terminal_frame = int(bounces[-1]["frame_idx"]) if bounces else int(stream[-1]["frame_idx"])
            inferred_winner = forced_rally_winner
            terminal_reason = f"Explicit rally winner: Team {forced_rally_winner}"

        snapshots: List[FrameRuleSnapshot] = []
        transition_applied = False
        bounces_by_frame = {int(b["frame_idx"]): b for b in bounces}
        contact_count = 0

        for fr in stream:
            f_idx = int(fr["frame_idx"])
            ts = float(fr.get("timestamp_s", f_idx / 30.0))

            if f_idx in bounces_by_frame:
                contact_count += 1

            emitted_dict: Optional[Dict[str, Any]] = None
            if (not transition_applied) and (terminal_frame is not None) and (f_idx >= terminal_frame) and (inferred_winner is not None):
                ev = self.fsm.advance_rally(inferred_winner, notes=terminal_reason)
                emitted_dict = ev.to_dict()
                transition_applied = True

            ctx = self.fsm.get_serve_context()

            if transition_applied:
                ev_last = self.fsm.history[-1]
                if ev_last.event_type == "SIDE_OUT":
                    rally_state = "RALLY_ENDED_SIDE_OUT"
                    banner = f"SIDE-OUT! {ev_last.prev_score_call} -> {ctx.score_call} (Next: Team {ctx.serving_team} {ctx.active_server_id} RIGHT)"
                elif ev_last.event_type == "GAME_WON":
                    rally_state = "MATCH_OVER"
                    banner = f"GAME OVER! Team {ctx.winner_team} wins ({ctx.score_call})"
                else:
                    rally_state = "RALLY_ENDED_POINT"
                    banner = f"POINT TEAM {ctx.serving_team}! {ev_last.prev_score_call} -> {ctx.score_call}"
            elif serve_bounce_frame is not None and f_idx < serve_bounce_frame:
                rally_state = "SERVE_IN_FLIGHT"
                banner = f"SERVE {ctx.score_call}: {ctx.active_server_id} ({ctx.server_box}) -> {ctx.active_receiver_id} ({ctx.target_receiver_box})"
            else:
                rally_state = "RALLY_LIVE"
                if serve_valid is True and first_ground_bounce is not None:
                    sx, sy = first_ground_bounce["court_ft"]
                    banner = f"RALLY LIVE ({ctx.score_call}) • Serve IN [{sx:.1f},{sy:.1f}]ft • Contacts: {contact_count}"
                else:
                    banner = f"RALLY LIVE ({ctx.score_call}) • Contacts: {contact_count}"

            snapshots.append(
                FrameRuleSnapshot(
                    frame_idx=f_idx,
                    timestamp_s=ts,
                    rally_state=rally_state,
                    score_call=ctx.score_call,
                    score_a=self.fsm.teams["A"].score,
                    score_b=self.fsm.teams["B"].score,
                    serving_team=ctx.serving_team,
                    receiving_team=ctx.receiving_team,
                    server_number=ctx.server_number,
                    active_server_id=ctx.active_server_id,
                    active_receiver_id=ctx.active_receiver_id,
                    server_box=ctx.server_box,
                    target_receiver_box=ctx.target_receiver_box,
                    target_quadrant_ft=ctx.target_quadrant_ft,
                    side_out_count=self.fsm.side_out_count,
                    rally_count=self.fsm.rally_count,
                    serve_valid_quadrant=serve_valid,
                    status_banner=banner,
                    event_emitted=emitted_dict,
                )
            )

        return {
            "phase": "Week 4 Day 2 (W4-01 & W4-02)",
            "video_path": telemetry_data.get("video_path"),
            "total_frames": len(snapshots),
            "serve_bounce_frame": serve_bounce_frame,
            "serve_valid_diagonal": serve_valid,
            "terminal_event_frame": terminal_frame,
            "fsm_summary": self.fsm.to_dict(),
            "frame_rules": [s.to_dict() for s in snapshots],
        }
