"""
match_tracker.py - Week 4 (W4-02, Day 15): Side-Out Tracking & Telemetry Score Progression.

Bridges sequential perception telemetry frames & bounce/hit events with ScoringFSM so
individual rally clips and full multi-inning rally streams emit per-frame USAP scoreboard states,
diagonal serve verification status, and terminal side-out / point transitions.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pbbase.court.homography import CourtGeometry
from pbbase.rules.nvz_detector import NVZFootFaultDetector
from pbbase.rules.scoring_fsm import SERVICE_QUADRANT_BOUNDS_FT, ScoringFSM, ServeContext, SequenceEvent
from pbbase.rules.two_bounce import TwoBounceEnforcer, infer_serve_start_from_telemetry


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
    two_bounce_phase: str = "AWAITING_SERVE_BOUNCE"
    two_bounce_count: int = 0
    two_bounce_text: str = ""
    nvz_status_text: str = ""
    nvz_players_in_zone: Optional[List[str]] = None
    nvz_player_states: Optional[Dict[str, Any]] = None
    fault_type: Optional[str] = None
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
            "two_bounce_phase": self.two_bounce_phase,
            "two_bounce_count": self.two_bounce_count,
            "two_bounce_text": self.two_bounce_text,
            "nvz_status_text": self.nvz_status_text,
            "nvz_players_in_zone": self.nvz_players_in_zone or [],
            "nvz_player_states": self.nvz_player_states or {},
            "fault_type": self.fault_type,
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

    # Check if the final ground bounce ends the point inside the court (e.g. dead net drop or unreturned winner)
    has_later_play = any(int(b["frame_idx"]) > b_frame + 4 for b in bounces)
    stream = telemetry_data.get("telemetry_stream", [])
    total_frames = len(stream)
    if (not has_later_play) and (total_frames - b_frame >= 25):
        landing_side = "B" if by < CourtGeometry.NET_Y_FT else "A"
        crossed_after = False
        post_speeds = []
        for fr in stream[b_frame + 1 :]:
            b_rec = fr.get("ball")
            if b_rec and b_rec.get("court_ft"):
                cy = float(b_rec["court_ft"][1])
                side_now = "B" if cy < CourtGeometry.NET_Y_FT else "A"
                if side_now != landing_side:
                    crossed_after = True
                    break
                if b_rec.get("speed_mph") is not None:
                    post_speeds.append(float(b_rec["speed_mph"]))

        if not crossed_after:
            # Check whether the shot originated from the opposite side within the prior 45 frames
            opposite_side = "A" if landing_side == "B" else "B"
            came_from_opposite = False
            for f_pre in range(max(0, b_frame - 45), max(0, b_frame - 8)):
                b_pre = (stream[f_pre].get("ball") or {}) if f_pre < total_frames else {}
                if b_pre.get("court_ft"):
                    py_ft = float(b_pre["court_ft"][1])
                    side_pre = "B" if py_ft < CourtGeometry.NET_Y_FT else "A"
                    if side_pre == opposite_side:
                        came_from_opposite = True
                        break

            if (not came_from_opposite) and post_speeds and min(post_speeds) < 8.0:
                # Net fault: ball failed to cross net and dropped dead on hitting team's own side
                winner = opposite_side
                detail = f"Net fault / dead ball on Team {landing_side} side @ ({bx:.1f}, {by:.1f})ft on f.{b_frame} -> Team {winner} wins rally"
                return b_frame, winner, detail
            elif came_from_opposite and (total_frames - b_frame) >= 32:
                # Unreturned winner bounce on receiving side
                winner = opposite_side
                detail = f"Unreturned winner bounce @ ({bx:.1f}, {by:.1f})ft on f.{b_frame} -> Team {winner} wins rally"
                return b_frame, winner, detail

    return None, None, ""


class MatchScoreTracker:
    """
    Week 4 (W4-02 to W4-04): Autonomous Side-Out, Score Progression, Two-Bounce & NVZ Rule Engine.
    Maps sequential frames + bounce events onto USAP scoreboard progression, side-out ledger,
    Two-Bounce verification, and Kitchen (NVZ) volley foot-fault detection.
    """

    def __init__(self, fsm: Optional[ScoringFSM] = None) -> None:
        self.fsm: Optional[ScoringFSM] = fsm

    def analyze_clip_telemetry(
        self,
        telemetry_data: Dict[str, Any],
        forced_rally_winner: Optional[str] = None,
        assume_serve_start: Optional[bool] = None,
        nvz_detector: Optional[NVZFootFaultDetector] = None,
    ) -> Dict[str, Any]:
        """
        Process a full perception telemetry dictionary (from BasePerceptionPipeline) and emit:
          - Per-frame `FrameRuleSnapshot` entries for video overlay rendering.
          - Initial 0-0-2 state, diagonal serve bounce validation, Two-Bounce status (`W4-03`),
            NVZ kitchen monitoring & foot-faults (`W4-04`), and side-out / score progression.
        """
        stream = telemetry_data.get("telemetry_stream", [])
        if not stream:
            raise ValueError("telemetry_stream is empty; cannot track match scoring.")

        frame0 = stream[0]
        f0_players = frame0.get("players", [])
        f0_ball = frame0.get("ball") or {}
        f0_ball_ft = tuple(f0_ball["court_ft"]) if f0_ball.get("court_ft") else None

        # Scan forward if frame 0 is briefly occluded (e.g., player behind broadcast scorebug)
        seed_players = f0_players
        for fr_cand in stream[:90]:
            pls = fr_cand.get("players", [])
            n_near = sum(
                1 for p in pls
                if ((float(p["court_ft"][1]) >= CourtGeometry.NET_Y_FT) if p.get("court_ft") else (p.get("side") == "near"))
            )
            n_far = len(pls) - n_near
            if n_near >= 2 and n_far >= 2:
                seed_players = pls
                break

        if self.fsm is None:
            self.fsm = ScoringFSM.seed_from_telemetry_players(
                seed_players,
                infer_initial_server=True,
                initial_ball_court_ft=f0_ball_ft,
            )

        initial_serving_team = self.fsm.serving_team
        if assume_serve_start is None:
            assume_serve_start = infer_serve_start_from_telemetry(telemetry_data, initial_serving_team)

        tb_enforcer = TwoBounceEnforcer(
            serving_team=initial_serving_team,
            assume_serve_start=assume_serve_start,
        )
        nvz = nvz_detector if nvz_detector is not None else NVZFootFaultDetector()
        nvz.reset()

        bounces = telemetry_data.get("bounces", [])
        ground_bounces = [b for b in bounces if b.get("is_ground_bounce", True)]
        first_ground_bounce = ground_bounces[0] if ground_bounces else None
        serve_bounce_frame = int(first_ground_bounce["frame_idx"]) if (assume_serve_start and first_ground_bounce) else None

        serve_valid: Optional[bool] = None
        override_serve_box: Optional[str] = None
        if assume_serve_start and first_ground_bounce is not None and first_ground_bounce.get("court_ft"):
            ctx0 = self.fsm.get_serve_context()
            sx, sy = float(first_ground_bounce["court_ft"][0]), float(first_ground_bounce["court_ft"][1])
            serve_valid = _point_in_bounds(sx, sy, ctx0.target_quadrant_ft)
            if not serve_valid:
                alt_box = "LEFT" if ctx0.target_receiver_box == "RIGHT" else "RIGHT"
                alt_quad = SERVICE_QUADRANT_BOUNDS_FT[(ctx0.receiving_team, alt_box)]
                if _point_in_bounds(sx, sy, alt_quad):
                    srv_team = self.fsm.teams[ctx0.serving_team]
                    alt_srv = srv_team.player_in_box(alt_box)
                    cur_srv = ctx0.active_server_id
                    pos_by_id = {
                        str(p.get("slot") or f"P{p.get('id', '?')}"): p.get("court_ft")
                        for p in seed_players if p.get("court_ft")
                    }
                    if alt_srv in pos_by_id and cur_srv in pos_by_id:
                        alt_depth = (pos_by_id[alt_srv][1] - 44.0) if ctx0.serving_team == "A" else (0.0 - pos_by_id[alt_srv][1])
                        cur_depth = (pos_by_id[cur_srv][1] - 44.0) if ctx0.serving_team == "A" else (0.0 - pos_by_id[cur_srv][1])
                        if alt_depth > cur_depth + 4.0:
                            override_serve_box = alt_box
                            serve_valid = True

        # Pre-pass Two-Bounce & NVZ detectors to capture earliest rule violation frame if any
        bounces_by_frame = {int(b["frame_idx"]): b for b in bounces}
        tb_states: Dict[int, Dict[str, Any]] = {}
        nvz_states: Dict[int, Dict[str, Any]] = {}
        for fr in stream:
            f_idx = int(fr["frame_idx"])
            ts = float(fr.get("timestamp_s", f_idx / 30.0))
            bounce = fr.get("bounce") or bounces_by_frame.get(f_idx)
            tb_st = tb_enforcer.update(
                frame_idx=f_idx,
                timestamp_s=ts,
                ball=fr.get("ball"),
                bounce=bounce,
                players=fr.get("players", []),
            )
            nvz_st = nvz.update(
                frame_idx=f_idx,
                timestamp_s=ts,
                players=fr.get("players", []),
                ball=fr.get("ball"),
                bounce=bounce,
            )
            tb_states[f_idx] = tb_st.to_dict()
            nvz_states[f_idx] = nvz_st

        # Resolve earliest terminal rally outcome (W4-03 Two-Bounce fault, W4-04 NVZ fault, or OOB)
        terminal_frame, inferred_winner, terminal_reason = _infer_rally_terminal_winner(telemetry_data, self.fsm)
        terminal_fault_type: Optional[str] = "OUT_OF_BOUNDS" if terminal_frame is not None else None

        candidates: List[Tuple[int, str, str, str]] = []
        if tb_enforcer.fault_event is not None:
            fe = tb_enforcer.fault_event
            candidates.append((fe.frame_idx, fe.rally_winner_team or "A", fe.reason, fe.event_type))
        if nvz.fault_event is not None:
            fe_n = nvz.fault_event
            candidates.append((fe_n.frame_idx, fe_n.rally_winner_team, fe_n.reason, fe_n.fault_type))
        if terminal_frame is not None and inferred_winner is not None:
            candidates.append((terminal_frame, inferred_winner, terminal_reason, "OUT_OF_BOUNDS"))

        if candidates:
            candidates.sort(key=lambda item: item[0])
            terminal_frame, inferred_winner, terminal_reason, terminal_fault_type = candidates[0]

        if forced_rally_winner in ("A", "B"):
            terminal_frame = int(bounces[-1]["frame_idx"]) if bounces else int(stream[-1]["frame_idx"])
            inferred_winner = forced_rally_winner
            terminal_reason = f"Explicit rally winner: Team {forced_rally_winner}"
            terminal_fault_type = "EXPLICIT_OUTCOME"

        snapshots: List[FrameRuleSnapshot] = []
        transition_applied = False
        contact_count = 0
        max_concurrent_nvz = 0

        for fr in stream:
            f_idx = int(fr["frame_idx"])
            ts = float(fr.get("timestamp_s", f_idx / 30.0))

            if f_idx in bounces_by_frame:
                contact_count += 1

            emitted_dict: Optional[Dict[str, Any]] = None
            frame_fault: Optional[str] = None
            if (not transition_applied) and (terminal_frame is not None) and (f_idx >= terminal_frame) and (inferred_winner is not None):
                ev = self.fsm.advance_rally(inferred_winner, notes=terminal_reason)
                emitted_dict = ev.to_dict()
                transition_applied = True
                frame_fault = terminal_fault_type

            ctx = self.fsm.get_serve_context()
            tb_f = tb_states[f_idx]
            nvz_f = nvz_states[f_idx]
            max_concurrent_nvz = max(max_concurrent_nvz, len(nvz_f["players_in_nvz"]))

            disp_srv_box = ctx.server_box
            disp_rcv_box = ctx.target_receiver_box
            disp_srv_id = ctx.active_server_id
            disp_rcv_id = ctx.active_receiver_id
            disp_quad = ctx.target_quadrant_ft
            if (not transition_applied) and override_serve_box is not None:
                disp_srv_box = override_serve_box
                disp_rcv_box = override_serve_box
                disp_srv_id = self.fsm.teams[ctx.serving_team].player_in_box(override_serve_box)
                disp_rcv_id = self.fsm.teams[ctx.receiving_team].player_in_box(override_serve_box)
                disp_quad = SERVICE_QUADRANT_BOUNDS_FT[(ctx.receiving_team, override_serve_box)]

            if transition_applied:
                ev_last = self.fsm.history[-1]
                prefix = ""
                if terminal_fault_type in ("SERVE_VOLLEY_FAULT", "RETURN_VOLLEY_FAULT"):
                    prefix = "2-BOUNCE FAULT! "
                elif terminal_fault_type and terminal_fault_type.startswith("NVZ_"):
                    prefix = "KITCHEN FOOT-FAULT! "

                if ev_last.event_type == "SIDE_OUT":
                    rally_state = "RALLY_ENDED_SIDE_OUT"
                    banner = f"{prefix}SIDE-OUT! {ev_last.prev_score_call} -> {ctx.score_call} (Next: Team {ctx.serving_team} {ctx.active_server_id} RIGHT)"
                elif ev_last.event_type == "GAME_WON":
                    rally_state = "MATCH_OVER"
                    banner = f"{prefix}GAME OVER! Team {ctx.winner_team} wins ({ctx.score_call})"
                else:
                    rally_state = "RALLY_ENDED_POINT"
                    banner = f"{prefix}POINT TEAM {ctx.serving_team}! {ev_last.prev_score_call} -> {ctx.score_call}"
            elif serve_bounce_frame is not None and f_idx < serve_bounce_frame:
                rally_state = "SERVE_IN_FLIGHT"
                banner = f"SERVE {ctx.score_call}: {disp_srv_id} ({disp_srv_box}) -> {disp_rcv_id} ({disp_rcv_box})"
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
                    active_server_id=disp_srv_id,
                    active_receiver_id=disp_rcv_id,
                    server_box=disp_srv_box,
                    target_receiver_box=disp_rcv_box,
                    target_quadrant_ft=disp_quad,
                    side_out_count=self.fsm.side_out_count,
                    rally_count=self.fsm.rally_count,
                    serve_valid_quadrant=serve_valid,
                    status_banner=banner,
                    two_bounce_phase=tb_f["phase"],
                    two_bounce_count=tb_f["bounces_completed"],
                    two_bounce_text=tb_f["status_text"],
                    nvz_status_text=nvz_f["status_text"],
                    nvz_players_in_zone=nvz_f["players_in_nvz"],
                    nvz_player_states=nvz_f["player_states"],
                    fault_type=terminal_fault_type if transition_applied else None,
                    event_emitted=emitted_dict,
                )
            )

        return {
            "phase": "Week 4 Day 3 (W4-01 to W4-04)",
            "video_path": telemetry_data.get("video_path"),
            "total_frames": len(snapshots),
            "serve_bounce_frame": serve_bounce_frame,
            "serve_valid_diagonal": serve_valid,
            "terminal_event_frame": terminal_frame,
            "terminal_fault_type": terminal_fault_type,
            "fsm_summary": self.fsm.to_dict(),
            "two_bounce_summary": {
                "assume_serve_start": assume_serve_start,
                "final_phase": tb_enforcer.phase,
                "bounces_completed": tb_enforcer.bounces_completed,
                "serve_bounce_frame": tb_enforcer.serve_bounce_frame,
                "return_bounce_frame": tb_enforcer.return_bounce_frame,
                "satisfied_frame": tb_enforcer.satisfied_frame,
                "fault_event": tb_enforcer.fault_event.to_dict() if tb_enforcer.fault_event else None,
            },
            "nvz_summary": {
                "fault_detected": nvz.fault_event is not None,
                "fault_event": nvz.fault_event.to_dict() if nvz.fault_event else None,
                "max_concurrent_nvz_players": max_concurrent_nvz,
            },
            "frame_rules": [s.to_dict() for s in snapshots],
        }
