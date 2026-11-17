"""
nvz_detector.py - Week 4 Day 3 (W4-04, Day 16): Official USAP Kitchen (NVZ) Foot-Fault Detector.

Implements USAP Rule 9 (Non-Volley Zone Rules):
  - Rule 9.A: Touching the NVZ surface or its boundary line (`Y = 15.0 ft` or `Y = 29.0 ft`)
              during a volley act is a fault (`NVZ_DIRECT_VOLLEY_FAULT`).
  - Rule 9.B: Momentum carrying a player onto or into the NVZ after executing a volley,
              even after the ball is dead or cleanly over the net, is a fault (`NVZ_MOMENTUM_FAULT`).
  - Rule 9.D: Re-entering from the NVZ requires re-establishing both feet completely outside
              the NVZ before volleying (`NVZ_NO_RESET_FAULT`).
  - Rule 9.A (Exception): Entering the NVZ to play a ball that has bounced is 100% legal
              (`IN_KITCHEN_LEGAL`) and distinguished cleanly from volley violations.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math

from pbbase.court.homography import CourtGeometry


NVZ_FAR_LINE_Y_FT = CourtGeometry.NET_Y_FT - CourtGeometry.NVZ_DEPTH_FT   # 15.0 ft
NVZ_NEAR_LINE_Y_FT = CourtGeometry.NET_Y_FT + CourtGeometry.NVZ_DEPTH_FT  # 29.0 ft


@dataclass
class NVZFaultEvent:
    """Structured Kitchen (NVZ) Foot-Fault audit record."""
    frame_idx: int
    timestamp_s: float
    volley_frame_idx: int
    fault_type: str                 # NVZ_DIRECT_VOLLEY_FAULT | NVZ_MOMENTUM_FAULT | NVZ_NO_RESET_FAULT
    faulting_team: str              # "A" or "B"
    rally_winner_team: str          # Opposite team
    offending_player_id: str
    foot_court_ft: Tuple[float, float]
    nvz_penetration_ft: float       # Depth crossed past NVZ line (>= 0 means touching/inside NVZ)
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": int(self.frame_idx),
            "timestamp_s": round(float(self.timestamp_s), 4),
            "volley_frame_idx": int(self.volley_frame_idx),
            "fault_type": self.fault_type,
            "faulting_team": self.faulting_team,
            "rally_winner_team": self.rally_winner_team,
            "offending_player_id": self.offending_player_id,
            "foot_court_ft": [round(float(v), 2) for v in self.foot_court_ft],
            "nvz_penetration_ft": round(float(self.nvz_penetration_ft), 2),
            "reason": self.reason,
        }


def compute_nvz_clearance_ft(
    foot_xy_ft: Tuple[float, float],
    line_tolerance_ft: float = 0.0,
) -> Tuple[str, float, bool]:
    """
    Evaluate a player's foot contact `(X, Y)` in feet against their side's Kitchen (NVZ).

    Returns `(half, clearance_ft, is_in_nvz)`:
      - `half`: `'A'` (`Y >= 22.0`) or `'B'` (`Y < 22.0`)
      - `clearance_ft`: distance outside NVZ line (`> 0` safe behind line; `<= 0` on/inside NVZ)
      - `is_in_nvz`: `True` when inside `[0, 20]` width margin and `clearance_ft <= line_tolerance_ft`
    """
    x_ft, y_ft = float(foot_xy_ft[0]), float(foot_xy_ft[1])
    in_width = (-0.25 <= x_ft <= CourtGeometry.WIDTH_FT + 0.25)

    if y_ft >= CourtGeometry.NET_Y_FT:
        half = "A"
        clearance = y_ft - NVZ_NEAR_LINE_Y_FT
        # Guard against far court projection wrap-around past net
        in_depth = (y_ft >= CourtGeometry.NET_Y_FT - 0.25) and (clearance <= line_tolerance_ft)
    else:
        half = "B"
        clearance = NVZ_FAR_LINE_Y_FT - y_ft
        in_depth = (y_ft <= CourtGeometry.NET_Y_FT + 0.25) and (clearance <= line_tolerance_ft)

    return half, float(clearance), bool(in_width and in_depth)


def _find_volley_striker(
    players: List[Dict[str, Any]],
    ball_ft: Tuple[float, float],
    half: str,
    max_reach_ft: float = 11.0,
) -> Optional[Dict[str, Any]]:
    """Find the player on court half `half` closest to the volley point `ball_ft`."""
    bx, by = float(ball_ft[0]), float(ball_ft[1])
    best_p: Optional[Dict[str, Any]] = None
    best_d = float("inf")
    for p in players:
        cf = p.get("court_ft")
        if not cf or len(cf) < 2:
            continue
        px, py = float(cf[0]), float(cf[1])
        p_half = "A" if py >= CourtGeometry.NET_Y_FT else "B"
        if p_half != half:
            continue
        d = math.hypot(px - bx, py - by)
        if d < best_d and d <= max_reach_ft:
            best_d = d
            best_p = p
    return best_p


class NVZFootFaultDetector:
    """
    Week 4 Day 3 (W4-04): USAP Rule 9 Non-Volley Zone (Kitchen) Foot-Fault Detector.

    Monitors all four player foot projections against `Y = 15.0 ft` and `Y = 29.0 ft` across
    every frame, tracking outside-reset state and linking confirmed airborne volleys to detect:
      1. Direct NVZ volley foot-faults (`NVZ_DIRECT_VOLLEY_FAULT`)
      2. Post-volley momentum carry-ins (`NVZ_MOMENTUM_FAULT`)
      3. Non-reset kitchen re-entry volleys (`NVZ_NO_RESET_FAULT`)
    While preserving legal kitchen entries (`IN_KITCHEN_LEGAL`) when no unbounced volley occurs.
    """

    def __init__(
        self,
        line_tolerance_ft: float = 0.0,
        edge_warning_ft: float = 1.25,
        reset_frames_required: int = 2,
        momentum_window_frames: int = 30,
        min_volley_height_ft: float = 2.0,
    ) -> None:
        self.line_tolerance_ft: float = float(line_tolerance_ft)
        self.edge_warning_ft: float = float(edge_warning_ft)
        self.reset_frames_required: int = int(reset_frames_required)
        self.momentum_window_frames: int = int(momentum_window_frames)
        self.min_volley_height_ft: float = float(min_volley_height_ft)
        self.reset()

    def reset(self) -> None:
        """Reset per-player NVZ state and active volley momentum trackers."""
        self.consecutive_outside_frames: Dict[str, int] = {}
        self.needs_outside_reset: Dict[str, bool] = {}
        self.pending_momentum_volleys: Dict[str, int] = {}  # slot -> volley_frame_idx
        self.last_ball_half: Optional[str] = None
        self.bounced_on_half: Dict[str, bool] = {"A": True, "B": True}
        self.last_ground_bounce_frame: Dict[str, int] = {"A": -999, "B": -999}
        self.fault_event: Optional[NVZFaultEvent] = None

    def _is_true_volley(
        self,
        bounce: Dict[str, Any],
        ball: Optional[Dict[str, Any]],
    ) -> bool:
        """
        Return True only for a genuine unbounced airborne volley contact (not a ground bounce
        or low court-level pickup below `min_volley_height_ft`).
        """
        if bounce.get("is_ground_bounce", True):
            return False
        if ball is not None and ball.get("height_ft") is not None:
            if float(ball["height_ft"]) < self.min_volley_height_ft:
                return False
        return True

    def update(
        self,
        frame_idx: int,
        timestamp_s: float,
        players: List[Dict[str, Any]],
        ball: Optional[Dict[str, Any]] = None,
        bounce: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate NVZ zone state and foot-faults for one telemetry frame."""
        # 1. Track net crossings & ground bounces to understand half bounce state
        if ball is not None and ball.get("court_ft") is not None:
            cur_half = "A" if float(ball["court_ft"][1]) >= CourtGeometry.NET_Y_FT else "B"
            if self.last_ball_half is not None and cur_half != self.last_ball_half:
                self.bounced_on_half[cur_half] = False
                self.last_ground_bounce_frame[cur_half] = -999
            self.last_ball_half = cur_half

        if bounce is not None and bounce.get("is_ground_bounce", True) and bounce.get("court_ft") is not None:
            bh = "A" if float(bounce["court_ft"][1]) >= CourtGeometry.NET_Y_FT else "B"
            self.bounced_on_half[bh] = True
            self.last_ground_bounce_frame[bh] = frame_idx

        # 2. Evaluate all players against NVZ boundaries
        player_states: Dict[str, Dict[str, Any]] = {}
        players_in_nvz: List[str] = []
        players_on_edge: List[str] = []

        for p in players:
            cf = p.get("court_ft")
            if not cf or len(cf) < 2:
                continue
            slot = str(p.get("slot") or f"P{p.get('id', '?')}")
            half, clearance, in_nvz = compute_nvz_clearance_ft(tuple(cf), self.line_tolerance_ft)

            if in_nvz:
                prior_needs_reset = True
                self.consecutive_outside_frames[slot] = 0
                self.needs_outside_reset[slot] = True
                players_in_nvz.append(slot)
                status = "IN_KITCHEN_LEGAL"
            else:
                prior_needs_reset = bool(self.needs_outside_reset.get(slot, False))
                cnt = self.consecutive_outside_frames.get(slot, 0) + 1
                self.consecutive_outside_frames[slot] = cnt
                if cnt >= self.reset_frames_required:
                    self.needs_outside_reset[slot] = False
                if clearance <= self.edge_warning_ft:
                    players_on_edge.append(slot)
                    status = "NVZ_EDGE"
                else:
                    status = "CLEAR"

            # Check active momentum fault if this player struck a volley recently
            if self.fault_event is None and slot in self.pending_momentum_volleys:
                v_frame = self.pending_momentum_volleys[slot]
                if (frame_idx - v_frame) <= self.momentum_window_frames:
                    if in_nvz:
                        winner = "B" if half == "A" else "A"
                        pen = max(0.0, -clearance)
                        self.fault_event = NVZFaultEvent(
                            frame_idx=frame_idx,
                            timestamp_s=timestamp_s,
                            volley_frame_idx=v_frame,
                            fault_type="NVZ_MOMENTUM_FAULT",
                            faulting_team=half,
                            rally_winner_team=winner,
                            offending_player_id=slot,
                            foot_court_ft=(float(cf[0]), float(cf[1])),
                            nvz_penetration_ft=pen,
                            reason=(
                                f"Kitchen Momentum Fault (Rule 9.B): {slot} volleyed @ f.{v_frame} "
                                f"and momentum entered NVZ ({pen:.2f}ft inside) @ f.{frame_idx}"
                            ),
                        )
                        status = "NVZ_FAULT"
                else:
                    del self.pending_momentum_volleys[slot]

            player_states[slot] = {
                "team": half,
                "foot_ft": [round(float(cf[0]), 2), round(float(cf[1]), 2)],
                "clearance_ft": round(clearance, 2),
                "in_nvz": in_nvz,
                "needs_reset": prior_needs_reset,
                "status": status,
            }

        # 3. Evaluate volley contact on this frame (excluding Rule 9.E groundstrokes played off a recent bounce)
        if self.fault_event is None and bounce is not None and self._is_true_volley(bounce, ball):
            c_ft_raw = bounce.get("court_ft") or (ball.get("court_ft") if ball else None)
            if c_ft_raw is not None:
                c_ft = (float(c_ft_raw[0]), float(c_ft_raw[1]))
                v_half = "A" if c_ft[1] >= CourtGeometry.NET_Y_FT else "B"
                since_bounce = frame_idx - self.last_ground_bounce_frame.get(v_half, -999)
                b_height = float(ball.get("height_ft") or 0.0) if ball is not None else 0.0
                is_post_bounce_pickup = (0 < since_bounce <= 12) and (b_height < 2.2)
                striker = None if is_post_bounce_pickup else _find_volley_striker(players, c_ft, v_half)
                if striker is not None:
                    s_slot = str(striker.get("slot") or f"P{striker.get('id', '?')}")
                    s_state = player_states.get(s_slot)
                    if s_state is not None:
                        winner = "B" if v_half == "A" else "A"
                        if s_state["in_nvz"]:
                            pen = max(0.0, -s_state["clearance_ft"])
                            self.fault_event = NVZFaultEvent(
                                frame_idx=frame_idx,
                                timestamp_s=timestamp_s,
                                volley_frame_idx=frame_idx,
                                fault_type="NVZ_DIRECT_VOLLEY_FAULT",
                                faulting_team=v_half,
                                rally_winner_team=winner,
                                offending_player_id=s_slot,
                                foot_court_ft=tuple(s_state["foot_ft"]),
                                nvz_penetration_ft=pen,
                                reason=(
                                    f"Kitchen Foot-Fault (Rule 9.A): {s_slot} volleyed while inside/on NVZ "
                                    f"({pen:.2f}ft past line) @ f.{frame_idx}"
                                ),
                            )
                            s_state["status"] = "NVZ_FAULT"
                        elif s_state["needs_reset"]:
                            self.fault_event = NVZFaultEvent(
                                frame_idx=frame_idx,
                                timestamp_s=timestamp_s,
                                volley_frame_idx=frame_idx,
                                fault_type="NVZ_NO_RESET_FAULT",
                                faulting_team=v_half,
                                rally_winner_team=winner,
                                offending_player_id=s_slot,
                                foot_court_ft=tuple(s_state["foot_ft"]),
                                nvz_penetration_ft=0.0,
                                reason=(
                                    f"Kitchen Re-Entry Fault (Rule 9.D): {s_slot} exited NVZ without "
                                    f"re-establishing {self.reset_frames_required} frames outside before volley @ f.{frame_idx}"
                                ),
                            )
                            s_state["status"] = "NVZ_FAULT"
                        else:
                            # Valid volley outside NVZ -> arm momentum monitor for this player
                            self.pending_momentum_volleys[s_slot] = frame_idx

        # Guarantee fault state highlights offending player
        if self.fault_event is not None:
            off_id = self.fault_event.offending_player_id
            if off_id in player_states:
                player_states[off_id]["status"] = "NVZ_FAULT"

        # 4. Construct crisp HUD readout
        if self.fault_event is not None:
            status_text = (
                f"KITCHEN FAULT! {self.fault_event.offending_player_id} "
                f"({self.fault_event.fault_type})"
            )
        elif players_in_nvz:
            joined = ",".join(players_in_nvz)
            status_text = f"NVZ: {joined} IN KITCHEN (LEGAL - NO VOLLEY)"
        elif players_on_edge:
            joined = ",".join(
                f"{sl}(+{player_states[sl]['clearance_ft']:.1f}ft)" for sl in players_on_edge
            )
            status_text = f"NVZ WATCH: {joined}"
        else:
            status_text = "NVZ KITCHEN: ALL CLEAR"

        return {
            "frame_idx": frame_idx,
            "fault_detected": self.fault_event is not None,
            "fault_event": self.fault_event.to_dict() if self.fault_event else None,
            "players_in_nvz": players_in_nvz,
            "players_on_edge": players_on_edge,
            "player_states": player_states,
            "status_text": status_text,
        }

    @classmethod
    def analyze_telemetry(cls, telemetry_data: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """Run full NVZ Kitchen Foot-Fault evaluation over a telemetry stream."""
        detector = cls(**kwargs)
        stream = telemetry_data.get("telemetry_stream", [])
        bounces_by_frame = {int(b["frame_idx"]): b for b in telemetry_data.get("bounces", [])}

        frame_states: List[Dict[str, Any]] = []
        max_nvz_occupancy = 0
        for fr in stream:
            fi = int(fr["frame_idx"])
            ts = float(fr.get("timestamp_s", fi / 30.0))
            bounce = fr.get("bounce") or bounces_by_frame.get(fi)
            res = detector.update(
                frame_idx=fi,
                timestamp_s=ts,
                players=fr.get("players", []),
                ball=fr.get("ball"),
                bounce=bounce,
            )
            max_nvz_occupancy = max(max_nvz_occupancy, len(res["players_in_nvz"]))
            frame_states.append(res)

        return {
            "fault_detected": detector.fault_event is not None,
            "fault_event": detector.fault_event.to_dict() if detector.fault_event else None,
            "max_concurrent_nvz_players": max_nvz_occupancy,
            "frame_states": frame_states,
        }
