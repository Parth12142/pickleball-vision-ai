"""
two_bounce.py - Week 4 Day 3 (W4-03, Day 16): Official USAP Two-Bounce Rule Enforcer.

Implements USAP Rule 7.A / 4.A.2 (The Two-Bounce Rule):
  1. Following a serve, the receiving team MUST allow the ball to bounce once before returning it.
  2. Following the return, the serving team MUST allow the ball to bounce once before hitting it.
  3. After both bounces have occurred (two bounces total, one on each side), either team may
     volley the ball (outside the Non-Volley Zone) or play it off a bounce.

Emits frame-by-frame Two-Bounce progression states (`TwoBounceFrameState`) and structured
violation events (`TwoBounceEvent`: `SERVE_VOLLEY_FAULT` or `RETURN_VOLLEY_FAULT`) whenever
either team volleys prematurely before its mandatory ground bounce.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math

from pbbase.court.homography import CourtGeometry


TWO_BOUNCE_PHASES = (
    "AWAITING_SERVE_BOUNCE",
    "SERVE_BOUNCED_AWAITING_RETURN",
    "AWAITING_RETURN_BOUNCE",
    "TWO_BOUNCE_SATISFIED",
    "MID_RALLY_OPEN",
    "VIOLATION_TWO_BOUNCE_FAULT",
)


@dataclass
class TwoBounceEvent:
    """Structured violation (or completion) record emitted by the Two-Bounce Rule Enforcer."""
    frame_idx: int
    timestamp_s: float
    event_type: str                 # SERVE_VOLLEY_FAULT | RETURN_VOLLEY_FAULT | TWO_BOUNCE_COMPLETE
    faulting_team: Optional[str]    # "A" or "B" on violation, None on completion
    rally_winner_team: Optional[str]
    offending_player_id: Optional[str]
    court_ft: Optional[Tuple[float, float]]
    phase_when_triggered: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": int(self.frame_idx),
            "timestamp_s": round(float(self.timestamp_s), 4),
            "event_type": self.event_type,
            "faulting_team": self.faulting_team,
            "rally_winner_team": self.rally_winner_team,
            "offending_player_id": self.offending_player_id,
            "court_ft": [round(float(v), 2) for v in self.court_ft] if self.court_ft else None,
            "phase_when_triggered": self.phase_when_triggered,
            "reason": self.reason,
        }


@dataclass
class TwoBounceFrameState:
    """Frame-aligned state emitted per telemetry frame for HUD rendering & rules logging."""
    frame_idx: int
    phase: str
    bounces_completed: int          # 0, 1, or 2
    serve_bounce_frame: Optional[int]
    return_bounce_frame: Optional[int]
    satisfied_frame: Optional[int]
    fault_detected: bool
    fault_event: Optional[Dict[str, Any]]
    status_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": int(self.frame_idx),
            "phase": self.phase,
            "bounces_completed": int(self.bounces_completed),
            "serve_bounce_frame": self.serve_bounce_frame,
            "return_bounce_frame": self.return_bounce_frame,
            "satisfied_frame": self.satisfied_frame,
            "fault_detected": bool(self.fault_detected),
            "fault_event": self.fault_event,
            "status_text": self.status_text,
        }


def _half_of_y(y_ft: float) -> str:
    """Return court half ('A' near >= 22 ft, 'B' far < 22 ft)."""
    return "A" if float(y_ft) >= CourtGeometry.NET_Y_FT else "B"


def _nearest_player_on_half(
    players: List[Dict[str, Any]],
    ball_ft: Tuple[float, float],
    half: str,
) -> Optional[str]:
    """Identify the slot (P1-P4) on `half` nearest to `ball_ft`."""
    bx, by = float(ball_ft[0]), float(ball_ft[1])
    best_slot: Optional[str] = None
    best_dist = float("inf")
    for p in players:
        cf = p.get("court_ft")
        if not cf or len(cf) < 2:
            continue
        px, py = float(cf[0]), float(cf[1])
        p_half = "A" if py >= CourtGeometry.NET_Y_FT else "B"
        if p_half != half:
            continue
        dist = math.hypot(px - bx, py - by)
        if dist < best_dist:
            best_dist = dist
            best_slot = str(p.get("slot") or f"P{p.get('id', '?')}")
    return best_slot


def _is_low_ground_pickup(
    bounce: Dict[str, Any],
    ball: Optional[Dict[str, Any]],
    players: List[Dict[str, Any]],
    max_ground_pickup_height_ft: float,
) -> bool:
    """
    Disambiguate a true airborne volley vs. a low groundstroke / half-volley bounce pickup.
    When an active player is crouched low next to a ground bounce, 2D player bounding boxes
    can overlap the bounce pixel and cause `is_ground_bounce=False`. Checking 3D height
    (when available) and deep-court ground proximity avoids false two-bounce faults on low pickups.
    """
    if bounce.get("is_ground_bounce", True):
        return True

    if ball is not None and ball.get("height_ft") is not None:
        h_ft = float(ball["height_ft"])
        if h_ft <= max_ground_pickup_height_ft:
            return True

    return False


def infer_serve_start_from_telemetry(
    telemetry_data: Dict[str, Any],
    serving_team: str,
) -> bool:
    """
    Determine whether a clip starts at an opening serve vs. mid-rally cut.
    A serve start has:
      - Serving team starting deep near baseline OR first bounce event on receiving half is a ground bounce.
      - Contrast: mid-rally clips (e.g., highlight cuts) start with low mid-court balls and unbounced exchanges.
    """
    stream = telemetry_data.get("telemetry_stream", [])
    bounces = telemetry_data.get("bounces", [])
    if not stream or not bounces:
        return True

    first_event = bounces[0]
    ev_ft = first_event.get("court_ft")
    if not ev_ft:
        return True

    receiving_team = "B" if serving_team == "A" else "A"
    ev_half = _half_of_y(float(ev_ft[1]))

    # If the very first contact/bounce is a non-ground paddle hit (mid-rally cut) before any ground bounce
    if not first_event.get("is_ground_bounce", True):
        return False

    # First ground bounce occurred on receiving half -> matches canonical serve start
    if ev_half == receiving_team:
        return True

    return False


class TwoBounceEnforcer:
    """
    Week 4 Day 3 (W4-03): Stateful USAP Two-Bounce Rule Enforcer.

    Evaluates frame-by-frame perception telemetry (`ball`, `bounce`, `players`) from the
    start of a serve until either:
      - Both the serve bounce (receiving half) and return bounce (serving half) complete
        (`TWO_BOUNCE_SATISFIED`), unlocking open volley play, OR
      - A player executes an illegal premature volley (`SERVE_VOLLEY_FAULT` or `RETURN_VOLLEY_FAULT`).
    """

    def __init__(
        self,
        serving_team: str = "A",
        assume_serve_start: bool = True,
        max_ground_pickup_height_ft: float = 1.95,
    ) -> None:
        if serving_team not in ("A", "B"):
            raise ValueError(f"serving_team must be 'A' or 'B', got '{serving_team}'.")
        self.serving_team: str = serving_team
        self.receiving_team: str = "B" if serving_team == "A" else "A"
        self.max_ground_pickup_height_ft: float = float(max_ground_pickup_height_ft)

        self.phase: str = "AWAITING_SERVE_BOUNCE" if assume_serve_start else "MID_RALLY_OPEN"
        self.bounces_completed: int = 0 if assume_serve_start else 2
        self.serve_bounce_frame: Optional[int] = None
        self.return_bounce_frame: Optional[int] = None
        self.satisfied_frame: Optional[int] = 0 if not assume_serve_start else None
        self.fault_event: Optional[TwoBounceEvent] = None
        self.events: List[TwoBounceEvent] = []
        self._last_ball_half: Optional[str] = None
        self._cross_return_frame: Optional[int] = None

    def _emit_fault(
        self,
        frame_idx: int,
        timestamp_s: float,
        fault_type: str,
        faulting_team: str,
        players: List[Dict[str, Any]],
        court_ft: Tuple[float, float],
        reason: str,
    ) -> TwoBounceEvent:
        prev_phase = self.phase
        self.phase = "VIOLATION_TWO_BOUNCE_FAULT"
        winner = "B" if faulting_team == "A" else "A"
        offender = _nearest_player_on_half(players, court_ft, faulting_team)
        ev = TwoBounceEvent(
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            event_type=fault_type,
            faulting_team=faulting_team,
            rally_winner_team=winner,
            offending_player_id=offender,
            court_ft=court_ft,
            phase_when_triggered=prev_phase,
            reason=reason,
        )
        self.fault_event = ev
        self.events.append(ev)
        return ev

    def update(
        self,
        frame_idx: int,
        timestamp_s: float,
        ball: Optional[Dict[str, Any]],
        bounce: Optional[Dict[str, Any]],
        players: List[Dict[str, Any]],
    ) -> TwoBounceFrameState:
        """Advance Two-Bounce state by one telemetry frame."""
        # Track current ball half to detect return crossing net back to serving team
        if ball is not None and ball.get("court_ft") is not None:
            bx, by = float(ball["court_ft"][0]), float(ball["court_ft"][1])
            cur_half = _half_of_y(by)

            if self.phase == "SERVE_BOUNCED_AWAITING_RETURN":
                # Once ball crosses net from receiving half back into serving half, awaiting 2nd bounce
                if cur_half == self.serving_team and (
                    self._last_ball_half == self.receiving_team
                    or (self.serve_bounce_frame is not None and frame_idx > self.serve_bounce_frame + 5)
                ):
                    self.phase = "AWAITING_RETURN_BOUNCE"
                    self._cross_return_frame = frame_idx

            self._last_ball_half = cur_half

        # Evaluate bounce / paddle contact event if present on this frame
        if bounce is not None and self.phase not in ("VIOLATION_TWO_BOUNCE_FAULT", "TWO_BOUNCE_SATISFIED", "MID_RALLY_OPEN"):
            c_ft_raw = bounce.get("court_ft") or (ball.get("court_ft") if ball else None)
            if c_ft_raw is not None:
                c_ft = (float(c_ft_raw[0]), float(c_ft_raw[1]))
                ev_half = _half_of_y(c_ft[1])
                is_ground = _is_low_ground_pickup(bounce, ball, players, self.max_ground_pickup_height_ft)

                if self.phase == "AWAITING_SERVE_BOUNCE":
                    if ev_half == self.receiving_team:
                        if is_ground:
                            self.serve_bounce_frame = frame_idx
                            self.bounces_completed = 1
                            self.phase = "SERVE_BOUNCED_AWAITING_RETURN"
                        else:
                            self._emit_fault(
                                frame_idx=frame_idx,
                                timestamp_s=timestamp_s,
                                fault_type="SERVE_VOLLEY_FAULT",
                                faulting_team=self.receiving_team,
                                players=players,
                                court_ft=c_ft,
                                reason=(
                                    f"Two-Bounce Fault: Receiving Team {self.receiving_team} volleyed "
                                    f"serve before bounce @ ({c_ft[0]:.1f}, {c_ft[1]:.1f})ft"
                                ),
                            )

                elif self.phase == "SERVE_BOUNCED_AWAITING_RETURN":
                    # Check if ball already crossed to serving half or returned early
                    if ev_half == self.serving_team:
                        if is_ground:
                            self.return_bounce_frame = frame_idx
                            self.bounces_completed = 2
                            self.satisfied_frame = frame_idx
                            self.phase = "TWO_BOUNCE_SATISFIED"
                            self.events.append(
                                TwoBounceEvent(
                                    frame_idx=frame_idx,
                                    timestamp_s=timestamp_s,
                                    event_type="TWO_BOUNCE_COMPLETE",
                                    faulting_team=None,
                                    rally_winner_team=None,
                                    offending_player_id=None,
                                    court_ft=c_ft,
                                    phase_when_triggered="SERVE_BOUNCED_AWAITING_RETURN",
                                    reason=f"Two-Bounce Rule satisfied @ ({c_ft[0]:.1f}, {c_ft[1]:.1f})ft",
                                )
                            )
                        else:
                            self._emit_fault(
                                frame_idx=frame_idx,
                                timestamp_s=timestamp_s,
                                fault_type="RETURN_VOLLEY_FAULT",
                                faulting_team=self.serving_team,
                                players=players,
                                court_ft=c_ft,
                                reason=(
                                    f"Two-Bounce Fault: Serving Team {self.serving_team} volleyed "
                                    f"return before bounce @ ({c_ft[0]:.1f}, {c_ft[1]:.1f})ft"
                                ),
                            )

                elif self.phase == "AWAITING_RETURN_BOUNCE":
                    if ev_half == self.serving_team:
                        if is_ground:
                            self.return_bounce_frame = frame_idx
                            self.bounces_completed = 2
                            self.satisfied_frame = frame_idx
                            self.phase = "TWO_BOUNCE_SATISFIED"
                            self.events.append(
                                TwoBounceEvent(
                                    frame_idx=frame_idx,
                                    timestamp_s=timestamp_s,
                                    event_type="TWO_BOUNCE_COMPLETE",
                                    faulting_team=None,
                                    rally_winner_team=None,
                                    offending_player_id=None,
                                    court_ft=c_ft,
                                    phase_when_triggered="AWAITING_RETURN_BOUNCE",
                                    reason=f"Two-Bounce Rule satisfied @ ({c_ft[0]:.1f}, {c_ft[1]:.1f})ft",
                                )
                            )
                        else:
                            self._emit_fault(
                                frame_idx=frame_idx,
                                timestamp_s=timestamp_s,
                                fault_type="RETURN_VOLLEY_FAULT",
                                faulting_team=self.serving_team,
                                players=players,
                                court_ft=c_ft,
                                reason=(
                                    f"Two-Bounce Fault: Serving Team {self.serving_team} volleyed "
                                    f"return before bounce @ ({c_ft[0]:.1f}, {c_ft[1]:.1f})ft"
                                ),
                            )

        # Build crisp human-readable HUD string
        if self.phase == "VIOLATION_TWO_BOUNCE_FAULT" and self.fault_event is not None:
            status = f"2-BOUNCE FAULT: Team {self.fault_event.faulting_team} ({self.fault_event.event_type})"
        elif self.phase == "TWO_BOUNCE_SATISFIED":
            status = "2-BOUNCE RULE: 2/2 CLEARED (VOLLEYS OPEN)"
        elif self.phase == "MID_RALLY_OPEN":
            status = "2-BOUNCE RULE: MID-RALLY (VOLLEYS OPEN)"
        elif self.phase == "AWAITING_RETURN_BOUNCE":
            status = f"2-BOUNCE RULE: 1/2 (WAIT TEAM {self.serving_team} BOUNCE)"
        elif self.phase == "SERVE_BOUNCED_AWAITING_RETURN":
            status = f"2-BOUNCE RULE: 1/2 SERVE BOUNCED (RETURN)"
        else:
            status = f"2-BOUNCE RULE: 0/2 (WAIT TEAM {self.receiving_team} BOUNCE)"

        return TwoBounceFrameState(
            frame_idx=frame_idx,
            phase=self.phase,
            bounces_completed=self.bounces_completed,
            serve_bounce_frame=self.serve_bounce_frame,
            return_bounce_frame=self.return_bounce_frame,
            satisfied_frame=self.satisfied_frame,
            fault_detected=self.fault_event is not None,
            fault_event=self.fault_event.to_dict() if self.fault_event else None,
            status_text=status,
        )

    @classmethod
    def analyze_telemetry(
        cls,
        telemetry_data: Dict[str, Any],
        serving_team: str = "A",
        assume_serve_start: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Run full Two-Bounce rule pass over a perception telemetry dictionary."""
        if assume_serve_start is None:
            assume_serve_start = infer_serve_start_from_telemetry(telemetry_data, serving_team)

        enforcer = cls(serving_team=serving_team, assume_serve_start=assume_serve_start)
        stream = telemetry_data.get("telemetry_stream", [])
        bounces_by_frame = {int(b["frame_idx"]): b for b in telemetry_data.get("bounces", [])}

        frame_states: List[Dict[str, Any]] = []
        for fr in stream:
            fi = int(fr["frame_idx"])
            ts = float(fr.get("timestamp_s", fi / 30.0))
            bounce = fr.get("bounce") or bounces_by_frame.get(fi)
            st = enforcer.update(
                frame_idx=fi,
                timestamp_s=ts,
                ball=fr.get("ball"),
                bounce=bounce,
                players=fr.get("players", []),
            )
            frame_states.append(st.to_dict())

        return {
            "assume_serve_start": assume_serve_start,
            "final_phase": enforcer.phase,
            "bounces_completed": enforcer.bounces_completed,
            "serve_bounce_frame": enforcer.serve_bounce_frame,
            "return_bounce_frame": enforcer.return_bounce_frame,
            "satisfied_frame": enforcer.satisfied_frame,
            "fault_event": enforcer.fault_event.to_dict() if enforcer.fault_event else None,
            "events": [e.to_dict() for e in enforcer.events],
            "frame_states": frame_states,
        }
