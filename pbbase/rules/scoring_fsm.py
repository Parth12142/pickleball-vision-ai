"""
scoring_fsm.py - Week 4 (W4-01, Day 14): USAP 0-0-2 Scoring Finite State Machine & Serving Sequence.

Implements official USA Pickleball (USAP) doubles serving sequence rules aligned with
Week 1-3 spatial court coordinates (X in [0, 20] ft, Y in [0, 44] ft):
  - Net plane: Y = 22.0 ft
  - Team A (Near court, Y > 22.0 ft, facing Far baseline Y = 0):
      * RIGHT (Even) service box: X in [10.0, 20.0], Y in [29.0, 44.0]
      * LEFT  (Odd)  service box: X in [0.0, 10.0],  Y in [29.0, 44.0]
  - Team B (Far court, Y < 22.0 ft, facing Near baseline Y = 44):
      * RIGHT (Even) service box: X in [0.0, 10.0],  Y in [0.0, 15.0]
      * LEFT  (Odd)  service box: X in [10.0, 20.0], Y in [0.0, 15.0]
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pbbase.court.homography import CourtGeometry

# Canonical USAP regulation service box bounds (x_min, x_max, y_min, y_max) in feet.
_HALF_W = CourtGeometry.WIDTH_FT / 2.0        # 10.0 ft centerline
_NET_Y = CourtGeometry.NET_Y_FT               # 22.0 ft net
_NVZ_FAR_Y = _NET_Y - CourtGeometry.NVZ_DEPTH_FT   # 15.0 ft
_NVZ_NEAR_Y = _NET_Y + CourtGeometry.NVZ_DEPTH_FT  # 29.0 ft

SERVICE_QUADRANT_BOUNDS_FT: Dict[Tuple[str, str], Tuple[float, float, float, float]] = {
    # Near team (Team A) facing North (Y -> 0): player's Right is +X, Left is -X
    ("A", "RIGHT"): (_HALF_W, CourtGeometry.WIDTH_FT, _NVZ_NEAR_Y, CourtGeometry.LENGTH_FT),
    ("A", "LEFT"):  (0.0, _HALF_W, _NVZ_NEAR_Y, CourtGeometry.LENGTH_FT),
    # Far team (Team B) facing South (Y -> 44): player's Right is -X, Left is +X
    ("B", "RIGHT"): (0.0, _HALF_W, 0.0, _NVZ_FAR_Y),
    ("B", "LEFT"):  (_HALF_W, CourtGeometry.WIDTH_FT, 0.0, _NVZ_FAR_Y),
}


@dataclass
class TeamCourtState:
    """Tracks score and current Right (Even) / Left (Odd) service-court occupants for one team."""
    team_id: str
    right_player: str
    left_player: str
    initial_right_player: str = ""
    score: int = 0

    def __post_init__(self) -> None:
        if not self.initial_right_player:
            self.initial_right_player = self.right_player

    def swap_courts(self) -> None:
        """Swap partners between Right (Even) and Left (Odd) service boxes after scoring a point."""
        self.right_player, self.left_player = self.left_player, self.right_player

    def player_in_box(self, box: str) -> str:
        if box == "RIGHT":
            return self.right_player
        if box == "LEFT":
            return self.left_player
        raise ValueError(f"Invalid service box '{box}'; expected 'RIGHT' or 'LEFT'.")

    def box_of_player(self, player_id: str) -> str:
        if player_id == self.right_player:
            return "RIGHT"
        if player_id == self.left_player:
            return "LEFT"
        raise ValueError(f"Player '{player_id}' is not assigned to Team '{self.team_id}'.")

    def verify_parity_invariant(self) -> bool:
        """
        Official USAP Doubles Invariant:
        - Even score -> initial starting-right player occupies RIGHT court.
        - Odd score  -> initial starting-right player occupies LEFT court.
        """
        if self.score % 2 == 0:
            return self.right_player == self.initial_right_player
        return self.left_player == self.initial_right_player

    def to_dict(self) -> Dict[str, Any]:
        return {
            "team_id": self.team_id,
            "score": self.score,
            "right_player": self.right_player,
            "left_player": self.left_player,
            "initial_right_player": self.initial_right_player,
            "parity_valid": self.verify_parity_invariant(),
        }


@dataclass
class ServeContext:
    """Snapshot of the active USAP serve state prior to a rally serve."""
    score_call: str
    serving_score: int
    receiving_score: int
    serving_team: str
    receiving_team: str
    server_number: int                  # 1 or 2 (2 at opening 0-0-2)
    active_server_id: str               # e.g., "P1"
    active_receiver_id: str             # Diagonal receiver ID, e.g., "P4"
    server_box: str                     # "RIGHT" or "LEFT"
    target_receiver_box: str            # Matches server_box diagonally on opponent court
    target_quadrant_ft: Tuple[float, float, float, float]
    is_opening_half_inning: bool
    game_over: bool = False
    winner_team: Optional[str] = None
    side_switch_due: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score_call": self.score_call,
            "serving_score": self.serving_score,
            "receiving_score": self.receiving_score,
            "serving_team": self.serving_team,
            "receiving_team": self.receiving_team,
            "server_number": self.server_number,
            "active_server_id": self.active_server_id,
            "active_receiver_id": self.active_receiver_id,
            "server_box": self.server_box,
            "target_receiver_box": self.target_receiver_box,
            "target_quadrant_ft": [round(v, 2) for v in self.target_quadrant_ft],
            "is_opening_half_inning": self.is_opening_half_inning,
            "game_over": self.game_over,
            "winner_team": self.winner_team,
            "side_switch_due": self.side_switch_due,
        }


@dataclass
class SequenceEvent:
    """Audit event emitted for each state transition inside the USAP Serving FSM."""
    event_index: int
    event_type: str                     # POINT_SCORED | SECOND_SERVER | SIDE_OUT | GAME_WON
    rally_winner_team: str
    prev_score_call: str
    next_score_call: str
    serving_team_before: str
    serving_team_after: str
    server_id_before: str
    server_id_after: str
    server_box_before: str
    server_box_after: str
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_index": self.event_index,
            "event_type": self.event_type,
            "rally_winner_team": self.rally_winner_team,
            "prev_score_call": self.prev_score_call,
            "next_score_call": self.next_score_call,
            "serving_team_before": self.serving_team_before,
            "serving_team_after": self.serving_team_after,
            "server_id_before": self.server_id_before,
            "server_id_after": self.server_id_after,
            "server_box_before": self.server_box_before,
            "server_box_after": self.server_box_after,
            "notes": self.notes,
        }


class ScoringFSM:
    """
    Week 4 (W4-01 & W4-02) USAP 0-0-2 Doubles Scoring Finite State Machine.

    Governs:
      1. Match initialization at official `0-0-2` (opening half-inning with single Server #2).
      2. Active server identity & box assignment (First Server vs Second Server rotation).
      3. Full multi-inning side-out transitions, server rotation, and rally undo checkpoints.
      4. Score progression through deuce/win-by-2 game completion (11 / 15 / 21 pts) & midpoint switch.
      5. Diagonal target service quadrant resolution and perception frame seeding (P1-P4 + ball).
    """

    def __init__(
        self,
        initial_serving_team: str = "A",
        team_a_players: Tuple[str, str] = ("P1", "P2"),
        team_b_players: Tuple[str, str] = ("P4", "P3"),
        target_score: int = 11,
        win_by: int = 2,
    ) -> None:
        if initial_serving_team not in ("A", "B"):
            raise ValueError(f"initial_serving_team must be 'A' or 'B', got '{initial_serving_team}'.")
        if target_score < 1 or win_by < 1:
            raise ValueError("target_score and win_by must be positive integers.")

        self.target_score: int = int(target_score)
        self.win_by: int = int(win_by)
        self.side_switch_score: int = (self.target_score // 2) + 1

        self.teams: Dict[str, TeamCourtState] = {
            "A": TeamCourtState(team_id="A", right_player=team_a_players[0], left_player=team_a_players[1]),
            "B": TeamCourtState(team_id="B", right_player=team_b_players[0], left_player=team_b_players[1]),
        }
        self.serving_team: str = initial_serving_team
        self.server_number: int = 2
        self.is_opening_half_inning: bool = True
        self.active_server_id: str = self.teams[self.serving_team].right_player

        # W4-02 progression counters & undo stack
        self.rally_count: int = 0
        self.side_out_count: int = 0
        self.half_inning_index: int = 1
        self.current_server_streak: int = 0
        self.game_over: bool = False
        self.winner_team: Optional[str] = None
        self.side_switch_triggered: bool = False
        self.history: List[SequenceEvent] = []
        self._undo_stack: List[Dict[str, Any]] = []

    @property
    def receiving_team(self) -> str:
        return "B" if self.serving_team == "A" else "A"

    @property
    def score_call(self) -> str:
        srv_score = self.teams[self.serving_team].score
        rcv_score = self.teams[self.receiving_team].score
        return f"{srv_score}-{rcv_score}-{self.server_number}"

    def _capture_snapshot(self) -> Dict[str, Any]:
        return {
            "A": (self.teams["A"].score, self.teams["A"].right_player, self.teams["A"].left_player),
            "B": (self.teams["B"].score, self.teams["B"].right_player, self.teams["B"].left_player),
            "serving_team": self.serving_team,
            "server_number": self.server_number,
            "is_opening_half_inning": self.is_opening_half_inning,
            "active_server_id": self.active_server_id,
            "rally_count": self.rally_count,
            "side_out_count": self.side_out_count,
            "half_inning_index": self.half_inning_index,
            "current_server_streak": self.current_server_streak,
            "game_over": self.game_over,
            "winner_team": self.winner_team,
            "side_switch_triggered": self.side_switch_triggered,
        }

    def undo_last_rally(self) -> Optional[SequenceEvent]:
        """Revert the most recent rally transition and restore exact prior state."""
        if not self._undo_stack or not self.history:
            return None
        snap = self._undo_stack.pop()
        self.teams["A"].score, self.teams["A"].right_player, self.teams["A"].left_player = snap["A"]
        self.teams["B"].score, self.teams["B"].right_player, self.teams["B"].left_player = snap["B"]
        self.serving_team = snap["serving_team"]
        self.server_number = snap["server_number"]
        self.is_opening_half_inning = snap["is_opening_half_inning"]
        self.active_server_id = snap["active_server_id"]
        self.rally_count = snap["rally_count"]
        self.side_out_count = snap["side_out_count"]
        self.half_inning_index = snap["half_inning_index"]
        self.current_server_streak = snap["current_server_streak"]
        self.game_over = snap["game_over"]
        self.winner_team = snap["winner_team"]
        self.side_switch_triggered = snap["side_switch_triggered"]
        return self.history.pop()

    def get_serve_context(self) -> ServeContext:
        """Return full live service context including active server, diagonal receiver, and target box."""
        srv_team = self.teams[self.serving_team]
        rcv_team = self.teams[self.receiving_team]
        server_box = srv_team.box_of_player(self.active_server_id)
        # Diagonal receiver stands in the identically named service box on the opposite half
        target_receiver_box = server_box
        active_receiver_id = rcv_team.player_in_box(target_receiver_box)
        target_quadrant = SERVICE_QUADRANT_BOUNDS_FT[(self.receiving_team, target_receiver_box)]

        return ServeContext(
            score_call=self.score_call,
            serving_score=srv_team.score,
            receiving_score=rcv_team.score,
            serving_team=self.serving_team,
            receiving_team=self.receiving_team,
            server_number=self.server_number,
            active_server_id=self.active_server_id,
            active_receiver_id=active_receiver_id,
            server_box=server_box,
            target_receiver_box=target_receiver_box,
            target_quadrant_ft=target_quadrant,
            is_opening_half_inning=self.is_opening_half_inning,
            game_over=self.game_over,
            winner_team=self.winner_team,
            side_switch_due=self.side_switch_triggered,
        )

    def advance_rally(self, rally_winner_team: str, notes: str = "") -> SequenceEvent:
        """
        Advance the USAP scoring & serving FSM based on which team won the rally ('A' or 'B').

        - Serving team wins: +1 point, serving team swaps Right <-> Left, same server continues
          from alternate box until reaching target_score with >= win_by margin (`GAME_WON`).
        - Receiving team wins when server_number == 1: advance to Server #2 (teammate in opposite
          court box), no score change, no position swap.
        - Receiving team wins when server_number == 2 (including opening 0-0-2): half-inning ends,
          side-out to opponent as Server #1 starting from opponent's RIGHT (Even) court.
        """
        if rally_winner_team not in ("A", "B"):
            raise ValueError(f"rally_winner_team must be 'A' or 'B', got '{rally_winner_team}'.")
        if self.game_over:
            raise RuntimeError(f"Match already won by Team {self.winner_team} at score {self.score_call}.")

        before = self.get_serve_context()
        self._undo_stack.append(self._capture_snapshot())
        self.rally_count += 1

        if rally_winner_team == self.serving_team:
            srv_team = self.teams[self.serving_team]
            rcv_team = self.teams[self.receiving_team]
            srv_team.score += 1
            self.current_server_streak += 1
            srv_team.swap_courts()

            if (not self.side_switch_triggered) and max(srv_team.score, rcv_team.score) >= self.side_switch_score:
                self.side_switch_triggered = True

            if srv_team.score >= self.target_score and (srv_team.score - rcv_team.score) >= self.win_by:
                self.game_over = True
                self.winner_team = self.serving_team
                event_type = "GAME_WON"
                detail = notes or f"Team {self.serving_team} wins match ({srv_team.score}-{rcv_team.score})!"
            else:
                event_type = "POINT_SCORED"
                detail = notes or f"Team {self.serving_team} scored; {self.active_server_id} swaps court box."
        elif self.server_number == 1:
            self.server_number = 2
            self.current_server_streak = 0
            srv_team = self.teams[self.serving_team]
            prev_box = before.server_box
            next_box = "LEFT" if prev_box == "RIGHT" else "RIGHT"
            self.active_server_id = srv_team.player_in_box(next_box)
            event_type = "SECOND_SERVER"
            detail = notes or f"Team {self.serving_team} advances to Server #2 ({self.active_server_id} in {next_box})."
        else:
            # Server #2 fault (including 0-0-2 opening fault) -> Side-out to opposite team as Server #1
            self.is_opening_half_inning = False
            self.serving_team = self.receiving_team
            self.server_number = 1
            self.side_out_count += 1
            self.half_inning_index += 1
            self.current_server_streak = 0
            new_srv_team = self.teams[self.serving_team]
            self.active_server_id = new_srv_team.right_player
            event_type = "SIDE_OUT"
            detail = notes or f"Side-out #{self.side_out_count} to Team {self.serving_team}; {self.active_server_id} serves #1 (RIGHT)."

        after = self.get_serve_context()
        event = SequenceEvent(
            event_index=len(self.history) + 1,
            event_type=event_type,
            rally_winner_team=rally_winner_team,
            prev_score_call=before.score_call,
            next_score_call=after.score_call,
            serving_team_before=before.serving_team,
            serving_team_after=after.serving_team,
            server_id_before=before.active_server_id,
            server_id_after=after.active_server_id,
            server_box_before=before.server_box,
            server_box_after=after.server_box,
            notes=detail,
        )
        self.history.append(event)
        return event

    def play_rallies(self, winners: List[str]) -> List[SequenceEvent]:
        """Replay a sequence of rally winners until completion or match win."""
        events: List[SequenceEvent] = []
        for winner in winners:
            if self.game_over:
                break
            events.append(self.advance_rally(winner))
        return events

    @classmethod
    def seed_from_telemetry_players(
        cls,
        players: List[Dict[str, Any]],
        infer_initial_server: bool = False,
        default_serving_team: str = "A",
        initial_ball_court_ft: Optional[Tuple[float, float]] = None,
        target_score: int = 11,
        win_by: int = 2,
    ) -> "ScoringFSM":
        """
        Build a ScoringFSM starting at `0-0-2` seeded directly from tracked player geometry
        in unified perception JSON (`slot`, `side`, `court_ft`).

        - Near side (`Y > 22 ft` or `side == "near"`): Right player has larger X_ft.
        - Far side  (`Y < 22 ft` or `side == "far"`):  Right player (facing South) has smaller X_ft.
        - When `initial_ball_court_ft` is supplied and `infer_initial_server=True`, the team side
          possessing the ball on frame 0 (`Y >= 22` -> 'A', `Y < 22` -> 'B') takes priority over
          heuristic baseline depth.
        """
        near_players: List[Tuple[str, float, float]] = []
        far_players: List[Tuple[str, float, float]] = []

        for p in players:
            pid = str(p.get("slot") or f"P{p.get('id', '?')}")
            court_ft = p.get("court_ft") or [10.0, 22.0]
            x_ft, y_ft = float(court_ft[0]), float(court_ft[1])
            side = p.get("side")
            is_near = (side == "near") if side in ("near", "far") else (y_ft >= _NET_Y)
            if is_near:
                near_players.append((pid, x_ft, y_ft))
            else:
                far_players.append((pid, x_ft, y_ft))

        if len(near_players) < 2 or len(far_players) < 2:
            raise ValueError(
                f"Expected 2 near and 2 far players for doubles seeding, got near={len(near_players)}, far={len(far_players)}."
            )

        # Near players face Y=0: larger X is player's RIGHT court
        near_sorted = sorted(near_players, key=lambda item: item[1], reverse=True)
        team_a_tuple = (near_sorted[0][0], near_sorted[1][0])

        # Far players face Y=44: smaller X is player's RIGHT court
        far_sorted = sorted(far_players, key=lambda item: item[1])
        team_b_tuple = (far_sorted[0][0], far_sorted[1][0])

        serving_team = default_serving_team
        if infer_initial_server:
            if initial_ball_court_ft is not None and len(initial_ball_court_ft) >= 2:
                serving_team = "A" if float(initial_ball_court_ft[1]) >= _NET_Y else "B"
            else:
                near_baseline_depth = max(y - CourtGeometry.LENGTH_FT for _, _, y in near_players)
                far_baseline_depth = max(0.0 - y for _, _, y in far_players)
                if far_baseline_depth > near_baseline_depth:
                    serving_team = "B"
                elif near_baseline_depth > far_baseline_depth:
                    serving_team = "A"

        return cls(
            initial_serving_team=serving_team,
            team_a_players=team_a_tuple,
            team_b_players=team_b_tuple,
            target_score=target_score,
            win_by=win_by,
        )

    def validate_player_alignment(self, players: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify whether observed player `(X_ft, Y_ft)` coordinates match the expected
        Left/Right court quadrant for the active server and diagonal receiver.
        """
        ctx = self.get_serve_context()
        by_id: Dict[str, Tuple[float, float]] = {}
        for p in players:
            pid = str(p.get("slot") or f"P{p.get('id', '?')}")
            c = p.get("court_ft")
            if c is not None and len(c) >= 2:
                by_id[pid] = (float(c[0]), float(c[1]))

        srv_xy = by_id.get(ctx.active_server_id)
        rcv_xy = by_id.get(ctx.active_receiver_id)

        def _expected_side_half(team: str, box: str) -> str:
            if team == "A":
                return "GE" if box == "RIGHT" else "LT"
            return "LT" if box == "RIGHT" else "GE"

        def _in_expected_half(team: str, box: str, xy: Optional[Tuple[float, float]]) -> Optional[bool]:
            if xy is None:
                return None
            op = _expected_side_half(team, box)
            return (xy[0] >= _HALF_W) if op == "GE" else (xy[0] < _HALF_W)

        srv_ok = _in_expected_half(ctx.serving_team, ctx.server_box, srv_xy)
        rcv_ok = _in_expected_half(ctx.receiving_team, ctx.target_receiver_box, rcv_xy)

        return {
            "score_call": ctx.score_call,
            "active_server_id": ctx.active_server_id,
            "server_box": ctx.server_box,
            "server_xy_ft": list(srv_xy) if srv_xy else None,
            "server_aligned": srv_ok,
            "active_receiver_id": ctx.active_receiver_id,
            "target_receiver_box": ctx.target_receiver_box,
            "receiver_xy_ft": list(rcv_xy) if rcv_xy else None,
            "receiver_aligned": rcv_ok,
            "both_aligned": bool(srv_ok is True and rcv_ok is True),
        }

    def verify_all_invariants(self) -> bool:
        """Confirm court-position parity and active-server membership invariants for both teams."""
        srv_team = self.teams[self.serving_team]
        valid_server = self.active_server_id in (srv_team.right_player, srv_team.left_player)
        return (
            valid_server
            and self.teams["A"].verify_parity_invariant()
            and self.teams["B"].verify_parity_invariant()
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": "Week 4 / W4-01 & W4-02 (USAP 0-0-2 Scoring & Side-Out Progression)",
            "serve_context": self.get_serve_context().to_dict(),
            "teams": {
                "A": self.teams["A"].to_dict(),
                "B": self.teams["B"].to_dict(),
            },
            "progression": {
                "target_score": self.target_score,
                "win_by": self.win_by,
                "side_switch_score": self.side_switch_score,
                "side_switch_triggered": self.side_switch_triggered,
                "rally_count": self.rally_count,
                "side_out_count": self.side_out_count,
                "half_inning_index": self.half_inning_index,
                "current_server_streak": self.current_server_streak,
                "game_over": self.game_over,
                "winner_team": self.winner_team,
            },
            "invariants_valid": self.verify_all_invariants(),
            "history": [ev.to_dict() for ev in self.history],
            "history_count": len(self.history),
        }
