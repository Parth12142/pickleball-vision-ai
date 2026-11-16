"""
minimap.py - 2D Orthographic Top-Down Bird's-Eye USAP Court Minimap.
Zero referee or scoring rules: strictly physical spatial rendering of court geometry,
player positions, ball trajectories, and bounce landing ripple rings.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

from pbbase.court.homography import CourtGeometry


# High-visibility broadcast color palette (BGR)
COLOR_BACKGROUND = (26, 30, 38)     # Apron / out-of-bounds border
COLOR_COURT_BASE = (65, 50, 40)     # In-bounds service boxes (Dark Navy/Slate)
COLOR_KITCHEN = (55, 75, 45)        # Non-Volley Zone (Kitchen Slate/Olive)
COLOR_LINES = (245, 245, 245)       # USAP boundary lines (Crisp White)
COLOR_NET = (190, 190, 190)         # Net line
COLOR_NET_POST = (120, 120, 120)    # Net posts
COLOR_BALL = (57, 255, 186)         # High-vis pickleball volt
COLOR_BOUNCE_IN = (80, 220, 80)     # In-bounds bounce ring (Green)
COLOR_BOUNCE_OUT = (60, 60, 230)    # Out-of-bounds bounce ring (Red)

PLAYER_COLORS = [
    (0, 145, 255),   # P1: Orange / Coral
    (255, 175, 0),   # P2: Electric Cyan
    (80, 220, 0),    # P3: Emerald Green
    (0, 215, 255),   # P4: Bright Gold
]


@dataclass
class MinimapBounceMarker:
    """Active bounce ripple on the minimap."""
    court_xy_ft: Tuple[float, float]
    frame_idx: int
    is_in_bounds: bool
    remaining_frames: int = 30
    max_frames: int = 30


class CourtMinimap:
    """
    Renders an orthographic 2D top-down USAP regulation court minimap (20 x 44 ft).
    Provides spatial mapping between court feet and minimap canvas pixel coordinates,
    drawing players, ball trajectory trails, bounce impact ripples, and HUD blending.
    """
    def __init__(
        self,
        court_scale: float = 10.0,
        padding: int = 20,
        trail_len: int = 12,
        bounce_duration: int = 30,
        ema_alpha: float = 0.60,
    ):
        """
        Args:
            court_scale: Canvas pixels per court foot (10.0 -> 200x440 px court).
            padding: Apron margin in pixels around the court boundary.
            trail_len: Maximum historical ball trail length rendered.
            bounce_duration: Lifetime of bounce ripple animation in frames.
            ema_alpha: Gentle 3-frame equivalent EMA smoothing weight (1.0 = unsmoothed).
        """
        self.court_scale = court_scale
        self.padding = padding
        self.trail_len = trail_len
        self.bounce_duration = bounce_duration
        self.ema_alpha = ema_alpha

        self.court_w_px = int(round(CourtGeometry.WIDTH_FT * self.court_scale))
        self.court_h_px = int(round(CourtGeometry.LENGTH_FT * self.court_scale))

        self.canvas_w = self.court_w_px + 2 * self.padding
        self.canvas_h = self.court_h_px + 2 * self.padding

        # Pre-render static court floor base
        self._static_base = self._draw_static_court()

        # Dynamic state
        self.ball_trail: List[Tuple[float, float]] = []
        self.active_bounces: List[MinimapBounceMarker] = []
        self._ema_ball_ft: Optional[Tuple[float, float]] = None
        self._missed_ball_frames: int = 0

    def reset(self):
        """Reset dynamic trails, EMA smoother, and bounce markers."""
        self.ball_trail.clear()
        self.active_bounces.clear()
        self._ema_ball_ft = None
        self._missed_ball_frames = 0

    def court_to_canvas(self, x_ft: float, y_ft: float) -> Tuple[int, int]:
        """Convert metric court coordinates (feet) to minimap canvas pixel coordinates."""
        cx = int(round(self.padding + x_ft * self.court_scale))
        cy = int(round(self.padding + y_ft * self.court_scale))
        return cx, cy

    def canvas_to_court(self, cx: int, cy: int) -> Tuple[float, float]:
        """Convert minimap canvas pixel coordinates to metric court coordinates (feet)."""
        x_ft = (cx - self.padding) / self.court_scale
        y_ft = (cy - self.padding) / self.court_scale
        return float(x_ft), float(y_ft)

    def is_in_court_bounds(self, x_ft: float, y_ft: float) -> bool:
        """Check if court coordinates fall within USAP boundaries (20x44 ft)."""
        return (0.0 <= x_ft <= CourtGeometry.WIDTH_FT) and (0.0 <= y_ft <= CourtGeometry.LENGTH_FT)

    def _draw_static_court(self) -> np.ndarray:
        """Render the static USAP regulation court lines, kitchen, and net."""
        canvas = np.full((self.canvas_h, self.canvas_w, 3), COLOR_BACKGROUND, dtype=np.uint8)

        # 1. Main Court Surface (0,0) to (20,44)
        c0 = self.court_to_canvas(0.0, 0.0)
        c1 = self.court_to_canvas(CourtGeometry.WIDTH_FT, CourtGeometry.LENGTH_FT)
        cv2.rectangle(canvas, c0, c1, COLOR_COURT_BASE, -1)

        # 2. Non-Volley Zone (Kitchen): Y=15 to Y=29
        k0 = self.court_to_canvas(0.0, 15.0)
        k1 = self.court_to_canvas(CourtGeometry.WIDTH_FT, 29.0)
        cv2.rectangle(canvas, k0, k1, COLOR_KITCHEN, -1)

        # 3. Outer Boundary Lines (Sidelines & Baselines)
        line_thick = max(1, int(round(self.court_scale * 0.18)))
        cv2.rectangle(canvas, c0, c1, COLOR_LINES, line_thick)

        # 4. Kitchen Lines (Y=15 and Y=29)
        k_top_l = self.court_to_canvas(0.0, 15.0)
        k_top_r = self.court_to_canvas(CourtGeometry.WIDTH_FT, 15.0)
        cv2.line(canvas, k_top_l, k_top_r, COLOR_LINES, line_thick)

        k_bot_l = self.court_to_canvas(0.0, 29.0)
        k_bot_r = self.court_to_canvas(CourtGeometry.WIDTH_FT, 29.0)
        cv2.line(canvas, k_bot_l, k_bot_r, COLOR_LINES, line_thick)

        # 5. Center Service Lines: X=10 from Y=0 to 15, and Y=29 to 44 (Kitchen has no centerline)
        c_top_start = self.court_to_canvas(10.0, 0.0)
        c_top_end = self.court_to_canvas(10.0, 15.0)
        cv2.line(canvas, c_top_start, c_top_end, COLOR_LINES, line_thick)

        c_bot_start = self.court_to_canvas(10.0, 29.0)
        c_bot_end = self.court_to_canvas(10.0, 44.0)
        cv2.line(canvas, c_bot_start, c_bot_end, COLOR_LINES, line_thick)

        # 6. Net Line (Y=22) and Net Posts
        net_l = self.court_to_canvas(-0.8, 22.0)
        net_r = self.court_to_canvas(CourtGeometry.WIDTH_FT + 0.8, 22.0)
        cv2.line(canvas, net_l, net_r, COLOR_NET, max(2, line_thick + 1))

        # Net Posts
        cv2.circle(canvas, net_l, max(2, int(line_thick * 1.5)), COLOR_NET_POST, -1)
        cv2.circle(canvas, net_r, max(2, int(line_thick * 1.5)), COLOR_NET_POST, -1)

        # Subtle court border border shadow
        cv2.rectangle(canvas, (0, 0), (self.canvas_w - 1, self.canvas_h - 1), (12, 14, 18), 1)

        return canvas

    def add_bounce(self, court_xy_ft: Tuple[float, float], frame_idx: int):
        """Register a new bounce impact ripple."""
        is_in = self.is_in_court_bounds(court_xy_ft[0], court_xy_ft[1])
        marker = MinimapBounceMarker(
            court_xy_ft=court_xy_ft,
            frame_idx=frame_idx,
            is_in_bounds=is_in,
            remaining_frames=self.bounce_duration,
            max_frames=self.bounce_duration,
        )
        self.active_bounces.append(marker)

    def render(
        self,
        players: Optional[List[Dict[str, Any]]] = None,
        ball: Optional[Dict[str, Any]] = None,
        frame_idx: int = 0,
    ) -> np.ndarray:
        """
        Render a full dynamic minimap frame containing:
        - Static USAP court
        - Dynamic bounce ripples
        - Ball trajectory trail & current ball dot
        - Player badges (P1-P4) with ID labels
        """
        canvas = self._static_base.copy()

        # 1. Update and render active bounce ripples
        surviving_bounces: List[MinimapBounceMarker] = []
        for b in self.active_bounces:
            if b.remaining_frames > 0:
                bx, by = self.court_to_canvas(b.court_xy_ft[0], b.court_xy_ft[1])
                # Expand radius as it ages
                progress = 1.0 - (b.remaining_frames / b.max_frames)
                radius = int(round(4 + progress * 16))
                color = COLOR_BOUNCE_IN if b.is_in_bounds else COLOR_BOUNCE_OUT

                # Draw concentric ripple rings
                alpha = max(0.1, 1.0 - progress)
                ring_color = tuple(int(c * alpha) for c in color)
                cv2.circle(canvas, (bx, by), radius, ring_color, 2)
                cv2.circle(canvas, (bx, by), max(2, radius // 2), ring_color, 1)
                # Center impact marker
                cv2.circle(canvas, (bx, by), 2, (255, 255, 255), -1)

                b.remaining_frames -= 1
                if b.remaining_frames > 0:
                    surviving_bounces.append(b)

        self.active_bounces = surviving_bounces

        # 2. Update and render Ball Trajectory Trail (Pure 1:1 H(u,v) + 3-frame EMA)
        raw_cft = ball.get("court_ft") if ball is not None else None

        if raw_cft is not None:
            self._missed_ball_frames = 0
            rx, ry = float(raw_cft[0]), float(raw_cft[1])
            if self._ema_ball_ft is not None:
                dist = float(np.hypot(rx - self._ema_ball_ft[0], ry - self._ema_ball_ft[1]))
                if dist > 14.0:
                    # Clean track reset on large discontinuity
                    self.ball_trail.clear()
                    sx, sy = rx, ry
                else:
                    sx = self.ema_alpha * rx + (1.0 - self.ema_alpha) * self._ema_ball_ft[0]
                    sy = self.ema_alpha * ry + (1.0 - self.ema_alpha) * self._ema_ball_ft[1]
            else:
                sx, sy = rx, ry

            self._ema_ball_ft = (sx, sy)
            self.ball_trail.append((sx, sy))
            if len(self.ball_trail) > self.trail_len:
                self.ball_trail.pop(0)
        else:
            self._missed_ball_frames += 1
            if self._missed_ball_frames > 4:
                self._ema_ball_ft = None
                self.ball_trail.clear()

        if len(self.ball_trail) >= 2:
            trail_pts = [self.court_to_canvas(x, y) for x, y in self.ball_trail]
            n_segs = len(trail_pts) - 1
            for i in range(1, len(trail_pts)):
                progress = i / float(max(1, n_segs))
                fade = 0.25 + 0.75 * progress
                col = tuple(int(c * fade) for c in COLOR_BALL)
                thickness = max(1, int(round(1 + progress * 2)))
                pt1 = (max(3, min(self.canvas_w - 4, trail_pts[i - 1][0])), max(3, min(self.canvas_h - 4, trail_pts[i - 1][1])))
                pt2 = (max(3, min(self.canvas_w - 4, trail_pts[i][0])), max(3, min(self.canvas_h - 4, trail_pts[i][1])))
                cv2.line(canvas, pt1, pt2, col, thickness, cv2.LINE_AA)

        # 3. Render Current Ball Position
        if self._ema_ball_ft is not None:
            bx, by = self.court_to_canvas(self._ema_ball_ft[0], self._ema_ball_ft[1])
            bx_c = max(5, min(self.canvas_w - 6, bx))
            by_c = max(5, min(self.canvas_h - 6, by))
            cv2.circle(canvas, (bx_c, by_c), 6, (15, 20, 25), -1, cv2.LINE_AA)
            cv2.circle(canvas, (bx_c, by_c), 5, COLOR_BALL, -1, cv2.LINE_AA)
            cv2.circle(canvas, (bx_c, by_c), 2, (255, 255, 255), -1, cv2.LINE_AA)

        # 4. Render Player Badges
        if players:
            for p in players:
                cft = p.get("court_ft")
                if not cft:
                    continue
                px, py = self.court_to_canvas(cft[0], cft[1])
                # Bounds safety check
                if not (-20 <= px < self.canvas_w + 20 and -20 <= py < self.canvas_h + 20):
                    continue

                pid = p.get("id")
                if pid is None:
                    slot_val = p.get("slot", 0)
                    if isinstance(slot_val, str) and slot_val.startswith("P"):
                        try:
                            pid = int(slot_val[1:])
                        except ValueError:
                            pid = 1
                    else:
                        try:
                            pid = int(slot_val) + 1
                        except (ValueError, TypeError):
                            pid = 1
                else:
                    try:
                        pid = int(pid)
                    except (ValueError, TypeError):
                        pid = 1

                slot = max(0, pid - 1)
                p_color = PLAYER_COLORS[slot % len(PLAYER_COLORS)]
                p_name = f"P{pid}"

                # Player dot with dark halo
                cv2.circle(canvas, (px, py), 7, (15, 15, 15), -1, cv2.LINE_AA)
                cv2.circle(canvas, (px, py), 6, p_color, -1, cv2.LINE_AA)

                # Player ID text label
                cv2.putText(
                    canvas,
                    p_name,
                    (px - 6, py - 9),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        return canvas

    def compose_onto_frame(
        self,
        frame: np.ndarray,
        minimap_canvas: np.ndarray,
        position: str = "bottom-right",
        margin: int = 18,
        alpha: float = 0.88,
    ) -> np.ndarray:
        """
        Composites the minimap onto the video frame at the designated corner.
        Supports 'bottom-right', 'bottom-left', 'top-right', 'top-left'.
        """
        fh, fw = frame.shape[:2]
        mh, mw = minimap_canvas.shape[:2]

        if position == "bottom-right":
            x1 = fw - mw - margin
            y1 = fh - mh - margin
        elif position == "bottom-left":
            x1 = margin
            y1 = fh - mh - margin
        elif position == "top-right":
            x1 = fw - mw - margin
            y1 = margin
        elif position == "top-left":
            x1 = margin
            y1 = margin
        else:
            x1 = fw - mw - margin
            y1 = fh - mh - margin

        x2, y2 = x1 + mw, y1 + mh

        # Clamp coordinates to frame boundary
        if x1 < 0 or y1 < 0 or x2 > fw or y2 > fh:
            return frame

        roi = frame[y1:y2, x1:x2]
        blended = cv2.addWeighted(minimap_canvas, alpha, roi, 1.0 - alpha, 0.0)

        # Outer sleek border
        cv2.rectangle(blended, (0, 0), (mw - 1, mh - 1), (80, 90, 105), 1)

        frame[y1:y2, x1:x2] = blended
        return frame
