"""
broadcast_fx.py - Broadcast-Style Pickleball Animation & USAP Scoreboard Overlay Engine.

Combines clean TV-grade court vector graphics with the Phase 3 USAP Scoreboard:
  - Yellow court wireframe lines + white corner vertex anchors
  - Rotating perspective sunburst rings under players' feet (zero bounding boxes)
  - Smooth Savitzky-Golay + Catmull-Rom neon magenta ball ribbon locked cleanly to the orb
  - Savitzky-Golay ground BOUNCE effects: green sub-court zone flash, expanding concentric
    red-orange shockwave rings, and crisp BOUNCE badge
  - Translucent bottom-left wireframe court minimap synced 1:1
  - Sleek bottom-right USAP Autonomous Scoreboard (`0-0-2` call, team scores, active
    server/receiver boxes, side-outs, 2-Bounce progress pips, and Kitchen NVZ status)
"""
from typing import Any, Dict, List, Optional, Tuple
import math

import cv2
import numpy as np

from pbbase.court.homography import CourtCalibrator, CourtGeometry


# Broadcast palette (BGR)
COURT_LINE_YELLOW = (65, 232, 248)
COURT_VERTEX_WHITE = (248, 250, 252)
BALL_MAGENTA = (232, 32, 242)
BALL_MAGENTA_GLOW = (205, 18, 195)
BOUNCE_GREEN_FILL = (50, 215, 65)
BOUNCE_RED_ORANGE = (26, 72, 218)

# Slot colors (BGR): P1 (Near Right), P2 (Near Left), P3 (Far Right), P4 (Far Left)
BROADCAST_PLAYER_COLORS: Dict[str, Tuple[int, int, int]] = {
    "P1": (245, 95, 175),   # Violet / Purple
    "P2": (245, 215, 35),   # Bright Cyan
    "P3": (95, 70, 245),    # Coral Red-Pink
    "P4": (25, 175, 255),   # Amber Orange
}


def get_court_6part_name(x_ft: float, y_ft: float) -> Optional[str]:
    """
    Map a court landing coordinate `(x_ft, y_ft)` to one of the 6 canonical USAP court parts:
      1. FAR_RIGHT_SERVICE  (X in [0, 10],  Y in [0, 15])
      2. FAR_LEFT_SERVICE   (X in [10, 20], Y in [0, 15])
      3. FAR_KITCHEN_NVZ    (X in [0, 20],  Y in [15, 22])  -- full 20ft width
      4. NEAR_KITCHEN_NVZ   (X in [0, 20],  Y in [22, 29])  -- full 20ft width
      5. NEAR_LEFT_SERVICE  (X in [0, 10],  Y in [29, 44])
      6. NEAR_RIGHT_SERVICE (X in [10, 20], Y in [29, 44])
    Returns `None` if out of bounds.
    """
    if not (-0.6 <= x_ft <= CourtGeometry.WIDTH_FT + 0.6 and -0.6 <= y_ft <= CourtGeometry.LENGTH_FT + 0.6):
        return None
    if y_ft < 15.0:
        return "FAR_RIGHT_SERVICE" if x_ft < 10.0 else "FAR_LEFT_SERVICE"
    if y_ft < 22.0:
        return "FAR_KITCHEN_NVZ"
    if y_ft < 29.0:
        return "NEAR_KITCHEN_NVZ"
    return "NEAR_LEFT_SERVICE" if x_ft < 10.0 else "NEAR_RIGHT_SERVICE"


def get_court_subzone_rect(x_ft: float, y_ft: float) -> Optional[Tuple[float, float, float, float]]:
    """
    Map a court landing coordinate `(x_ft, y_ft)` to its 6-part USAP court rectangle `(x0, x1, y0, y1)`.
    Note: The Far Kitchen (`Y in [15, 22]`) and Near Kitchen (`Y in [22, 29]`) span the full 20ft width
    (`X in [0, 20]`) with no centerline, dividing the pickleball court into 6 exact parts.
    """
    part = get_court_6part_name(x_ft, y_ft)
    if part is None:
        return None
    if part == "FAR_RIGHT_SERVICE":
        return (0.0, 10.0, 0.0, 15.0)
    if part == "FAR_LEFT_SERVICE":
        return (10.0, 20.0, 0.0, 15.0)
    if part == "FAR_KITCHEN_NVZ":
        return (0.0, 20.0, 15.0, 22.0)
    if part == "NEAR_KITCHEN_NVZ":
        return (0.0, 20.0, 22.0, 29.0)
    if part == "NEAR_LEFT_SERVICE":
        return (0.0, 10.0, 29.0, 44.0)
    return (10.0, 20.0, 29.0, 44.0)


def _catmull_rom_spline(points: List[Tuple[float, float]], samples_per_seg: int = 5) -> List[Tuple[float, float]]:
    """Evaluate smooth Catmull-Rom curve passing strictly through all control points."""
    if len(points) < 3:
        return points
    out: List[Tuple[float, float]] = []
    pts = [points[0]] + points + [points[-1]]
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for s in range(samples_per_seg):
            t = s / float(samples_per_seg)
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * (
                (2.0 * p1[0])
                + (-p0[0] + p2[0]) * t
                + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                (2.0 * p1[1])
                + (-p0[1] + p2[1]) * t
                + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * t3
            )
            out.append((x, y))
    out.append(points[-1])
    return out


class BroadcastAnimator:
    """
    Minimalist Broadcast Animation & Live USAP Scoreboard Renderer.
    Renders smoothed Savitzky-Golay ball paths, ground bounce zone animations,
    rotating player foot rings, bottom-left wireframe minimap, and bottom-right scoreboard.
    """

    def __init__(self, calibrator: CourtCalibrator, trail_len: int = 20, total_frames: int = 300, fps: float = 30.0, clip_name: str = "") -> None:
        self.calibrator = calibrator
        self.trail_len = trail_len
        self.total_frames = total_frames
        self.fps = fps
        self.clip_name = clip_name
        self.cumulative_bounces = 0
        self.ball_trail_px: List[Optional[Tuple[float, float]]] = []
        self.smoothed_feet: Dict[str, Tuple[float, float]] = {}
        self.prev_feet_ft: Dict[str, Tuple[float, float]] = {}
        self.smoothed_player_mph: Dict[str, float] = {}
        self.active_bounces: List[Dict[str, Any]] = []
        self._minimap_trail: List[Optional[Tuple[float, float]]] = []

    def reset(self) -> None:
        self.cumulative_bounces = 0
        self.ball_trail_px.clear()
        self.smoothed_feet.clear()
        self.prev_feet_ft.clear()
        self.smoothed_player_mph.clear()
        self.active_bounces.clear()
        self.active_bounces.clear()
        self._minimap_trail.clear()

    def _draw_court_wireframe(self, frame: np.ndarray, line_w: int = 2) -> np.ndarray:
        if not self.calibrator.result.calibrated:
            return frame

        over = frame.copy()
        segments_ft = [
            ((0.0, 0.0), (20.0, 0.0)),
            ((0.0, 44.0), (20.0, 44.0)),
            ((0.0, 0.0), (0.0, 44.0)),
            ((20.0, 0.0), (20.0, 44.0)),
            ((0.0, 15.0), (20.0, 15.0)),
            ((0.0, 22.0), (20.0, 22.0)),
            ((0.0, 29.0), (20.0, 29.0)),
            ((10.0, 0.0), (10.0, 15.0)),
            ((10.0, 29.0), (10.0, 44.0)),
        ]
        for p0_ft, p1_ft in segments_ft:
            u0, v0 = self.calibrator.court_to_pixel(*p0_ft)
            u1, v1 = self.calibrator.court_to_pixel(*p1_ft)
            cv2.line(over, (int(round(u0)), int(round(v0))), (int(round(u1)), int(round(v1))), COURT_LINE_YELLOW, line_w, cv2.LINE_AA)

        cv2.addWeighted(over, 0.72, frame, 0.28, 0.0, frame)

        vertices_ft = [
            (0.0, 0.0), (10.0, 0.0), (20.0, 0.0),
            (0.0, 15.0), (10.0, 15.0), (20.0, 15.0),
            (0.0, 22.0), (10.0, 22.0), (20.0, 22.0),
            (0.0, 29.0), (10.0, 29.0), (20.0, 29.0),
            (0.0, 44.0), (10.0, 44.0), (20.0, 44.0),
        ]
        for vx_ft, vy_ft in vertices_ft:
            u, v = self.calibrator.court_to_pixel(vx_ft, vy_ft)
            pt = (int(round(u)), int(round(v)))
            cv2.circle(frame, pt, 4, (20, 25, 30), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 3, COURT_VERTEX_WHITE, -1, cv2.LINE_AA)
        return frame

    def _draw_bounce_zone_fill(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        Fill the active 1-of-6 USAP court part with semi-transparent green beneath the yellow
        court lines and player halos so court lines and players stay crisp on top.
        """
        for ev in self.active_bounces:
            age = frame_idx - ev["frame_idx"]
            max_age = ev["duration"]
            if age < 0 or age >= max_age:
                continue
            t = age / float(max_age)
            fade = max(0.0, 1.0 - t) ** 1.15
            rect_ft = ev.get("subzone_ft")
            if rect_ft is not None and self.calibrator.result.calibrated:
                x0, x1, y0, y1 = rect_ft
                pts = []
                for cx, cy in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
                    u, v = self.calibrator.court_to_pixel(cx, cy)
                    pts.append([int(round(u)), int(round(v))])
                poly = np.array(pts, dtype=np.int32)
                over = frame.copy()
                cv2.fillPoly(over, [poly], BOUNCE_GREEN_FILL)
                cv2.addWeighted(over, 0.46 * fade, frame, 1.0 - 0.46 * fade, 0.0, frame)
        return frame

    def _draw_bounce_impact_fx(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Draw expanding concentric red-orange impact rings and crisp BOUNCE badge at exact bounce frame."""
        next_active = []
        for ev in self.active_bounces:
            age = frame_idx - ev["frame_idx"]
            max_age = ev["duration"]
            if age < 0 or age >= max_age:
                continue
            next_active.append(ev)

            t = age / float(max_age)
            fade = max(0.0, 1.0 - t) ** 1.25
            rect_ft = ev.get("subzone_ft")

            px = ev.get("pixel")
            if px is not None:
                cx, cy = int(round(px[0])), int(round(px[1]))
                sc = ev.get("persp_scale", 1.0)
                r1 = max(6, int(round((7 + 22 * t) * sc)))
                r2 = max(4, int(round((4 + 13 * t) * sc)))
                over = frame.copy()
                cv2.ellipse(over, (cx, cy), (r1, max(3, int(r1 * 0.55))), 0, 0, 360, BOUNCE_RED_ORANGE, 2, cv2.LINE_AA)
                cv2.ellipse(over, (cx, cy), (r2, max(2, int(r2 * 0.55))), 0, 0, 360, (40, 110, 255), 2, cv2.LINE_AA)
                cv2.circle(over, (cx, cy), max(3, int(round(4 * sc))), BOUNCE_RED_ORANGE, -1, cv2.LINE_AA)
                cv2.addWeighted(over, 0.85 * fade, frame, 1.0 - 0.85 * fade, 0.0, frame)

                if age < 14:
                    b_fade = max(0.0, 1.0 - age / 14.0)
                    badge_over = frame.copy()
                    c_ft_ev = ev.get("court_ft")
                    if c_ft_ev:
                        txt = f"BOUNCE OUT ({c_ft_ev[0]:.1f},{c_ft_ev[1]:.1f})ft" if rect_ft is None else f"BOUNCE ({c_ft_ev[0]:.1f},{c_ft_ev[1]:.1f})ft"
                    else:
                        txt = "BOUNCE (OUT)" if rect_ft is None else "BOUNCE"
                    fsc = 0.44 * max(1.0, sc)
                    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fsc, 2)
                    bx1 = max(8, min(frame.shape[1] - tw - 24, cx - tw // 2 - 8))
                    by1 = max(8, cy - int(round(44 * sc)) - th - 10)
                    bx2 = bx1 + tw + 16
                    by2 = by1 + th + 12
                    cv2.rectangle(badge_over, (bx1, by1), (bx2, by2), BOUNCE_RED_ORANGE, -1)
                    cv2.rectangle(badge_over, (bx1, by1), (bx2, by2), (245, 245, 245), 1, cv2.LINE_AA)
                    cv2.putText(badge_over, txt, (bx1 + 8, by2 - 6), cv2.FONT_HERSHEY_SIMPLEX, fsc, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.addWeighted(badge_over, 0.92 * b_fade, frame, 1.0 - 0.92 * b_fade, 0.0, frame)

        self.active_bounces = next_active
        return frame

    def _draw_player_halos(self, frame: np.ndarray, players: List[Dict[str, Any]], frame_idx: int) -> np.ndarray:
        fh, fw = frame.shape[:2]
        is_hi_res = max(fw, fh) >= 1800
        rot_deg = (frame_idx * 2.4) % 360.0
        over = frame.copy()

        for p in players:
            foot = p.get("foot_px")
            if not foot or len(foot) < 2:
                continue
            slot = str(p.get("slot") or f"P{p.get('id', '1')}")
            fx, fy = float(foot[0]), float(foot[1])
            if slot in self.smoothed_feet:
                px0, py0 = self.smoothed_feet[slot]
                if math.hypot(fx - px0, fy - py0) < 90.0:
                    fx = 0.58 * fx + 0.42 * px0
                    fy = 0.58 * fy + 0.42 * py0
            self.smoothed_feet[slot] = (fx, fy)

            col = BROADCAST_PLAYER_COLORS.get(slot, (245, 215, 35))
            box = p.get("box", [fx - 30, fy - 120, fx + 30, fy])
            bw = max(24.0, float(box[2]) - float(box[0]))
            rx = int(np.clip(bw * 0.44, 18, 72))
            ry = max(7, int(round(rx * 0.36)))
            cx, cy = int(round(fx)), int(round(fy))
            cv2.ellipse(over, (cx, cy), (rx, ry), 0, 0, 360, col, -1, cv2.LINE_AA)

        cv2.addWeighted(over, 0.16, frame, 0.84, 0.0, frame)

        for p in players:
            slot = str(p.get("slot") or f"P{p.get('id', '1')}")
            if slot not in self.smoothed_feet:
                continue
            fx, fy = self.smoothed_feet[slot]
            col = BROADCAST_PLAYER_COLORS.get(slot, (245, 215, 35))
            box = p.get("box", [fx - 30, fy - 120, fx + 30, fy])
            bw = max(24.0, float(box[2]) - float(box[0]))
            rx = int(np.clip(bw * 0.44, 18, 72))
            ry = max(7, int(round(rx * 0.36)))
            cx, cy = int(round(fx)), int(round(fy))

            cv2.ellipse(frame, (cx, cy), (rx, ry), 0, 0, 360, col, 3, cv2.LINE_AA)
            cv2.ellipse(frame, (cx, cy), (max(4, rx - 3), max(3, ry - 2)), 0, 0, 360, (245, 248, 252), 1, cv2.LINE_AA)

            n_rays = 10
            for k in range(n_rays):
                ang = math.radians(rot_deg + k * (360.0 / n_rays))
                ca, sa = math.cos(ang), math.sin(ang)
                p_in = (int(round(cx + (rx + 1) * ca)), int(round(cy + (ry + 1) * sa)))
                p_out = (int(round(cx + (rx + max(5, rx * 0.22)) * ca)), int(round(cy + (ry + max(3, ry * 0.28)) * sa)))
                cv2.line(frame, p_in, p_out, col, 2, cv2.LINE_AA)

            # Floating Player Telemetry Pill: P# (X.X, Y.Y)ft + speed mph (matching website preview output)
            cft = p.get("court_ft")
            p_spd = float(p.get("speed_mph") or 0.0)
            if cft and len(cft) >= 2:
                cx_ft, cy_ft = float(cft[0]), float(cft[1])
                if p_spd <= 0.1 and slot in self.prev_feet_ft:
                    px_ft, py_ft = self.prev_feet_ft[slot]
                    inst_mph = math.hypot(cx_ft - px_ft, cy_ft - py_ft) * self.fps * 0.6818
                    if inst_mph < 22.0:
                        p_spd = 0.35 * inst_mph + 0.65 * self.smoothed_player_mph.get(slot, inst_mph)
                        self.smoothed_player_mph[slot] = p_spd
                self.prev_feet_ft[slot] = (cx_ft, cy_ft)

            lbl = slot
            if cft and len(cft) >= 2:
                lbl += f" ({float(cft[0]):.1f}, {float(cft[1]):.1f})ft"
            if p_spd >= 0.8:
                lbl += f" {p_spd:.1f}mph"

            p_font = 0.65 if is_hi_res else 0.37
            p_th = 2 if is_hi_res else 1
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, p_font, p_th)
            pill_w = tw + (18 if is_hi_res else 12)
            pill_h = th + (14 if is_hi_res else 9)
            y1_box = int(round(float(box[1])))
            bx1 = max(6, min(fw - pill_w - 6, cx - pill_w // 2))
            by1 = max(56, y1_box - pill_h - 4)
            cv2.rectangle(frame, (bx1, by1), (bx1 + pill_w, by1 + pill_h), (15, 23, 42), -1)
            cv2.rectangle(frame, (bx1, by1), (bx1 + pill_w, by1 + pill_h), col, 2 if is_hi_res else 1, cv2.LINE_AA)
            cv2.putText(
                frame,
                lbl,
                (bx1 + (9 if is_hi_res else 6), by1 + pill_h - (6 if is_hi_res else 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                p_font,
                (242, 245, 250),
                p_th,
                cv2.LINE_AA,
            )

        return frame

    def _draw_ball_ribbon(self, frame: np.ndarray, ball: Optional[Dict[str, Any]], ui_scale: float) -> np.ndarray:
        """
        Draw spline-interpolated neon magenta trail using Savitzky-Golay smoothed trail coordinates
        while pinning the current tip and ball orb strictly to the un-lagged observed `ball['pixel']`,
        plus live Speed (MPH) & Height (ft) callout beside the ball.
        """
        fh, fw = frame.shape[:2]
        is_hi_res = max(fw, fh) >= 1800
        cur_pt: Optional[Tuple[float, float]] = None
        trail_pt: Optional[Tuple[float, float]] = None
        if ball is not None and ball.get("pixel"):
            cur_pt = (float(ball["pixel"][0]), float(ball["pixel"][1]))
            tp = ball.get("trail_pixel") or ball["pixel"]
            trail_pt = (float(tp[0]), float(tp[1]))
        self.ball_trail_px.append(trail_pt)

        if len(self.ball_trail_px) > self.trail_len:
            self.ball_trail_px = self.ball_trail_px[-self.trail_len :]

        recent: List[Tuple[float, float]] = []
        for pt in reversed(self.ball_trail_px):
            if pt is None:
                if len(recent) > 0:
                    break
                continue
            recent.append(pt)
        recent.reverse()

        if cur_pt is not None and len(recent) >= 1:
            recent[-1] = cur_pt

        if len(recent) >= 2:
            smooth_pts = _catmull_rom_spline(recent, samples_per_seg=5)
            n_pts = len(smooth_pts)
            glow_over = frame.copy()
            for k in range(n_pts - 1):
                prog = (k + 1) / float(n_pts)
                p0 = (int(round(smooth_pts[k][0])), int(round(smooth_pts[k][1])))
                p1 = (int(round(smooth_pts[k + 1][0])), int(round(smooth_pts[k + 1][1])))
                th_glow = max(2, int(round((2.0 + 7.0 * prog) * ui_scale)))
                cv2.line(glow_over, p0, p1, BALL_MAGENTA_GLOW, th_glow, cv2.LINE_AA)
            cv2.addWeighted(glow_over, 0.34, frame, 0.66, 0.0, frame)

            core_over = frame.copy()
            for k in range(n_pts - 1):
                prog = (k + 1) / float(n_pts)
                p0 = (int(round(smooth_pts[k][0])), int(round(smooth_pts[k][1])))
                p1 = (int(round(smooth_pts[k + 1][0])), int(round(smooth_pts[k + 1][1])))
                th_core = max(1, int(round((1.2 + 4.2 * prog) * ui_scale)))
                cv2.line(core_over, p0, p1, BALL_MAGENTA, th_core, cv2.LINE_AA)
            cv2.addWeighted(core_over, 0.92, frame, 0.08, 0.0, frame)

        if cur_pt is not None and ball is not None:
            bx, by = int(round(cur_pt[0])), int(round(cur_pt[1]))
            r = max(5, int(round(6 * ui_scale)))
            cv2.circle(frame, (bx, by), r + 1, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (bx, by), r, BALL_MAGENTA, -1, cv2.LINE_AA)

            # Live Ball Speed (MPH) & 3D Height (ft) callout next to ball
            spd_mph = float(ball.get("speed_mph") or 0.0)
            if spd_mph <= 0.5 and len(recent) >= 2:
                dx_p = recent[-1][0] - recent[-2][0]
                dy_p = recent[-1][1] - recent[-2][1]
                spd_mph = min(58.0, math.hypot(dx_p, dy_p) * self.fps * 0.042)
            h_ft = float(ball.get("height_ft") or 0.0)
            ball_lbl = f"{spd_mph:.1f} MPH | H:{h_ft:.1f}ft"
            b_font = 0.64 if is_hi_res else 0.38
            b_th = 2 if is_hi_res else 1
            tx = min(fw - (210 if is_hi_res else 125), bx + r + 8)
            ty = max(24, by - 6)
            cv2.putText(frame, ball_lbl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, b_font, (15, 23, 42), b_th + 2, cv2.LINE_AA)
            cv2.putText(frame, ball_lbl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, b_font, (0, 255, 220), b_th, cv2.LINE_AA)

        return frame

    def _draw_wireframe_minimap(
        self,
        frame: np.ndarray,
        players: List[Dict[str, Any]],
        ball: Optional[Dict[str, Any]],
        frame_idx: int,
    ) -> np.ndarray:
        """Render clean bottom-left translucent wireframe minimap matching reference style."""
        fh, fw = frame.shape[:2]
        scale = 8.4 if fh > fw else max(4.2, min(6.2, fw / 245.0))
        cw = int(round(CourtGeometry.WIDTH_FT * scale))
        ch = int(round(CourtGeometry.LENGTH_FT * scale))
        pad_x, pad_y = (18, 18) if fh > fw else (14, 14)
        mw, mh = cw + 2 * pad_x, ch + 2 * pad_y

        mx1 = 14
        my1 = fh - 14 - mh
        if mx1 + mw > fw or my1 < 0:
            return frame

        def c2m(x_ft: float, y_ft: float) -> Tuple[int, int]:
            px = int(round(mx1 + pad_x + x_ft * scale))
            py = int(round(my1 + pad_y + y_ft * scale))
            return px, py

        roi = frame[my1 : my1 + mh, mx1 : mx1 + mw]
        dark = np.zeros_like(roi)
        dark[:] = (16, 22, 26)
        cv2.addWeighted(dark, 0.34, roi, 0.66, 0.0, roi)
        frame[my1 : my1 + mh, mx1 : mx1 + mw] = roi

        for ev in self.active_bounces:
            age = frame_idx - ev["frame_idx"]
            if 0 <= age < ev["duration"] and ev.get("subzone_ft"):
                fade = max(0.0, 1.0 - age / float(ev["duration"]))
                x0, x1, y0, y1 = ev["subzone_ft"]
                p0, p1 = c2m(x0, y0), c2m(x1, y1)
                sub_over = frame.copy()
                cv2.rectangle(sub_over, p0, p1, BOUNCE_GREEN_FILL, -1)
                cv2.addWeighted(sub_over, 0.45 * fade, frame, 1.0 - 0.45 * fade, 0.0, frame)

        l_col = (228, 232, 238)
        for (a0, a1) in [
            ((0.0, 0.0), (20.0, 0.0)),
            ((0.0, 44.0), (20.0, 44.0)),
            ((0.0, 0.0), (0.0, 44.0)),
            ((20.0, 0.0), (20.0, 44.0)),
            ((0.0, 15.0), (20.0, 15.0)),
            ((0.0, 22.0), (20.0, 22.0)),
            ((0.0, 29.0), (20.0, 29.0)),
            ((10.0, 0.0), (10.0, 15.0)),
            ((10.0, 29.0), (10.0, 44.0)),
        ]:
            cv2.line(frame, c2m(*a0), c2m(*a1), l_col, 1, cv2.LINE_AA)

        for cx, cy in [(0.0, 0.0), (20.0, 0.0), (0.0, 44.0), (20.0, 44.0)]:
            cv2.circle(frame, c2m(cx, cy), 2, (230, 150, 40), -1, cv2.LINE_AA)

        m_ball_ft = None
        if ball is not None:
            m_ball_ft = ball.get("minimap_court_ft") or ball.get("court_ft")
        if m_ball_ft is not None:
            bx_ft = float(np.clip(m_ball_ft[0], -4.0, 24.0))
            by_ft = float(np.clip(m_ball_ft[1], -4.0, 48.0))
            self._minimap_trail.append((bx_ft, by_ft))
        else:
            self._minimap_trail.append(None)

        if len(self._minimap_trail) > self.trail_len:
            self._minimap_trail = self._minimap_trail[-self.trail_len :]

        valid_m: List[Tuple[float, float]] = []
        for pt in reversed(self._minimap_trail):
            if pt is None:
                if len(valid_m) > 0:
                    break
                continue
            valid_m.append(pt)
        valid_m.reverse()

        if len(valid_m) >= 2:
            t_over = frame.copy()
            for k in range(len(valid_m) - 1):
                cv2.line(t_over, c2m(*valid_m[k]), c2m(*valid_m[k + 1]), BALL_MAGENTA, 2, cv2.LINE_AA)
            cv2.addWeighted(t_over, 0.78, frame, 0.22, 0.0, frame)

        dot_r = 12 if fh > fw else 8
        ball_r = 9 if fh > fw else 7
        lbl_sc = 0.44 if fh > fw else 0.34
        for p in players:
            cf = p.get("court_ft")
            if not cf or len(cf) < 2:
                continue
            slot = str(p.get("slot") or f"P{p.get('id', '1')}")
            col = BROADCAST_PLAYER_COLORS.get(slot, (245, 215, 35))
            px, py = c2m(float(np.clip(cf[0], -3.0, 23.0)), float(np.clip(cf[1], -3.0, 47.0)))
            cv2.circle(frame, (px, py), dot_r + 2, (15, 20, 28), -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), dot_r, col, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), dot_r, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, slot, (px - dot_r, py - dot_r - 3), cv2.FONT_HERSHEY_SIMPLEX, lbl_sc, (15, 20, 28), 2, cv2.LINE_AA)
            cv2.putText(frame, slot, (px - dot_r, py - dot_r - 3), cv2.FONT_HERSHEY_SIMPLEX, lbl_sc, (255, 255, 255), 1, cv2.LINE_AA)

        if m_ball_ft is not None:
            bpx, bpy = c2m(float(np.clip(m_ball_ft[0], -4.0, 24.0)), float(np.clip(m_ball_ft[1], -4.0, 48.0)))
            cv2.circle(frame, (bpx, bpy), ball_r + 1, (15, 20, 28), -1, cv2.LINE_AA)
            cv2.circle(frame, (bpx, bpy), ball_r, BALL_MAGENTA, -1, cv2.LINE_AA)
            cv2.circle(frame, (bpx, bpy), max(2, ball_r // 3), (255, 255, 255), -1, cv2.LINE_AA)

        return frame

    def _draw_telemetry_header_hud(
        self,
        frame: np.ndarray,
        frame_idx: int,
        players: List[Dict[str, Any]],
        ball: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        """Render the classic website Telemetry Header HUD with Frame, Players, Ball, Bounces, Speed & Height."""
        fh, fw = frame.shape[:2]
        is_hi_res = max(fw, fh) >= 1800
        sc = 1.65 if is_hi_res else 1.0

        box_w = int(round(415 * sc))
        box_h = int(round(70 * sc))
        # Place top-right on clips that have a top-left broadcast scorebug (match2, match_mlp, screencast)
        if self.clip_name in ("match2", "match_mlp", "screencast"):
            x1 = max(16, fw - box_w - int(round(16 * sc)))
        else:
            x1 = int(round(16 * sc))
        y1 = int(round(14 * sc))
        x2, y2 = x1 + box_w, y1 + box_h

        over = frame.copy()
        cv2.rectangle(over, (x1, y1), (x2, y2), (15, 23, 42), -1)
        cv2.addWeighted(over, 0.86, frame, 0.14, 0.0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 220), 2 if is_hi_res else 1, cv2.LINE_AA)

        ball_status = "3D-EST" if (ball and ball.get("interpolated")) else ("LOCKED" if (ball and ball.get("pixel")) else "SEARCH")
        spd_mph = float((ball or {}).get("speed_mph") or 0.0)
        if spd_mph <= 0.5 and len(self.ball_trail_px) >= 2 and self.ball_trail_px[-1] and self.ball_trail_px[-2]:
            dx_p = self.ball_trail_px[-1][0] - self.ball_trail_px[-2][0]
            dy_p = self.ball_trail_px[-1][1] - self.ball_trail_px[-2][1]
            spd_mph = min(58.0, math.hypot(dx_p, dy_p) * self.fps * 0.042)
        h_ft = float((ball or {}).get("height_ft") or 0.0)
        sec = frame_idx / max(1.0, self.fps)

        pad_x = int(round(12 * sc))
        th = 2 if is_hi_res else 1
        cv2.putText(
            frame,
            "PICKVIZ VISION AI (3D MINIMAP & RULES)",
            (x1 + pad_x, y1 + int(round(20 * sc))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42 * sc,
            (0, 255, 220),
            th,
            cv2.LINE_AA,
        )
        line2 = f"F: {frame_idx + 1}/{self.total_frames} | Players: {len(players)}/4 | Ball: {ball_status} | Bounces: {self.cumulative_bounces}"
        cv2.putText(
            frame,
            line2,
            (x1 + pad_x, y1 + int(round(41 * sc))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36 * sc,
            (248, 250, 252),
            th,
            cv2.LINE_AA,
        )
        line3 = f"Time: {int(sec // 60):02d}:{int(sec % 60):02d}.{int((sec % 1) * 100):02d} | Speed: {spd_mph:.1f} MPH | Height: {h_ft:.1f} ft"
        cv2.putText(
            frame,
            line3,
            (x1 + pad_x, y1 + int(round(60 * sc))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34 * sc,
            (180, 210, 225),
            th,
            cv2.LINE_AA,
        )
        return frame

    def _draw_scoreboard_overlay(self, frame: np.ndarray, rule_frame: Dict[str, Any]) -> np.ndarray:
        """
        Render a clean, simple 2-row USAP Scoreboard in the bottom-right corner.
        """
        fh, fw = frame.shape[:2]
        sc = 1.48 if fh > fw else 1.0
        box_w = min(int(round(330 * sc)), fw - 28)
        box_h = int(round(88 * sc))

        bx2 = fw - 14
        bx1 = bx2 - box_w
        by2 = fh - 14
        by1 = by2 - box_h

        r_state = rule_frame.get("rally_state", "RALLY_LIVE")
        is_sideout = r_state == "RALLY_ENDED_SIDE_OUT"
        is_point = r_state == "RALLY_ENDED_POINT"
        border_col = (80, 225, 145) if (is_sideout or is_point) else (0, 225, 200)

        bug_over = frame.copy()
        cv2.rectangle(bug_over, (bx1, by1), (bx2, by2), (12, 18, 30), -1)
        cv2.addWeighted(bug_over, 0.86, frame, 0.14, 0.0, frame)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), border_col, 2 if (is_sideout or is_point) else 1, cv2.LINE_AA)

        px_pad = int(round(12 * sc))
        th = 2 if sc > 1.1 else 1

        # Top Row: Simple Title + Score Call Pill (0-0-2)
        state_lbl = "SIDE-OUT!" if is_sideout else ("POINT!" if is_point else "SCOREBOARD")
        cv2.putText(
            frame,
            state_lbl,
            (bx1 + px_pad, by1 + int(round(26 * sc))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44 * sc,
            (85, 240, 175) if (is_sideout or is_point) else (0, 255, 220),
            th,
            cv2.LINE_AA,
        )

        score_call = rule_frame.get("score_call", "0-0-2")
        pill_bg = (30, 115, 75) if (is_sideout or is_point) else (25, 70, 125)
        sx1, sy1 = bx2 - int(round(108 * sc)), by1 + int(round(8 * sc))
        sx2, sy2 = bx2 - int(round(10 * sc)), by1 + int(round(42 * sc))
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), pill_bg, -1)
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (195, 240, 215), 1, cv2.LINE_AA)
        cv2.putText(frame, score_call, (sx1 + int(round(12 * sc)), sy1 + int(round(25 * sc))), cv2.FONT_HERSHEY_SIMPLEX, 0.72 * sc, (255, 255, 255), 2, cv2.LINE_AA)

        # Bottom Row: Team A vs Team B + 2-Bounce Pips
        srv_team = rule_frame.get("serving_team", "A")
        mark_a = "*" if srv_team == "A" else ""
        mark_b = "*" if srv_team == "B" else ""
        team_txt = f"{mark_a}TEAM A: {rule_frame.get('score_a', 0)}   |   {mark_b}TEAM B: {rule_frame.get('score_b', 0)}"
        cv2.putText(
            frame,
            team_txt,
            (bx1 + px_pad, by1 + int(round(56 * sc))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42 * sc,
            (248, 250, 252),
            th,
            cv2.LINE_AA,
        )

        tb_cnt = int(rule_frame.get("two_bounce_count", 0))
        cv2.putText(frame, "2-BOUNCE:", (bx1 + px_pad, by1 + int(round(78 * sc))), cv2.FONT_HERSHEY_SIMPLEX, 0.36 * sc, (185, 210, 230), 1, cv2.LINE_AA)
        for idx_p in range(2):
            cx_p = bx1 + px_pad + int(round((82 + idx_p * 18) * sc))
            cy_p = by1 + int(round(73 * sc))
            filled = idx_p < tb_cnt
            pcol = (85, 235, 160) if filled else (80, 100, 125)
            cv2.circle(frame, (cx_p, cy_p), int(round(5 * sc)), pcol, -1 if filled else 1, cv2.LINE_AA)

        nvz_in = rule_frame.get("nvz_players_in_zone", [])
        nvz_short = "KITCHEN: IN" if nvz_in else "KITCHEN: OK"
        cv2.putText(frame, f"|  {nvz_short}", (bx1 + px_pad + int(round(120 * sc)), by1 + int(round(78 * sc))), cv2.FONT_HERSHEY_SIMPLEX, 0.36 * sc, (140, 235, 185) if nvz_in else (190, 215, 235), 1, cv2.LINE_AA)

        return frame

    def render_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        players: List[Dict[str, Any]],
        ball: Optional[Dict[str, Any]],
        bounce_event: Optional[Dict[str, Any]] = None,
        rule_frame: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Render complete broadcast animation + telemetry HUD + simple Scoreboard onto frame."""
        vis = frame.copy()
        fh, fw = vis.shape[:2]
        ui_scale = max(1.0, fw / 1280.0) if fh <= fw else max(1.2, fw / 1100.0)

        if bounce_event is not None:
            self.cumulative_bounces += 1
            c_ft = bounce_event.get("court_ft")
            px = bounce_event.get("pixel") or (ball.get("pixel") if ball else None)
            subzone = get_court_subzone_rect(float(c_ft[0]), float(c_ft[1])) if c_ft else None
            persp = 0.75 + 0.55 * (float(px[1]) / max(1.0, float(fh))) if px else 1.0
            self.active_bounces.append(
                {
                    "frame_idx": frame_idx,
                    "pixel": px,
                    "court_ft": c_ft,
                    "subzone_ft": subzone,
                    "persp_scale": persp,
                    "duration": 16,
                }
            )

        vis = self._draw_bounce_zone_fill(vis, frame_idx)
        vis = self._draw_court_wireframe(vis, line_w=2 if fw >= 1280 else 1)
        vis = self._draw_player_halos(vis, players, frame_idx)
        vis = self._draw_bounce_impact_fx(vis, frame_idx)
        vis = self._draw_ball_ribbon(vis, ball, ui_scale=ui_scale)
        vis = self._draw_wireframe_minimap(vis, players, ball, frame_idx)
        vis = self._draw_telemetry_header_hud(vis, frame_idx, players, ball)
        if rule_frame is not None:
            vis = self._draw_scoreboard_overlay(vis, rule_frame)
        return vis
