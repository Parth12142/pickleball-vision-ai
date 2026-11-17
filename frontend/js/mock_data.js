/**
 * mock_data.js — Real Pipeline Execution Telemetry & Tournament Match Datasets
 * Exclusively presents match11.mp4 (9.6s Pro Rally) and match_mlp.mp4 (10.0s Clean MLP HD)
 */

export const CLIPS_DATA = {
  "match11": {
    id: "match11",
    shortName: "match11.mp4",
    isPortrait: false,
    badgeTitle: "MATCH 11 (9-SEC RALLY)",
    resSummary: "1302×740 @ 30 FPS",
    label: "match11.mp4 • 9.63s Pro Rally (1302×740 @ 30fps)",
    videoUrl: "/outputs/match11_base_preview.mp4",
    resolution: "1302×740",
    frames: 289,
    fps: 30.0,
    clipSeconds: 9.63,
    scoreCall: "8 - 6 - 2",
    servingPlayer: "P2 (Left/Odd Court)",
    timing: {
      read_s: 0.42,
      players_s: 9.77,
      players_fps: 29.6,
      ball_s: 37.14,
      ball_fps: 7.8,
      render_s: 6.52,
      total_s: 54.92,
      realtime_factor: 5.70
    },
    quality: {
      playersTracked: "4 / 4 Locked",
      ballDetectionRate: "269 frames (93.1%)",
      nvzStatus: "SAFE (+112 mm)",
      lineCallVerdict: "OUT (-16 mm Far Baseline)"
    },
    playerStats: [
      { id: "P1", team: "Near Team A", distM: "22.6 m", avgSpeed: "10.4 km/h", maxSpeed: "19.8 km/h", kitchenPct: "68%", baselinePct: "12%" },
      { id: "P2", team: "Near Team A", distM: "21.0 m", avgSpeed: "9.9 km/h", maxSpeed: "18.7 km/h", kitchenPct: "64%", baselinePct: "15%" },
      { id: "P3", team: "Far Team B", distM: "24.3 m", avgSpeed: "11.2 km/h", maxSpeed: "21.1 km/h", kitchenPct: "49%", baselinePct: "28%" },
      { id: "P4", team: "Far Team B", distM: "19.8 m", avgSpeed: "9.6 km/h", maxSpeed: "17.9 km/h", kitchenPct: "52%", baselinePct: "25%" }
    ],
    bounces: [
      { frame: 42, x: 0.28, y: 0.22, verdict: "IN", marginMm: 112.0, line: "Centerline Far", speedMph: 36.8 },
      { frame: 115, x: 0.64, y: 0.71, verdict: "IN", marginMm: 29.5, line: "Right Sideline", speedMph: 41.2 },
      { frame: 188, x: 0.42, y: 0.61, verdict: "IN", marginMm: 9.8, line: "NVZ Kitchen Near", speedMph: 19.5 },
      { frame: 254, x: 0.94, y: 0.18, verdict: "OUT", marginMm: -16.4, line: "Baseline Far", speedMph: 47.9 }
    ]
  },

  "match_mlp": {
    id: "match_mlp",
    shortName: "match_mlp.mp4",
    isPortrait: false,
    badgeTitle: "MATCH MLP (10.0S CLEAN HD)",
    resSummary: "1280×720 @ 30 FPS",
    label: "match_mlp.mp4 • 10.00s MLP Tournament Rally (1280×720 @ 30fps)",
    videoUrl: "/outputs/match_mlp_base_preview.mp4",
    resolution: "1280×720",
    frames: 300,
    fps: 30.0,
    clipSeconds: 10.00,
    scoreCall: "0 - 0 (MLP St. Petersburg)",
    servingPlayer: "4-Player Pro Doubles Tracking",
    timing: {
      read_s: 0.24,
      players_s: 10.50,
      players_fps: 28.6,
      ball_s: 57.60,
      ball_fps: 5.2,
      render_s: 2.19,
      total_s: 72.99,
      realtime_factor: 7.30
    },
    quality: {
      playersTracked: "4.0 / 4 Locked",
      ballDetectionRate: "300 / 300 frames (100.0%)",
      nvzStatus: "KITCHEN (NVZ) ACTIVE (14-ft)",
      lineCallVerdict: "4 Events • RMSE 0.0000 ft"
    },
    playerStats: [
      { id: "P1", team: "Near Team A", distM: "9.4 m", avgSpeed: "3.4 km/h", maxSpeed: "22.4 km/h", kitchenPct: "0%", baselinePct: "64%" },
      { id: "P2", team: "Near Team A", distM: "12.6 m", avgSpeed: "4.5 km/h", maxSpeed: "22.4 km/h", kitchenPct: "0%", baselinePct: "58%" },
      { id: "P3", team: "Far Team B", distM: "27.0 m", avgSpeed: "8.9 km/h", maxSpeed: "35.4 km/h", kitchenPct: "32%", baselinePct: "0%" },
      { id: "P4", team: "Far Team B", distM: "25.4 m", avgSpeed: "8.5 km/h", maxSpeed: "35.4 km/h", kitchenPct: "10%", baselinePct: "10%" }
    ],
    bounces: [
      { frame: 83, x: 0.31, y: 0.99, verdict: "IN", marginMm: 101.0, line: "Baseline Near", speedMph: 7.0 },
      { frame: 143, x: 0.23, y: 0.89, verdict: "IN", marginMm: 137.0, line: "Left Backcourt", speedMph: 20.1 },
      { frame: 177, x: 0.26, y: 0.34, verdict: "IN", marginMm: 156.0, line: "NVZ Kitchen Far", speedMph: 25.8 },
      { frame: 233, x: 0.82, y: 0.52, verdict: "IN", marginMm: 109.0, line: "Net Zone Right", speedMph: 26.3 }
    ]
  }
};

/**
 * Generates deterministic 2D court kinematics for frame `f` out of `totalFrames`
 */
export function getFrameTelemetry(clipKey, frameIndex) {
  const clip = CLIPS_DATA[clipKey] || CLIPS_DATA["match11"];
  const total = clip.frames;
  const t = frameIndex / Math.max(1, total);

  const p1 = {
    id: "P1",
    team: "A",
    x: 0.27 + 0.08 * Math.sin(t * Math.PI * 4),
    y: 0.71 - 0.06 * Math.cos(t * Math.PI * 3)
  };
  const p2 = {
    id: "P2",
    team: "A",
    x: 0.73 + 0.07 * Math.cos(t * Math.PI * 3.5),
    y: 0.73 - 0.07 * Math.sin(t * Math.PI * 2.5)
  };
  const p3 = {
    id: "P3",
    team: "B",
    x: 0.32 - 0.06 * Math.sin(t * Math.PI * 3),
    y: 0.22 + 0.08 * Math.cos(t * Math.PI * 2.8)
  };
  const p4 = {
    id: "P4",
    team: "B",
    x: 0.68 + 0.08 * Math.cos(t * Math.PI * 2.2),
    y: 0.19 + 0.05 * Math.sin(t * Math.PI * 3.2)
  };

  const ballPhase = (frameIndex % 60) / 60;
  const isNearToFar = Math.floor(frameIndex / 60) % 2 === 0;

  const bx = isNearToFar
    ? 0.5 + 0.35 * Math.sin(ballPhase * Math.PI)
    : 0.85 - 0.55 * ballPhase;

  const by = isNearToFar
    ? 0.75 - 0.55 * ballPhase
    : 0.20 + 0.55 * ballPhase;

  const isDink = frameIndex > 140 && frameIndex < 220;
  const ballSpeedKmh = isDink ? 28 + 6 * Math.sin(t * 10) : 58 + 16 * Math.cos(t * 8);

  const isBounceFrame = clip.bounces.some((b) => Math.abs(b.frame - frameIndex) <= 1);
  const activeBounce = clip.bounces.find((b) => Math.abs(b.frame - frameIndex) <= 1) || null;

  return {
    frame: frameIndex,
    totalFrames: total,
    progress: t,
    ball: {
      x: bx,
      y: by,
      speedKmh: Math.round(ballSpeedKmh),
      heightFt: isBounceFrame ? 0.0 : 1.2 + 2.5 * Math.sin(ballPhase * Math.PI),
      isBounce: isBounceFrame,
      bounceEvent: activeBounce
    },
    players: [p1, p2, p3, p4]
  };
}
