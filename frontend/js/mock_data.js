/**
 * mock_data.js — Real Pipeline Execution Telemetry & Tournament Match Datasets
 * Contains match1.mp4, match_mlp.mp4 (Clean 720p MLP), and cut.mp4 (1440x2560 @ 60fps)
 */

export const CLIPS_DATA = {
  "cut": {
    id: "cut",
    shortName: "cut.mp4",
    isPortrait: true,
    badgeTitle: "CUT CLIP (4.9S HD 60FPS)",
    resSummary: "1440×2560 @ 60 FPS",
    label: "cut.mp4 • 4.92s Upscaled HD Rally (1440×2560 @ 59.93fps)",
    videoUrl: "/outputs/cut_base_preview.mp4",
    resolution: "1440×2560",
    frames: 295,
    fps: 59.93,
    clipSeconds: 4.92,
    scoreCall: "Club Doubles (2x HD • 60fps)",
    servingPlayer: "4-Player 2D Court Sync",
    timing: {
      read_s: 0.28,
      players_s: 14.01,
      players_fps: 21.1,
      ball_s: 98.65,
      ball_fps: 3.0,
      render_s: 7.88,
      total_s: 130.44,
      realtime_factor: 26.51
    },
    quality: {
      playersTracked: "4.0 / 4 Locked",
      ballDetectionRate: "245 / 295 frames (83.1%)",
      nvzStatus: "KITCHEN (NVZ) ACTIVE (14-ft)",
      lineCallVerdict: "2 Events • RMSE 0.0000 ft"
    },
    playerStats: [
      { id: "P1", team: "Near Team A", distM: "6.8 m", avgSpeed: "4.5 km/h", maxSpeed: "35.4 km/h", kitchenPct: "9%", baselinePct: "65%" },
      { id: "P4", team: "Near Team A", distM: "4.7 m", avgSpeed: "3.4 km/h", maxSpeed: "16.9 km/h", kitchenPct: "0%", baselinePct: "100%" },
      { id: "P2", team: "Far Team B", distM: "6.5 m", avgSpeed: "4.7 km/h", maxSpeed: "32.7 km/h", kitchenPct: "0%", baselinePct: "0%" },
      { id: "P3", team: "Far Team B", distM: "7.1 m", avgSpeed: "5.2 km/h", maxSpeed: "24.6 km/h", kitchenPct: "0%", baselinePct: "48%" }
    ],
    bounces: [
      { frame: 97, x: 0.17, y: 0.92, verdict: "IN", marginMm: 140.0, line: "Near Backcourt", speedMph: 15.9 },
      { frame: 164, x: 0.81, y: 0.36, verdict: "IN", marginMm: 115.0, line: "NVZ Kitchen Far", speedMph: 19.9 }
    ]
  },

  "match1": {
    id: "match1",
    shortName: "match1.mp4",
    label: "match1.mp4 • 9.63s Pro Rally (1302×740 @ 30fps)",
    videoUrl: "/outputs/match1_base_preview.mp4",
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
    badgeTitle: "MATCH 2 • CLEAN MLP 720P (10.0S)",
    resSummary: "1280×720 @ 30 FPS",
    label: "match_mlp.mp4 • 10.00s Clean MLP Rally (1280×720 @ 30fps)",
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
  },

  "base_preview": {
    id: "base_preview",
    shortName: "match1_base_preview.mp4",
    label: "match1_base_preview.mp4 (Base Model Perception & Kitchen - Decoupled)",
    videoUrl: "/outputs/match1_base_preview.mp4",
    resolution: "1302×740",
    frames: 289,
    fps: 30.0,
    clipSeconds: 9.63,
    scoreCall: "Base Perception Model (Decoupled)",
    servingPlayer: "4-Player Tracking Active",
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
      ballDetectionRate: "272 / 289 frames (94.1%)",
      nvzStatus: "KITCHEN (NVZ) ACTIVE (14-ft)",
      lineCallVerdict: "Perception Only (No Scoring Rules)"
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
      { frame: 188, x: 0.42, y: 0.61, verdict: "KITCHEN", marginMm: 9.8, line: "NVZ Kitchen Near", speedMph: 19.5 },
      { frame: 254, x: 0.94, y: 0.18, verdict: "OUT", marginMm: -16.4, line: "Baseline Far", speedMph: 47.9 }
    ]
  }
};

/**
 * Generates deterministic 2D court kinematics for frame `f` out of `totalFrames`
 */
export function getFrameTelemetry(clipKey, frameIndex) {
  const clip = CLIPS_DATA[clipKey] || CLIPS_DATA["match1"];
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
    x: 0.28 + 0.09 * Math.cos(t * Math.PI * 3),
    y: 0.28 + 0.06 * Math.sin(t * Math.PI * 4)
  };
  const p4 = {
    id: "P4",
    team: "B",
    x: 0.72 - 0.08 * Math.sin(t * Math.PI * 3.5),
    y: 0.26 + 0.05 * Math.cos(t * Math.PI * 3)
  };

  const rallyCycle = t * Math.PI * 5.5;
  const bx = 0.5 + 0.24 * Math.sin(rallyCycle * 0.85);
  const by = 0.5 + 0.28 * Math.cos(rallyCycle);
  const rawSpeedPx = 14.0 + 11.5 * Math.abs(Math.sin(rallyCycle));
  const speedMph = (rawSpeedPx * 1.95).toFixed(1);

  let shotType = "RALLY DRIVE";
  if (rawSpeedPx >= 22.0) shotType = "FAST SMASH";
  else if (rawSpeedPx < 11.0) shotType = "KITCHEN DINK";

  let activeBounce = clip.bounces[0];
  for (const b of clip.bounces) {
    if (frameIndex >= b.frame) {
      activeBounce = b;
    }
  }

  return {
    frameIndex,
    players: [p1, p2, p3, p4],
    ball: {
      x: bx,
      y: by,
      speedPx: rawSpeedPx,
      speedMph,
      shotType
    },
    activeBounce
  };
}
