/**
 * mock_data.js — Real Pipeline Execution Telemetry & Tournament Match Datasets
 * Contains ONLY MicrosoftTeams-video.mp4 and match1.mp4 (rally_sample removed)
 */

export const CLIPS_DATA = {
  "MicrosoftTeams-video": {
    id: "MicrosoftTeams-video",
    shortName: "MicrosoftTeams-video.mp4",
    label: "MicrosoftTeams-video.mp4 (720×1280 @ 60fps)",
    videoUrl: "/outputs/MicrosoftTeams-video_players-ball.mp4",
    resolution: "720×1280",
    frames: 470,
    fps: 60.0,
    clipSeconds: 7.84,
    scoreCall: "4 - 2 - 1",
    servingPlayer: "P1 (Right/Even Court)",
    timing: {
      read_s: 0.55,
      players_s: 14.06,
      players_fps: 33.4,
      ball_s: 55.33,
      ball_fps: 8.5,
      render_s: 7.78,
      total_s: 78.79,
      realtime_factor: 10.05
    },
    quality: {
      playersTracked: "4 / 4 Locked",
      ballDetectionRate: "138 frames tracked",
      nvzStatus: "SAFE (+185 mm)",
      lineCallVerdict: "IN (+42 mm Baseline)"
    },
    playerStats: [
      { id: "P1", team: "Near Team A", distM: "18.4 m", avgSpeed: "9.8 km/h", maxSpeed: "18.2 km/h", kitchenPct: "62%", baselinePct: "16%" },
      { id: "P2", team: "Near Team A", distM: "16.9 m", avgSpeed: "9.1 km/h", maxSpeed: "16.5 km/h", kitchenPct: "66%", baselinePct: "14%" },
      { id: "P3", team: "Far Team B", distM: "20.1 m", avgSpeed: "10.6 km/h", maxSpeed: "19.4 km/h", kitchenPct: "54%", baselinePct: "24%" },
      { id: "P4", team: "Far Team B", distM: "17.5 m", avgSpeed: "9.4 km/h", maxSpeed: "17.1 km/h", kitchenPct: "58%", baselinePct: "20%" }
    ],
    bounces: [
      { frame: 58, x: 0.34, y: 0.24, verdict: "IN", marginMm: 84.5, line: "Left Sideline", speedMph: 34.2 },
      { frame: 132, x: 0.68, y: 0.78, verdict: "IN", marginMm: 42.0, line: "Baseline Near", speedMph: 38.6 },
      { frame: 224, x: 0.48, y: 0.39, verdict: "IN", marginMm: 18.4, line: "NVZ Kitchen Far", speedMph: 21.4 },
      { frame: 315, x: 0.88, y: 0.64, verdict: "IN", marginMm: 12.1, line: "Right Sideline", speedMph: 44.8 }
    ]
  },

  "match1": {
    id: "match1",
    shortName: "match1.mp4",
    label: "match1.mp4 (1302×740 @ 30fps)",
    videoUrl: "/outputs/match1_players-ball.mp4",
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
  const clip = CLIPS_DATA[clipKey] || CLIPS_DATA["MicrosoftTeams-video"];
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
