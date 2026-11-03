/**
 * telemetry_tray.js — Right Panel VAR Line Calling, Shot Velocity, & Bottom Timing Tray
 */

export function renderTelemetryTray(frameTelemetry, clipData) {
  // 1. Speedometer
  const speedVal = document.getElementById("telemetry-speed-val");
  const shotClass = document.getElementById("telemetry-shot-class");
  if (speedVal && frameTelemetry.ball) {
    speedVal.textContent = frameTelemetry.ball.speedMph;
  }
  if (shotClass && frameTelemetry.ball) {
    shotClass.textContent = frameTelemetry.ball.shotType;
  }

  // 2. Hawk-Eye VAR Bounce Decision Card
  const b = frameTelemetry.activeBounce;
  if (b) {
    const verdictEl = document.getElementById("var-verdict-badge");
    const lineEl = document.getElementById("var-line-name");
    const marginEl = document.getElementById("var-margin-val");
    const frameEl = document.getElementById("var-bounce-frame");
    const impactSpeedEl = document.getElementById("var-impact-speed");

    if (verdictEl) {
      verdictEl.textContent = b.verdict;
      verdictEl.className = `var-verdict-badge ${b.verdict.toLowerCase()}`;
    }
    if (lineEl) lineEl.textContent = b.line;
    if (marginEl) {
      const sign = b.marginMm > 0 ? "+" : "";
      marginEl.textContent = `${sign}${b.marginMm.toFixed(1)} mm`;
    }
    if (frameEl) frameEl.textContent = `Frame #${b.frame}`;
    if (impactSpeedEl) impactSpeedEl.textContent = `${b.speedMph} mph`;
  }

  // 3. Rule Watchdogs
  const nvzEl = document.getElementById("rule-nvz-val");
  const twoBounceEl = document.getElementById("rule-twobounce-val");
  const ballRateEl = document.getElementById("rule-ballrate-val");
  const playerSlotsEl = document.getElementById("rule-playerslots-val");

  if (nvzEl) nvzEl.textContent = clipData.quality.nvzStatus;
  if (twoBounceEl) twoBounceEl.textContent = clipData.quality.twoBounceStatus;
  if (ballRateEl) ballRateEl.textContent = clipData.quality.ballDetectionRate;
  if (playerSlotsEl) playerSlotsEl.textContent = clipData.quality.playersTracked;
}

export function renderBottomTimingTray(clipData) {
  const t = clipData.timing;
  const setTxt = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };

  setTxt("timing-clip-len", `${clipData.clipSeconds.toFixed(2)} s`);
  setTxt("timing-clip-sub", `${clipData.frames}f @ ${clipData.fps.toFixed(0)} FPS`);

  setTxt("timing-read-val", `${t.read_s.toFixed(2)} s`);
  setTxt("timing-read-sub", `${Math.round(clipData.frames / t.read_s)} FPS decode`);

  setTxt("timing-players-val", `${t.players_s.toFixed(2)} s`);
  setTxt("timing-players-sub", `${t.players_fps} FPS (YOLO11n)`);

  setTxt("timing-ball-val", `${t.ball_s.toFixed(2)} s`);
  setTxt("timing-ball-sub", `${t.ball_fps} FPS (1312px)`);

  setTxt("timing-render-val", `${t.render_s.toFixed(2)} s`);
  setTxt("timing-render-sub", `H.264 Broadcast`);

  setTxt("timing-total-val", `${t.total_s.toFixed(2)} s`);
  setTxt("timing-total-sub", `${t.realtime_factor.toFixed(2)}x clip length`);

  const scoreEl = document.getElementById("header-score-call");
  if (scoreEl) scoreEl.textContent = clipData.scoreCall;
}
