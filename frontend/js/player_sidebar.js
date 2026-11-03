/**
 * player_sidebar.js — Renders Left Panel: Canonical Player Slots (P1-P4) & Shot Distribution
 */

export function renderPlayerSidebar(frameTelemetry, clipData) {
  const container = document.getElementById("player-slots-container");
  const distContainer = document.getElementById("shot-distribution-container");

  if (container && frameTelemetry.players) {
    container.innerHTML = frameTelemetry.players
      .map((p) => {
        const isServer = clipData.servingPlayer === p.id;
        const mX = (p.x * 6.10).toFixed(2);
        const mY = (p.y * 13.41).toFixed(2);
        return `
          <div class="player-row ${isServer ? "active-server" : ""}">
            <div class="player-tag ${p.id.toLowerCase()}">${p.id}</div>
            <div class="player-meta">
              <div class="player-zone">
                ${p.role}
                ${isServer ? '<span class="serve-pill" style="margin-left:6px;">SERVE</span>' : ""}
              </div>
              <div class="player-coords">POS: ${mX}m, ${mY}m</div>
            </div>
            <div class="player-speed-badge">${p.speedMph} mph</div>
          </div>
        `;
      })
      .join("");
  }

  if (distContainer && clipData.shotDistribution) {
    distContainer.innerHTML = clipData.shotDistribution
      .map(
        (s) => `
        <div class="bar-item">
          <div class="bar-header">
            <span>${s.label}</span>
            <span class="font-mono-tabular">${s.pct}%</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill ${s.colorClass}" style="width: ${s.pct}%"></div>
          </div>
        </div>
      `
      )
      .join("");
  }
}
