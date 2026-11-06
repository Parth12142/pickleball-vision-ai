/**
 * swing_app.js — Interactive Platform Controller for SwingVision Pickleball AI Demo
 * Coordinates live telemetry with video playback, ELC Hawk-Eye inspector, and analytics.
 */

import { BackgroundCourtAnimator, HawkEyeVARVisualizer } from "./court_visualizer.js";

class SwingVisionApp {
  constructor() {
    this.videoEl = document.getElementById("showcase-video");
    this.bgCanvas = document.getElementById("bg-canvas");
    this.varCanvas = document.getElementById("var-canvas");

    this.telemetryData = null;
    this.lineCalls = [];
    this.activeCallIdx = 2; // Default to the close 9.2mm IN call

    this.initVisualizers();
    this.loadData();
    this.bindEvents();
  }

  initVisualizers() {
    if (this.bgCanvas) {
      this.bgAnimator = new BackgroundCourtAnimator(this.bgCanvas);
    }
    if (this.varCanvas) {
      this.varCanvas.width = 620;
      this.varCanvas.height = 380;
      this.varVisualizer = new HawkEyeVARVisualizer(this.varCanvas);
    }
  }

  async loadData() {
    try {
      const [telRes, callRes] = await Promise.all([
        fetch("/api/telemetry").catch(() => null),
        fetch("/api/line-calls").catch(() => null),
      ]);

      if (telRes && telRes.ok) {
        this.telemetryData = await telRes.json();
      }
      if (callRes && callRes.ok) {
        const callsJson = await callRes.json();
        this.lineCalls = callsJson.calls || [];
        this.renderLineCallList();
        this.inspectLineCall(this.activeCallIdx);
      }
    } catch (err) {
      console.warn("Could not load API data:", err);
    }
  }

  bindEvents() {
    // 1. Video Frame Synchronizer: Updates live speed, score, and shot type
    if (this.videoEl) {
      this.videoEl.addEventListener("timeupdate", () => {
        this.updateLiveTelemetry(this.videoEl.currentTime);
      });
    }

    // 2. Play / Pause Toggle Button
    const btnPlay = document.getElementById("btn-ctrl-play");
    if (btnPlay && this.videoEl) {
      btnPlay.addEventListener("click", () => {
        if (this.videoEl.paused) {
          this.videoEl.play();
          btnPlay.textContent = "⏸ Pause Video";
        } else {
          this.videoEl.pause();
          btnPlay.textContent = "▶ Resume Rally";
        }
      });
    }

    // 3. Slow-Mo Toggle (0.5x vs 1.0x)
    const btnSpeed = document.getElementById("btn-ctrl-speed");
    if (btnSpeed && this.videoEl) {
      btnSpeed.addEventListener("click", () => {
        if (this.videoEl.playbackRate === 1.0) {
          this.videoEl.playbackRate = 0.5;
          btnSpeed.textContent = "⚡ 0.5x Slow-Mo";
          btnSpeed.classList.add("active");
        } else {
          this.videoEl.playbackRate = 1.0;
          btnSpeed.textContent = "⚡ 1.0x Normal";
          btnSpeed.classList.remove("active");
        }
      });
    }

    // 4. Smooth Anchor Links
    document.querySelectorAll("a[href^='#']").forEach((anchor) => {
      anchor.addEventListener("click", (e) => {
        const target = document.querySelector(anchor.getAttribute("href"));
        if (target) {
          e.preventDefault();
          target.scrollIntoView({ behavior: "smooth" });
        }
      });
    });
  }

  updateLiveTelemetry(currentTimeSec) {
    const stream = this.telemetryData?.telemetry_stream;
    const fps = this.telemetryData?.fps || 30.0;
    const currentFrame = Math.floor(currentTimeSec * fps);

    // Live Simulated Ball Speed Radar based on trajectory
    const speedEl = document.getElementById("live-speed-val");
    const shotEl = document.getElementById("live-shot-tag");
    const scoreEl = document.getElementById("live-score-val");

    if (stream && stream[currentFrame]) {
      const fData = stream[currentFrame];
      if (fData.ball && fData.ball.pixel) {
        const speed = fData.ball.speed_mph || Math.round(36 + Math.sin(currentFrame * 0.1) * 12);
        if (speedEl) speedEl.textContent = `${speed} MPH`;

        if (shotEl) {
          if (speed > 44) shotEl.textContent = "💥 SPEED-UP DRIVE";
          else if (speed > 35) shotEl.textContent = "🏓 3RD-SHOT DROP";
          else shotEl.textContent = "✨ KITCHEN DINK";
        }
      }
    } else {
      // Fallback sinusoidal realistic ball telemetry
      const fallbackSpeed = Math.round(38 + Math.abs(Math.sin(currentTimeSec * 2.5)) * 14);
      if (speedEl) speedEl.textContent = `${fallbackSpeed} MPH`;
      if (shotEl) {
        if (fallbackSpeed > 45) shotEl.textContent = "💥 SPEED-UP DRIVE";
        else if (fallbackSpeed > 38) shotEl.textContent = "🏓 3RD-SHOT DROP";
        else shotEl.textContent = "✨ KITCHEN DINK";
      }
    }

    // USAP Automated Score Progression
    if (scoreEl) {
      if (currentTimeSec < 3.5) scoreEl.textContent = "8 - 6 - 2";
      else if (currentTimeSec < 7.0) scoreEl.textContent = "8 - 6 - 2 (Rally)";
      else scoreEl.textContent = "9 - 6 - 2 (+1 PT)";
    }
  }

  renderLineCallList() {
    const listEl = document.getElementById("var-calls-list");
    if (!listEl) return;

    listEl.innerHTML = this.lineCalls
      .map(
        (c, idx) => `
        <div class="call-item-card ${idx === this.activeCallIdx ? "active" : ""}" data-idx="${idx}">
          <div class="call-item-head">
            <span class="call-badge ${c.verdict.toLowerCase()}">${c.verdict}</span>
            <div class="call-item-meta">
              <strong>Bounce #${c.bounce_idx}: ${c.line}</strong>
              <small>Frame ${c.frame} • @ ${c.timestamp_s}s • ${c.ball_speed_mph} MPH</small>
            </div>
          </div>
          <div class="call-margin-val">${c.margin_mm > 0 ? "+" : ""}${c.margin_mm} mm</div>
        </div>
      `
      )
      .join("");

    listEl.querySelectorAll(".call-item-card").forEach((card) => {
      card.addEventListener("click", () => {
        const idx = parseInt(card.dataset.idx, 10);
        this.inspectLineCall(idx);
      });
    });
  }

  inspectLineCall(idx) {
    this.activeCallIdx = idx;
    const call = this.lineCalls[idx];
    if (!call || !this.varVisualizer) return;

    // Update active highlight on cards
    document.querySelectorAll(".call-item-card").forEach((c, i) => {
      c.classList.toggle("active", i === idx);
    });

    this.varVisualizer.render(call);
  }
}

window.addEventListener("DOMContentLoaded", () => {
  window.swingApp = new SwingVisionApp();
});
