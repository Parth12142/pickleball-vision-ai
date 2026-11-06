/**
 * landmarks.js — Interactive 3D World Landmarks & Feature Modals
 */

export const WORLD_LANDMARKS = [
  {
    id: "center_court",
    name: "🏟️ CENTER COURT (9s RALLY VISION)",
    tag: "LIVE MATCH TELEMETRY",
    x: 0,
    z: 0,
    radius: 13,
    color: "#CCFF00",
    description: "Official USAP 20×44 ft Championship Court — Watch the real 9.63-second rally with 4-player Hungarian tracking and ball velocity radar.",
  },
  {
    id: "kitchen",
    name: "🛑 7-FT KITCHEN NVZ SENSOR LAB",
    tag: "USAP RULE 9 WATCHDOG",
    x: -34,
    z: -22,
    radius: 10,
    color: "#FF5722",
    description: "Real-time ankle and shoe keypoint tracking along the 7-foot Non-Volley Zone line to catch foot faults and post-volley momentum.",
  },
  {
    id: "hawkeye",
    name: "🦅 HAWK-EYE LINE VAR TOWER",
    tag: "MILLIMETER BOUNCE VAR",
    x: 34,
    z: -22,
    radius: 10,
    color: "#38BDF8",
    description: "Sub-pixel parabolic trajectory fitting and homography projection that pinpoints ball compression and calls IN or OUT to the millimeter.",
  },
  {
    id: "coach",
    name: "🤖 AI BIOMECHANICS PAVILION",
    tag: "17-KEYPOINT POSE COACH",
    x: -34,
    z: 24,
    radius: 10,
    color: "#A855F7",
    description: "Measures knee bend, elbow angle, paddle contact point in front of hip, and stance width against pro USAP biomechanics.",
  },
  {
    id: "pro_shop",
    name: "🎟️ TOURNAMENT PRO SHOP",
    tag: "MEMBERSHIP PASSES",
    x: 34,
    z: 24,
    radius: 10,
    color: "#10B981",
    description: "Unlock Paddle Free, Paddle Pro ($29/mo), or Paddle Premium ($79/mo) tournament analytics and automated officiating.",
  },
  {
    id: "calibration",
    name: "🎯 COURT CALIBRATION GARAGE",
    tag: "4-POINT HOMOGRAPHY PINNING",
    x: 0,
    z: 36,
    radius: 10,
    color: "#F59E0B",
    description: "Human-in-the-Loop 4-corner court pinning workbench. Fine-tune baseline corners with 0.0000 ft reprojection RMSE.",
  },
  {
    id: "golden_egg",
    name: "🏆 SECRET GOLDEN PADDLE MONUMENT",
    tag: "HIDDEN EASTER EGG",
    x: 0,
    z: -42,
    radius: 8,
    color: "#FFD700",
    description: "Congratulations! You discovered the legendary Golden Paddle hidden behind the North Bleachers!",
  },
];

export class LandmarkModalManager {
  constructor(achievementMgr) {
    this.achievementMgr = achievementMgr;
    this.modalEl = document.getElementById("landmark-modal");
    this.tagEl = document.getElementById("modal-tag");
    this.titleEl = document.getElementById("modal-title");
    this.bodyEl = document.getElementById("modal-body-content");
    this.activeZone = null;

    document.getElementById("btn-close-modal")?.addEventListener("click", () => {
      this.close();
    });

    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !this.modalEl?.classList.contains("hidden")) {
        this.close();
      }
    });
  }

  open(landmark) {
    if (!this.modalEl) return;
    this.activeZone = landmark;
    this.achievementMgr?.recordZoneVisit(landmark.id);

    if (this.tagEl) this.tagEl.textContent = landmark.tag;
    if (this.titleEl) this.titleEl.textContent = landmark.name;
    if (this.bodyEl) {
      this.bodyEl.innerHTML = this.renderZoneHTML(landmark);
      this.bindDynamicEvents(landmark);
    }

    this.modalEl.classList.remove("hidden");
  }

  close() {
    if (!this.modalEl) return;
    this.modalEl.classList.add("hidden");
    if (this.bodyEl) {
      // Stop any playing video when closing
      const vid = this.bodyEl.querySelector("video");
      if (vid) vid.pause();
    }
    this.activeZone = null;
  }

  renderZoneHTML(lm) {
    if (lm.id === "center_court") {
      return `
        <div class="modal-grid-2col">
          <div class="modal-video-wrap">
            <video src="/outputs/match1_base_preview.mp4" autoplay loop muted playsinline controls></video>
          </div>
          <div>
            <p style="color:#CBD5E1; line-height:1.6; margin-top:0;">${lm.description}</p>
            <div class="modal-stat-cards">
              <div class="m-stat">
                <small>Clip Duration</small>
                <strong>9.63s (289f)</strong>
              </div>
              <div class="m-stat">
                <small>Players Locked</small>
                <strong>4.0 / 4.0 (0 Swaps)</strong>
              </div>
              <div class="m-stat">
                <small>Ball Tracked</small>
                <strong>91.0% (263f)</strong>
              </div>
              <div class="m-stat">
                <small>Court RMSE</small>
                <strong>0.0000 ft</strong>
              </div>
            </div>
            <div style="margin-top: 18px; padding: 14px; background: #141E30; border-radius: 12px; border-left: 4px solid #CCFF00;">
              <strong style="display:block; font-size:13px; color:#CCFF00; margin-bottom:4px;">LIVE USAP SCORE CALL</strong>
              <span style="font-family:var(--font-mono); font-size:18px; font-weight:800;">8 - 6 - 2 (P2 Serving Left/Odd)</span>
            </div>
          </div>
        </div>
      `;
    }

    if (lm.id === "kitchen") {
      return `
        <div class="modal-grid-2col">
          <div style="background:#101929; border-radius:14px; padding:24px; border:1px solid #26334D;">
            <h4 style="margin:0 0 12px; color:#FF5722; font-family:var(--font-display);">USAP RULE 9 • 7-FT NON-VOLLEY ZONE</h4>
            <p style="color:#94A3B8; font-size:14px; line-height:1.6;">
              Players cannot volley the ball while touching the 7-foot Kitchen area or the Kitchen line—including post-volley momentum landings.
            </p>
            <div class="modal-stat-cards">
              <div class="m-stat"><small>P1 Left Ankle</small><strong style="color:#10B981;">SAFE (+185 mm)</strong></div>
              <div class="m-stat"><small>P2 Right Toe</small><strong style="color:#10B981;">SAFE (+112 mm)</strong></div>
              <div class="m-stat"><small>Momentum Window</small><strong>1.50 sec</strong></div>
              <div class="m-stat"><small>Zone Depth</small><strong>7.00 ft (2.13 m)</strong></div>
            </div>
          </div>
          <div style="display:flex; flex-direction:column; justify-content:center; gap:12px;">
            <div class="m-stat">
              <small>STEP 1 • POSE KEYPOINTS</small>
              <strong style="font-size:15px; color:#FFF;">Shoe &amp; Ankle Extraction</strong>
              <p style="margin:4px 0 0; font-size:12px; color:#94A3B8;">Tracks left/right ankles and toes for all 4 players in court feet.</p>
            </div>
            <div class="m-stat">
              <small>STEP 2 • VOLLEY VS DINK</small>
              <strong style="font-size:15px; color:#FFF;">Bounce State Verification</strong>
              <p style="margin:4px 0 0; font-size:12px; color:#94A3B8;">Distinguishes airborne volleys from legal kitchen dinks after a bounce.</p>
            </div>
          </div>
        </div>
      `;
    }

    if (lm.id === "hawkeye") {
      return `
        <div class="modal-grid-2col">
          <div style="background:#101929; border-radius:14px; padding:24px; border:1px solid #26334D;">
            <h4 style="margin:0 0 12px; color:#38BDF8; font-family:var(--font-display);">OPTIC HAWK-EYE LINE CALLING VAR</h4>
            <p style="color:#94A3B8; font-size:14px; line-height:1.6;">${lm.description}</p>
            <div class="modal-stat-cards">
              <div class="m-stat"><small>Bounce #1 (Frame 42)</small><strong style="color:#10B981;">IN (+112.0 mm)</strong></div>
              <div class="m-stat"><small>Bounce #2 (Frame 115)</small><strong style="color:#10B981;">IN (+29.5 mm)</strong></div>
              <div class="m-stat"><small>Bounce #3 (Frame 188)</small><strong style="color:#10B981;">IN (+9.8 mm)</strong></div>
              <div class="m-stat"><small>Bounce #4 (Frame 254)</small><strong style="color:#EF4444;">OUT (-16.4 mm)</strong></div>
            </div>
          </div>
          <div class="modal-video-wrap">
            <img src="/api/frame0?clip=match1.mp4" alt="Court VAR Frame" style="width:100%; display:block;" />
          </div>
        </div>
      `;
    }

    if (lm.id === "calibration") {
      return `
        <div class="modal-grid-2col">
          <div class="modal-video-wrap">
            <img src="/api/frame0?clip=match1.mp4" alt="Calibration Frame 0" style="width:100%; display:block;" />
          </div>
          <div>
            <p style="color:#CBD5E1; font-size:14px; line-height:1.5; margin-top:0;">
              Verified 4-Point Broadcast Homography for <strong>match1.mp4 (1302×740)</strong>:
            </p>
            <div class="modal-stat-cards">
              <div class="m-stat"><small>🔴 P1 Far-Left</small><strong>(425, 177) px</strong></div>
              <div class="m-stat"><small>🔵 P2 Far-Right</small><strong>(855, 177) px</strong></div>
              <div class="m-stat"><small>🟡 P3 Near-Right</small><strong>(1080, 610) px</strong></div>
              <div class="m-stat"><small>🟢 P4 Near-Left</small><strong>(204, 602) px</strong></div>
            </div>
            <button id="btn-3d-retrack" class="btn-start-drive" style="width:100%; margin-top:18px; font-size:14px; padding:12px;">
              ⚡ Retrack 9s Rally Now (289 Frames)
            </button>
            <div id="retrack-3d-status" style="margin-top:10px; font-family:var(--font-mono); font-size:12px; color:#CCFF00;"></div>
          </div>
        </div>
      `;
    }

    if (lm.id === "pro_shop") {
      return `
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:16px;">
          <div class="m-stat" style="padding:20px;">
            <small>STARTER TIER</small>
            <strong style="font-size:24px; color:#FFF;">Paddle (Free)</strong>
            <p style="color:#94A3B8; font-size:13px;">5 clip uploads/month • Standard 2D Court Twin &amp; Speed Stats.</p>
          </div>
          <div class="m-stat" style="padding:20px; border-color:#CCFF00;">
            <small style="color:#CCFF00;">MOST POPULAR</small>
            <strong style="font-size:24px; color:#CCFF00;">Paddle Pro ($29/mo)</strong>
            <p style="color:#94A3B8; font-size:13px;">50 clips/month • Hawk-Eye Line VAR, Kitchen Watchdog &amp; Heatmaps.</p>
          </div>
          <div class="m-stat" style="padding:20px; border-color:#FF5722;">
            <small style="color:#FF5722;">TOURNAMENT CLUBS</small>
            <strong style="font-size:24px; color:#FF5722;">Paddle Premium ($79/mo)</strong>
            <p style="color:#94A3B8; font-size:13px;">Unlimited clips • Full Multi-Court Broadcast HUD &amp; API Export.</p>
          </div>
        </div>
      `;
    }

    return `
      <div style="padding: 16px;">
        <p style="font-size:16px; color:#E2E8F0; line-height:1.6;">${lm.description}</p>
        <div class="modal-stat-cards">
          <div class="m-stat"><small>Status</small><strong>Unlocked &amp; Active</strong></div>
          <div class="m-stat"><small>Coordinates</small><strong>(${lm.x}, ${lm.z})</strong></div>
        </div>
      </div>
    `;
  }

  bindDynamicEvents(lm) {
    if (lm.id === "calibration") {
      const btn = document.getElementById("btn-3d-retrack");
      const statusEl = document.getElementById("retrack-3d-status");
      btn?.addEventListener("click", async () => {
        btn.disabled = true;
        if (statusEl) statusEl.textContent = "Starting AI Perception Pipeline on 9s Rally (289 frames)...";
        try {
          await fetch("/api/run-pipeline", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ clip: "match1.mp4", frames: 289 }),
          });
          if (statusEl) statusEl.textContent = "✅ Pipeline running in background! Drive to Center Court to watch.";
        } catch {
          if (statusEl) statusEl.textContent = "Error triggering pipeline.";
        } finally {
          setTimeout(() => { btn.disabled = false; }, 2000);
        }
      });
    }
  }
}
