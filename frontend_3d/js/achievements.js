/**
 * achievements.js — Bruno Simon Style Gamified Achievements & Easter Eggs
 */

export const ACHIEVEMENTS_LIST = [
  {
    id: "first_drive",
    title: "First Serve",
    desc: "Drove your first 15 meters across the Pickleball Complex.",
    icon: "🛞",
  },
  {
    id: "speed_demon",
    title: "Speed Demon (45+ MPH)",
    desc: "Hit 45+ MPH using Shift Turbo Boost on the stadium straightaway.",
    icon: "⚡",
  },
  {
    id: "kitchen_commando",
    title: "Kitchen Commando",
    desc: "Inspected the 7-ft Non-Volley Zone (Kitchen) sensor pad.",
    icon: "🛑",
  },
  {
    id: "atp_jump",
    title: "Airborne Dink (Jump)",
    desc: "Launched the rover airborne over the court net using Spacebar.",
    icon: "🚀",
  },
  {
    id: "night_owl",
    title: "Night Match Floodlights",
    desc: "Switched the stadium to Championship Night Mode (Press N).",
    icon: "🌙",
  },
  {
    id: "golden_paddle",
    title: "Secret Golden Paddle",
    desc: "Found the hidden Golden Paddle Easter Egg in the North Woods.",
    icon: "🏆",
  },
  {
    id: "grand_slam",
    title: "Grand Slam Explorer",
    desc: "Visited all 6 interactive Pickleball Vision AI landmarks.",
    icon: "👑",
  },
];

export class AchievementManager {
  constructor() {
    this.unlocked = new Set();
    this.visitedZones = new Set();
    this.load();
    this.updateUI();
  }

  load() {
    try {
      const saved = JSON.parse(localStorage.getItem("pb_3d_achievements") || "[]");
      saved.forEach((id) => this.unlocked.add(id));
    } catch {
      // ignore
    }
  }

  save() {
    try {
      localStorage.setItem("pb_3d_achievements", JSON.stringify([...this.unlocked]));
    } catch {
      // ignore
    }
  }

  unlock(id) {
    if (this.unlocked.has(id)) return false;
    const ach = ACHIEVEMENTS_LIST.find((a) => a.id === id);
    if (!ach) return false;

    this.unlocked.add(id);
    this.save();
    this.playChime();
    this.showToast(ach);
    this.updateUI();
    return true;
  }

  recordZoneVisit(zoneId) {
    this.visitedZones.add(zoneId);
    if (zoneId === "kitchen") {
      this.unlock("kitchen_commando");
    }
    if (zoneId === "golden_egg") {
      this.unlock("golden_paddle");
    }
    if (this.visitedZones.size >= 6) {
      this.unlock("grand_slam");
    }
  }

  showToast(ach) {
    const toast = document.getElementById("achievement-toast");
    const iconEl = document.getElementById("toast-icon");
    const titleEl = document.getElementById("toast-title");
    if (!toast) return;

    if (iconEl) iconEl.textContent = ach.icon;
    if (titleEl) titleEl.textContent = `${ach.title} — ${ach.desc}`;

    toast.classList.add("show");
    clearTimeout(this._toastTimer);
    this._toastTimer = setTimeout(() => {
      toast.classList.remove("show");
    }, 4200);
  }

  updateUI() {
    const countEl = document.getElementById("ach-count-text");
    if (countEl) {
      countEl.textContent = `${this.unlocked.size} / ${ACHIEVEMENTS_LIST.length}`;
    }

    const grid = document.getElementById("ach-list-grid");
    if (grid) {
      grid.innerHTML = ACHIEVEMENTS_LIST.map((a) => {
        const isUn = this.unlocked.has(a.id);
        return `
          <div class="ach-card ${isUn ? "unlocked" : ""}">
            <div class="ach-icon">${isUn ? a.icon : "🔒"}</div>
            <div class="ach-info">
              <strong>${a.title}</strong>
              <span>${a.desc}</span>
            </div>
          </div>
        `;
      }).join("");
    }
  }

  playChime() {
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      const notes = [523.25, 659.25, 783.99, 1046.5]; // C5, E5, G5, C6
      notes.forEach((freq, idx) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = "triangle";
        osc.frequency.value = freq;
        gain.gain.setValueAtTime(0.12, ctx.currentTime + idx * 0.09);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + idx * 0.09 + 0.35);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(ctx.currentTime + idx * 0.09);
        osc.stop(ctx.currentTime + idx * 0.09 + 0.36);
      });
    } catch {
      // ignore audio context restriction
    }
  }
}
