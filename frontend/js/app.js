/**
 * app.js — Pickleball Vision AI Full-Page SaaS & Multi-View Controller
 * - Live 60 FPS Pickleball Background Animation (PickleballBgAnimator)
 * - Dedicated Sign Up / Login Paddle-vs-Paddle Volley Animation (AuthPaddleDuelAnimator)
 * - 3-Tier Business Subscription Page (Paddle Free, Paddle Pro, Paddle Premium)
 * - Real PostgreSQL + SQLAlchemy Authentication & Subscription API integration
 */

import { CLIPS_DATA, getFrameTelemetry } from "./mock_data.js";
import {
  WorkbenchCourtRenderer,
  PickleballBgAnimator,
  AuthPaddleDuelAnimator
} from "./court_canvas.js";
import { CourtPinningWorkbench } from "./court_pinning.js";

const UPCOMING_PAGES = {
  "ai-coach": {
    badge: "COMING SOON • HOW IT WORKS · AI PICKLEBALL COACH",
    title: "ANALYSE YOUR OWN TRAINING. RAISE YOUR PERFORMANCE.",
    desc: "Film one stroke on your phone. Pickleball AI measures your technique using 17-keypoint skeletal biomechanics and tells you exactly what to work on.",
    specs: [
      { tag: "STEP 1 • PHONE VIDEO", title: "Film One Stroke on Your Phone", text: "Record a single Forehand Dink, Third-Shot Drive, Volley, or Serve from your phone camera." },
      { tag: "STEP 2 • 17-KEYPOINT AI", title: "Measures Your Technique", text: "Pretrained YOLO11-Pose tracks your knee bend, elbow angle, contact point in front of hip, and stance width." },
      { tag: "STEP 3 • PRO FEEDBACK", title: "Tells You What to Work On", text: "Compares your joint angles against USAP Pro biomechanics and prescribes targeted drills." }
    ]
  },
  scoring: {
    badge: "COMING SOON • USAP RULEBOOK ENGINE",
    title: "AUTOMATED USAP 0-0-2 SCORING",
    desc: "Autonomous tournament point-counting state machine tracking Side-Out rotations, Server #1 / Server #2 transitions, Left/Right court swaps, and the Two-Bounce rule in real time.",
    specs: [
      { tag: "RULE 4.A", title: "Two-Bounce Verification", text: "Verifies serve bounce and return bounce before allowing any volley." },
      { tag: "SIDE-OUT FSM", title: "3-Number Score Call", text: "Automatically updates [Serving - Receiving - Server #] after every rally." },
      { tag: "SERVER ROTATION", title: "Left / Right Court Check", text: "Tracks P1–P4 positions to flag wrong-server or wrong-court faults." }
    ]
  },
  "line-var": {
    badge: "COMING SOON • HAWK-EYE BOUNCE VAR",
    title: "IN / OUT LINE CALLING VAR",
    desc: "Sub-pixel parabolic trajectory fitting and homography projection that pinpoints exact ball compression on the court and calls IN or OUT with millimeter margin.",
    specs: [
      { tag: "SUB-PIXEL FIT", title: "Parabolic Bounce Detection", text: "Fits a 5-frame curve around vertical velocity inflection points." },
      { tag: "MILLIMETER VAR", title: "Baseline & Sideline Margin", text: "Measures exact distance in millimeters from the ball contact ellipse to the line." },
      { tag: "INSTANT REPLAY", title: "Auto-Zoom VAR Clip", text: "Generates a 0.25x slow-motion challenge replay for close calls." }
    ]
  },
  "kitchen-nvz": {
    badge: "COMING SOON • USAP RULE 9 WATCHDOG",
    title: "7-FT KITCHEN FOOT-FAULT AI",
    desc: "Real-time ankle and shoe keypoint tracking along the 7-foot Non-Volley Zone (Kitchen) line to detect foot touches during volleys and post-volley momentum.",
    specs: [
      { tag: "POSE KEYPOINTS", title: "Shoe & Toe Tracking", text: "Extracts left and right ankle/toe coordinates for all 4 players." },
      { tag: "RULE 9 LOGIC", title: "Volley vs Groundstroke", text: "Distinguishes airborne volleys from legal kitchen dinks after a bounce." },
      { tag: "MOMENTUM CHECK", title: "Post-Volley Landing", text: "Monitors player feet for 1.5s after paddle impact to catch momentum faults." }
    ]
  },
  "shot-analytics": {
    badge: "COMING SOON • KINEMATICS & HEATMAPS",
    title: "SHOT SPEED & HEATMAP LAB",
    desc: "Automated shot classification (Kitchen Dinks, Third-Shot Drives, Volleys, Lobs, and Overhead Smashes) with live ball velocity in MPH/km/h and spatial court heatmaps.",
    specs: [
      { tag: "5 SHOT TYPES", title: "Dink / Drive / Smash AI", text: "Classifies every stroke from trajectory arc curvature and net clearance." },
      { tag: "BALL VELOCITY", title: "Real-World Speed (km/h)", text: "Computes instantaneous and peak ball speed using 20×44 ft court scale." },
      { tag: "SPATIAL DENSITY", title: "Player Zone Heatmaps", text: "Shows % of match time spent in Kitchen, Transition, and Baseline zones." }
    ]
  }
};

const TRANSITION_METADATA = {
  home: {
    tag: "TOURNAMENT STAGE",
    title: "LOADING COURT CENTER"
  },
  subscriptions: {
    tag: "MEMBERSHIP PASS",
    title: "LOADING PRO TIERS"
  },
  "ai-coach": {
    tag: "PADDLE AI ENGINE",
    title: "INITIALIZING TACTICAL COACH"
  },
  scoring: {
    tag: "USAP RULEBOOK AI",
    title: "CALIBRATING SCORING MATRIX"
  },
  var: {
    tag: "OPTIC HAWK VAR",
    title: "CALIBRATING IN/OUT VAR CAMERA"
  },
  kitchen: {
    tag: "NVZ SENSOR AI",
    title: "LOADING KITCHEN MONITOR"
  },
  "shot-analytics": {
    tag: "TELEMETRY ENGINE",
    title: "ANALYZING SHOT HEATMAPS"
  },
  signup: {
    tag: "ATHLETE ACCESS",
    title: "OPENING REGISTRATION"
  },
  login: {
    tag: "PLAYER PORTAL",
    title: "AUTHENTICATING MEMBER"
  }
};

class PickleballWebsiteApp {
  constructor() {
    this.activeClipKey = "MicrosoftTeams-video";
    this.activePage = "home";
    this.billingCycle = "monthly";
    this.authToken = localStorage.getItem("pb_auth_token") || "";
    this.currentUser = null;
    this.isTransitioning = false;

    this.heroVideoEl = document.getElementById("hero-preview-video");
    this.demoVideoEl = document.getElementById("demo-video-player");

    // 1. Live 60 FPS Pickleball Background Animation
    const bgCanvas = document.getElementById("pickleball-bg-canvas");
    this.bgAnimator = bgCanvas ? new PickleballBgAnimator(bgCanvas) : null;

    // 2. 2D USAP Court Twin
    const courtCanvas = document.getElementById("demo-court-canvas");
    this.courtRenderer = courtCanvas ? new WorkbenchCourtRenderer(courtCanvas) : null;

    // 3. Dedicated Sign Up / Login Paddle-vs-Paddle Volley Animation
    const authCanvas = document.getElementById("auth-duel-canvas");
    this.authAnimator = authCanvas ? new AuthPaddleDuelAnimator(authCanvas) : null;

    // 4. Interactive 4-Point Court Calibration & Pinning Workbench
    this.pinningWorkbench = new CourtPinningWorkbench({
      defaultClip: "MicrosoftTeams-video.mp4",
      onCalibrated: (clipName, corners) => {
        console.log(`[App] Court calibrated for ${clipName}:`, corners);
      },
      onPipelineCompleted: (result) => {
        console.log(`[App] Pipeline run completed:`, result);
        if (this.demoVideoEl && result.preview_url) {
          const freshUrl = `${result.preview_url}?t=${Date.now()}`;
          this.demoVideoEl.src = freshUrl;
          this.demoVideoEl.load();
          this.demoVideoEl.play().catch(() => {});
        }
        if (this.heroVideoEl && result.preview_url) {
          this.heroVideoEl.src = `${result.preview_url}?t=${Date.now()}`;
          this.heroVideoEl.load();
          this.heroVideoEl.play().catch(() => {});
        }
        this.setText("demo-score-call", `Court Verified • RMSE ${result.court_rmse_ft} ft`);
        this.setText("demo-ball-tracked", `${result.ball_detect_pct}% frames tracked`);
      }
    });

    this.bindEvents();
    this.loadClip(this.activeClipKey);
    this.checkDbAndSession();
    this.startLoop();
  }

  bindEvents() {
    // 1. Multi-Page Menu Navigation with Pickleball Smash & Spin Transition
    document.querySelectorAll("[data-page]").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.preventDefault();
        const targetPage = btn.dataset.page;
        if (targetPage) {
          this.transitionToPage(targetPage);
        }
      });
    });

    // 2. Hero Watch Demo Button
    document.getElementById("hero-watch-btn")?.addEventListener("click", (e) => {
      e.preventDefault();
      this.transitionToPage("home", () => {
        document.getElementById("match-demo")?.scrollIntoView({ behavior: "smooth" });
      });
    });

    // 3. Two-Clip Switcher Buttons (MicrosoftTeams-video.mp4 & match1.mp4)
    document.querySelectorAll(".clip-tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const clipKey = btn.dataset.clip;
        if (!clipKey) return;
        document.querySelectorAll(".clip-tab-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        this.loadClip(clipKey);
      });
    });

    // 4. 2D Court View Lens Buttons
    document.querySelectorAll(".lens-pill").forEach((btn) => {
      btn.addEventListener("click", () => {
        const lens = btn.dataset.lens;
        if (!lens || !this.courtRenderer) return;
        document.querySelectorAll(".lens-pill").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        this.courtRenderer.setLens(lens);
      });
    });

    // 5. Subscription Billing Cycle Toggle (Monthly vs Annual Save 20%)
    document.querySelectorAll(".cycle-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const cycle = btn.dataset.cycle || "monthly";
        this.billingCycle = cycle;
        document.querySelectorAll(".cycle-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");

        document.querySelectorAll(".plan-price-val").forEach((priceEl) => {
          const val = priceEl.dataset[cycle];
          if (val) priceEl.textContent = val;
        });
      });
    });

    // 6. Subscription Plan Selection Buttons (Paddle, Paddle Pro, Paddle Premium)
    document.querySelectorAll("[data-select-tier]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const tier = btn.dataset.selectTier;
        if (tier) {
          this.handleSelectSubscription(tier);
        }
      });
    });

    // 7. Auth Mode Tabs (Sign Up vs Log In)
    document.querySelectorAll("[data-auth-mode]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const mode = btn.dataset.authMode;
        this.setAuthFormMode(mode);
      });
    });

    // 8. Sign Up Form Submission
    document.getElementById("signup-form")?.addEventListener("submit", (e) => {
      e.preventDefault();
      this.submitSignUp();
    });

    // 9. Log In Form Submission
    document.getElementById("login-form")?.addEventListener("submit", (e) => {
      e.preventDefault();
      this.submitLogin();
    });

    // 10. Log Out Button
    document.getElementById("nav-logout-btn")?.addEventListener("click", () => {
      this.authToken = "";
      this.currentUser = null;
      localStorage.removeItem("pb_auth_token");
      this.updateNavbarUserUI();
      this.transitionToPage("home");
    });
  }

  transitionToPage(pageKey, callback) {
    if (this.isTransitioning) return;

    // Normalize auth targets
    const resolvedTarget = (pageKey === "signup" || pageKey === "login") ? "auth" : pageKey;
    if (resolvedTarget === this.activePage && pageKey !== "signup" && pageKey !== "login") {
      window.scrollTo({ top: 0, behavior: "smooth" });
      if (callback) callback();
      return;
    }

    const loader = document.getElementById("pickleball-menu-loader");
    if (!loader) {
      this.switchPage(pageKey);
      if (callback) callback();
      return;
    }

    this.isTransitioning = true;

    // Set tactical HUD status and title
    const meta = TRANSITION_METADATA[pageKey] || {
      tag: "PICKLEBALL AI",
      title: "SWITCHING COURT VIEW"
    };
    const tagEl = document.getElementById("pb-loader-target-tag");
    const titleEl = document.getElementById("pb-loader-title");
    if (tagEl) tagEl.textContent = meta.tag;
    if (titleEl) titleEl.textContent = meta.title;

    // Reset and trigger CSS animation
    loader.classList.remove("active");
    void loader.offsetWidth; // Force reflow
    loader.classList.add("active");

    // At apex of contained volley arc (250ms), swap the underlying DOM view
    setTimeout(() => {
      this.switchPage(pageKey);
      if (callback) callback();
    }, 250);

    // Conclude transition and release lock at 500ms
    setTimeout(() => {
      loader.classList.remove("active");
      this.isTransitioning = false;
    }, 500);

    // Failsafe timer: ensure overlay never stays active if tab was backgrounded
    setTimeout(() => {
      if (this.isTransitioning) {
        loader.classList.remove("active");
        this.isTransitioning = false;
      }
    }, 750);
  }

  switchPage(pageKey) {
    // Handle 'signup' and 'login' both routing to #page-auth with the right tab active
    if (pageKey === "signup" || pageKey === "login") {
      this.activePage = "auth";
      this.setAuthFormMode(pageKey);
    } else {
      this.activePage = pageKey;
    }

    document.querySelectorAll(".nav-menu-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.page === pageKey);
    });

    const pages = ["page-home", "page-subscriptions", "page-auth", "page-coming-soon"];
    pages.forEach((id) => {
      const el = document.getElementById(id);
      if (el) el.classList.add("hidden");
    });

    if (this.activePage === "home") {
      document.getElementById("page-home")?.classList.remove("hidden");
      this.heroVideoEl?.play().catch(() => {});
      this.demoVideoEl?.play().catch(() => {});
      if (this.courtRenderer) this.courtRenderer.resize();
      window.scrollTo({ top: 0, behavior: "smooth" });
      return;
    }

    if (this.activePage === "subscriptions") {
      document.getElementById("page-subscriptions")?.classList.remove("hidden");
      window.scrollTo({ top: 0, behavior: "smooth" });
      return;
    }

    if (this.activePage === "auth") {
      document.getElementById("page-auth")?.classList.remove("hidden");
      if (this.authAnimator) this.authAnimator.resize();
      window.scrollTo({ top: 0, behavior: "smooth" });
      return;
    }

    // Upcoming feature pages -> #page-coming-soon
    const info = UPCOMING_PAGES[pageKey] || UPCOMING_PAGES.scoring;
    this.setText("soon-badge-text", info.badge);
    this.setText("soon-title-text", info.title);
    this.setText("soon-desc-text", info.desc);

    const specGrid = document.getElementById("soon-spec-grid");
    if (specGrid) {
      specGrid.innerHTML = info.specs
        .map(
          (s) => `
          <div class="soon-spec-item">
            <small>${s.tag}</small>
            <strong>${s.title}</strong>
            <span>${s.text}</span>
          </div>
        `
        )
        .join("");
    }

    document.getElementById("page-coming-soon")?.classList.remove("hidden");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  setAuthFormMode(mode) {
    const signupForm = document.getElementById("signup-form");
    const loginForm = document.getElementById("login-form");
    const tabSignup = document.getElementById("tab-btn-signup");
    const tabLogin = document.getElementById("tab-btn-login");
    const alertBox = document.getElementById("auth-alert-box");
    if (alertBox) alertBox.style.display = "none";

    if (mode === "login") {
      if (signupForm) signupForm.style.display = "none";
      if (loginForm) loginForm.style.display = "block";
      tabSignup?.classList.remove("active");
      tabLogin?.classList.add("active");
    } else {
      if (signupForm) signupForm.style.display = "block";
      if (loginForm) loginForm.style.display = "none";
      tabSignup?.classList.add("active");
      tabLogin?.classList.remove("active");
    }
  }

  showAuthAlert(msg, isError = false) {
    const alertBox = document.getElementById("auth-alert-box");
    if (!alertBox) return;
    alertBox.textContent = msg;
    alertBox.className = `auth-status-banner ${isError ? "error" : "success"}`;
    alertBox.style.display = "block";
  }

  async checkDbAndSession() {
    try {
      const dbRes = await fetch("/api/db-status");
      const dbJson = await dbRes.json();
      if (dbJson.ok) {
        this.setText("db-engine-label", dbJson.engine);
        this.setText("db-users-count", `${dbJson.registered_users} Account(s) Synced`);
      }
    } catch {
      // ignore offline
    }

    if (!this.authToken) return;
    try {
      const meRes = await fetch("/api/auth/me", {
        headers: { Authorization: `Bearer ${this.authToken}` }
      });
      const meJson = await meRes.json();
      if (meJson.ok && meJson.user) {
        this.currentUser = meJson.user;
        this.updateNavbarUserUI();
      }
    } catch {
      // ignore
    }
  }

  async submitSignUp() {
    const fullName = document.getElementById("signup-name")?.value || "";
    const email = document.getElementById("signup-email")?.value || "";
    const role = document.getElementById("signup-role")?.value || "Tournament Player";
    const tier = document.getElementById("signup-tier")?.value || "paddle";
    const password = document.getElementById("signup-password")?.value || "";

    try {
      const res = await fetch("/api/auth/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          full_name: fullName,
          email,
          club_or_role: role,
          subscription_tier: tier,
          billing_cycle: this.billingCycle,
          password
        })
      });
      const data = await res.json();
      if (!data.ok) {
        this.showAuthAlert(data.error || "Sign up failed.", true);
        return;
      }

      this.authToken = data.token;
      this.currentUser = data.user;
      localStorage.setItem("pb_auth_token", data.token);
      this.updateNavbarUserUI();
      this.checkDbAndSession();
      this.showAuthAlert(`${data.message} Active Plan: ${data.user.subscription_name}`);

      setTimeout(() => {
        this.transitionToPage("subscriptions");
        this.showSubscriptionFeedback(
          `Account created! Your ${data.user.subscription_name} subscription is now active.`
        );
      }, 500);
    } catch {
      this.showAuthAlert("Network error communicating with database server.", true);
    }
  }

  async submitLogin() {
    const email = document.getElementById("login-email")?.value || "";
    const password = document.getElementById("login-password")?.value || "";

    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password })
      });
      const data = await res.json();
      if (!data.ok) {
        this.showAuthAlert(data.error || "Invalid credentials.", true);
        return;
      }

      this.authToken = data.token;
      this.currentUser = data.user;
      localStorage.setItem("pb_auth_token", data.token);
      this.updateNavbarUserUI();
      this.showAuthAlert(`${data.message} Active Plan: ${data.user.subscription_name}`);

      setTimeout(() => {
        this.transitionToPage("home");
      }, 500);
    } catch {
      this.showAuthAlert("Network error communicating with database server.", true);
    }
  }

  async handleSelectSubscription(tier) {
    // If not logged in yet, take user to Sign Up page with their chosen Paddle tier pre-selected
    if (!this.authToken || !this.currentUser) {
      const tierSelect = document.getElementById("signup-tier");
      if (tierSelect) tierSelect.value = tier;
      this.transitionToPage("signup");
      this.showAuthAlert(
        `Create your account below to activate the ${tier.replace("_", " ").toUpperCase()} plan.`
      );
      return;
    }

    try {
      const res = await fetch("/api/subscriptions/select", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${this.authToken}`
        },
        body: JSON.stringify({
          subscription_tier: tier,
          billing_cycle: this.billingCycle
        })
      });
      const data = await res.json();
      if (data.ok && data.user) {
        this.currentUser = data.user;
        this.updateNavbarUserUI();
        this.showSubscriptionFeedback(data.message);
      }
    } catch {
      // ignore
    }
  }

  showSubscriptionFeedback(msg) {
    const banner = document.getElementById("sub-feedback-banner");
    if (!banner) return;
    banner.textContent = msg;
    banner.style.display = "block";
    banner.classList.remove("hidden");
  }

  updateNavbarUserUI() {
    const guestBox = document.getElementById("nav-guest-actions");
    const userBox = document.getElementById("nav-user-actions");

    if (this.currentUser) {
      if (guestBox) guestBox.style.display = "none";
      if (userBox) {
        userBox.style.display = "flex";
        userBox.classList.remove("hidden");
      }
      this.setText("nav-user-name", this.currentUser.full_name);
      this.setText(
        "nav-user-tier",
        this.currentUser.subscription_tier.replace("_", " ").toUpperCase()
      );
    } else {
      if (guestBox) guestBox.style.display = "flex";
      if (userBox) userBox.style.display = "none";
    }
  }

  loadClip(clipKey) {
    const clip = CLIPS_DATA[clipKey];
    if (!clip) return;
    this.activeClipKey = clipKey;

    const clipFilenameMap = {
      "MicrosoftTeams-video": "MicrosoftTeams-video.mp4",
      "match1": "match1.mp4",
      "base_preview": "match1.mp4",
      "rally_sample": "rally_sample.mp4",
      "match": "match.mp4",
    };
    const clipFilename = clipFilenameMap[clipKey] || `${clipKey}.mp4`;
    if (this.pinningWorkbench && this.pinningWorkbench.activeClip !== clipFilename) {
      this.pinningWorkbench.loadClipCalibration(clipFilename);
      const sel = document.getElementById("calib-clip-select");
      if (sel) sel.value = clipFilename;
    }

    if (this.courtRenderer) {
      this.courtRenderer.clearTrail();
    }

    if (this.heroVideoEl) {
      this.heroVideoEl.src = clip.videoUrl;
      this.heroVideoEl.load();
      this.heroVideoEl.play().catch(() => {});
    }
    if (this.demoVideoEl) {
      this.demoVideoEl.src = clip.videoUrl;
      this.demoVideoEl.load();
      this.demoVideoEl.play().catch(() => {});
    }

    this.setText("hero-clip-badge", `LIVE OUTPUT • ${clip.shortName}`);
    this.setText("mini-score-val", clip.scoreCall);
    this.setText("mini-players-val", clip.quality.playersTracked);
    this.setText("mini-nvz-val", clip.quality.nvzStatus);
    this.setText("mini-line-val", clip.quality.lineCallVerdict);
    this.setText("mini-time-val", `${clip.timing.total_s.toFixed(2)}s`);

    this.setText("demo-video-title", clip.label);
    this.setText("demo-ball-tracked", clip.quality.ballDetectionRate);
    this.setText("demo-score-call", `USAP Score: ${clip.scoreCall} (${clip.servingPlayer})`);

    this.setText("t-clip-len", `${clip.clipSeconds.toFixed(2)}s`);
    this.setText("t-total-sec", `${clip.timing.total_s.toFixed(2)}s`);
    this.setText("t-players-sec", `${clip.timing.players_s.toFixed(2)}s`);
    this.setText("t-ball-sec", `${clip.timing.ball_s.toFixed(2)}s`);

    const tbody = document.getElementById("player-stats-tbody");
    if (tbody && clip.playerStats) {
      tbody.innerHTML = clip.playerStats
        .map(
          (p) => `
          <tr>
            <td><strong>${p.id}</strong></td>
            <td>${p.team}</td>
            <td><strong>${p.distM}</strong></td>
            <td>${p.avgSpeed}</td>
            <td><strong>${p.maxSpeed}</strong></td>
            <td>${p.kitchenPct}</td>
            <td>${p.baselinePct}</td>
          </tr>
        `
        )
        .join("");
    }
  }

  setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  }

  startLoop() {
    const step = () => {
      if (this.activePage === "home") {
        const clip = CLIPS_DATA[this.activeClipKey] || CLIPS_DATA["MicrosoftTeams-video"];
        let frameIdx = 0;

        if (this.demoVideoEl && !isNaN(this.demoVideoEl.duration) && this.demoVideoEl.duration > 0) {
          const progress = this.demoVideoEl.currentTime / this.demoVideoEl.duration;
          frameIdx = Math.min(clip.frames - 1, Math.floor(progress * clip.frames));
        } else {
          frameIdx = Math.floor((performance.now() / 33) % clip.frames);
        }

        const telemetry = getFrameTelemetry(this.activeClipKey, frameIdx);
        if (this.courtRenderer) {
          this.courtRenderer.render(telemetry);
        }
        this.setText("demo-frame-counter", `Frame ${String(frameIdx).padStart(3, "0")} / ${clip.frames}`);
      }

      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
}

window.addEventListener("DOMContentLoaded", () => {
  new PickleballWebsiteApp();
});
