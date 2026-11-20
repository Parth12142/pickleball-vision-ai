/**
 * app.js — Pickviz Full-Page SaaS & Multi-View Controller
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
    desc: "Film one stroke on your phone. Pickviz measures your technique using 17-keypoint skeletal biomechanics and tells you exactly what to work on.",
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
    this.activeClipKey = "match11";
    this.activePage = "home";
    this.billingCycle = "monthly";
    this.authToken = localStorage.getItem("pb_auth_token") || "";
    this.currentUser = null;
    this.isTransitioning = false;
    this.isTheaterMode = false;
    this.realTelemetry = null;
    this.theme = "light";

    this.initTheme();

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

    // 4. Interactive 4-Point Court Calibration & Pinning Workbench (Locked to 9s Rally)
    this.pinningWorkbench = new CourtPinningWorkbench({
      defaultClip: "match11.mp4",
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
        const clipStem = result.clip ? result.clip.replace(/\.[^.]+$/, "") : this.activeClipKey;
        const activeMeta = CLIPS_DATA[clipStem] || CLIPS_DATA[this.activeClipKey];
        const scoreStr = activeMeta ? activeMeta.scoreCall : "8 - 6 - 2";
        this.setText("demo-score-call", `USAP Score: ${scoreStr} (Verified RMSE: ${result.court_rmse_ft} ft)`);
        this.setText("demo-ball-tracked", `${result.ball_detect_pct}% frames tracked (${result.frames_processed}f)`);
        this.loadRealTelemetry(clipStem);
      }
    });

    this.bindEvents();
    this.loadClip(this.activeClipKey);
    this.checkDbAndSession();
    this.initBaselineMotionEngine();
    this.startLoop();
  }

  initTheme() {
    const saved = localStorage.getItem("pb_theme_pref");
    if (saved === "light" || saved === "dark") {
      this.applyTheme(saved);
    } else {
      // Auto-detect based on local time of day: 6:00 AM - 6:29 PM = Day (light), else Night (dark)
      const now = new Date();
      const totalMins = now.getHours() * 60 + now.getMinutes();
      const isDay = totalMins >= 360 && totalMins < 1110;
      this.applyTheme(isDay ? "light" : "dark");
    }
  }

  applyTheme(theme) {
    this.theme = theme === "dark" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", this.theme);
    document.body?.setAttribute("data-theme", this.theme);

    const iconEl = document.getElementById("theme-toggle-icon");
    const labelEl = document.getElementById("theme-toggle-label");
    if (iconEl) {
      iconEl.textContent = this.theme === "dark" ? "🌙" : "☀️";
    }
    if (labelEl) {
      labelEl.textContent = this.theme === "dark" ? "Night Arena" : "Day Court";
    }
  }

  toggleTheme() {
    const next = this.theme === "dark" ? "light" : "dark";
    localStorage.setItem("pb_theme_pref", next);
    this.applyTheme(next);
  }

  bindEvents() {
    // 0. Day / Night Stadium Theme Toggle
    document.getElementById("theme-toggle-btn")?.addEventListener("click", () => {
      this.toggleTheme();
    });

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

    // 3. Cinema Broadcast Theater Mode Toggle
    document.getElementById("btn-theater-mode")?.addEventListener("click", () => {
      this.toggleTheaterMode();
    });

    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && this.isTheaterMode) {
        this.toggleTheaterMode();
      }
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

    // 4b. Clip Switcher Tabs (Match 1 vs Match 2)
    document.querySelectorAll(".clip-tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const clipKey = btn.dataset.clip;
        if (clipKey) {
          this.loadClip(clipKey);
        }
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
    this.scrollToSection(pageKey, callback);
  }

  switchPage(pageKey) {
    this.scrollToSection(pageKey);
  }

  scrollToSection(pageKey, callback) {
    if (pageKey === "signup" || pageKey === "login") {
      this.setAuthFormMode(pageKey);
    }

    const sectionMap = {
      home: "#sec-home",
      "ai-coach": "#sec-ai-coach",
      scoring: "#sec-scoring",
      "line-var": "#sec-line-var",
      "kitchen-nvz": "#sec-kitchen-nvz",
      "shot-analytics": "#sec-shot-analytics",
      subscriptions: "#page-subscriptions",
      signup: "#page-auth",
      login: "#page-auth",
      auth: "#page-auth"
    };

    const selector = sectionMap[pageKey] || "#sec-home";
    const targetEl = document.querySelector(selector);

    document.querySelectorAll(".nav-menu-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.page === pageKey);
    });

    if (targetEl) {
      if (this.lenis) {
        this.lenis.scrollTo(targetEl, {
          offset: -84,
          duration: 1.15,
          onComplete: () => {
            if (callback) callback();
          }
        });
      } else {
        const y = targetEl.getBoundingClientRect().top + window.scrollY - 84;
        window.scrollTo({ top: Math.max(0, y), behavior: "smooth" });
        if (callback) setTimeout(callback, 400);
      }
    } else if (callback) {
      callback();
    }
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

    document.querySelectorAll(".clip-tab-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.clip === clipKey);
    });

    const clipFilenameMap = {
      "match11": "match11.mp4",
      "match_mlp": "match_mlp.mp4",
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

    this.realTelemetry = null;
    this.loadRealTelemetry(clipKey);

    if (this.heroVideoEl) {
      this.heroVideoEl.classList.toggle("is-portrait", Boolean(clip.isPortrait));
      this.heroVideoEl.src = clip.videoUrl;
      this.heroVideoEl.load();
      this.heroVideoEl.play().catch(() => {});
    }
    if (this.demoVideoEl) {
      this.demoVideoEl.src = clip.videoUrl;
      this.demoVideoEl.load();
      this.demoVideoEl.play().catch(() => {});
    }

    this.setText("hero-watch-label", "Explore Court Analytics");
    this.setText("footer-active-clip", `Active Clip: ${clip.shortName} (${clip.clipSeconds.toFixed(2)}s Tournament Rally)`);

    this.setText("mini-players-val", clip.quality.playersTracked);
    this.setText("mini-nvz-val", clip.quality.nvzStatus);
    this.setText("mini-line-val", clip.quality.lineCallVerdict);
    this.setText("mini-time-val", `${clip.timing.total_s.toFixed(2)}s`);

    this.setText("demo-video-title", clip.label);
    this.setText("demo-ball-tracked", clip.quality.ballDetectionRate);

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

  toggleTheaterMode() {
    const demoSec = document.getElementById("match-demo");
    const theaterBtn = document.getElementById("btn-theater-mode");
    const theaterText = document.getElementById("theater-btn-text");
    if (!demoSec) return;

    this.isTheaterMode = !this.isTheaterMode;
    demoSec.classList.toggle("theater-mode", this.isTheaterMode);
    document.body.classList.toggle("theater-active", this.isTheaterMode);

    if (theaterBtn && theaterText) {
      theaterText.textContent = this.isTheaterMode ? "Exit Theater" : "Theater Mode";
      theaterBtn.classList.toggle("active", this.isTheaterMode);
    }

    if (this.courtRenderer) {
      setTimeout(() => this.courtRenderer.resize(), 150);
    }
  }

  async loadRealTelemetry(clipKey) {
    try {
      const res = await fetch(`/outputs/${clipKey}_base_telemetry.json`);
      if (res.ok) {
        const data = await res.json();
        if (data && Array.isArray(data.telemetry_stream)) {
          this.realTelemetry = data.telemetry_stream;
          console.log(`[App] Loaded ${this.realTelemetry.length} frames of real telemetry for ${clipKey}`);
        }
      }
    } catch {
      this.realTelemetry = null;
    }
  }

  setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  }

  /* =========================================================================
     BASELINE-STYLE MOTION ENGINE:
     1) Lenis Smooth Scroll
     2) Critically-Tunable rAF Spring Physics ({ tension, friction })
     3) Opening Navy Curtain Loader (gates Hero reveals)
     4) Clip-Mask Word & Stacked-Line Reveals
     5) IntersectionObserver Inview Spring Reveals & ScrollSpy
     6) Scroll-Progress Parallax & Hover Micro-Interactions
     ========================================================================= */
  initBaselineMotionEngine() {
    this.springs = [];
    this.parallaxItems = [];
    this.lastRafTime = performance.now();
    this.lenis = null;

    // 0. Ensure auth duel canvas is sized since #page-auth is now in the scrollable flow
    setTimeout(() => {
      if (this.authAnimator) this.authAnimator.resize();
      if (this.courtRenderer) this.courtRenderer.resize();
    }, 100);

    // 1. Dynamically load Lenis via importmap (falls back gracefully if offline)
    import("lenis")
      .then((mod) => {
        const LenisClass = mod.default || mod.Lenis || mod;
        this.lenis = new LenisClass({ smoothWheel: true });
        if (this.loaderActive) {
          this.lenis.stop();
        }
      })
      .catch(() => {
        // Native smooth scroll fallback
      });

    // 2. Prepare word-by-word clip-mask elements ([data-hero-words], [data-anim-words])
    document.querySelectorAll("[data-hero-words], [data-anim-words]").forEach((el) => {
      if (el.dataset.wordsPrepared) return;
      el.dataset.wordsPrepared = "1";
      const rawText = (el.textContent || "").trim().replace(/\s+/g, " ");
      if (!rawText) return;
      const words = rawText.split(" ");
      el.innerHTML = words
        .map((w) => `<span class="clip-word-box"><span class="clip-word-inner">${w}</span></span>`)
        .join(" ");
    });

    // 3. Helper to register a critically-tunable spring state for an element
    this.getOrCreateSpring = (el, initial = {}) => {
      if (el._pbSpring) return el._pbSpring;
      const state = {
        el,
        props: {
          opacity: { x: initial.opacity ?? 1, v: 0, target: initial.opacity ?? 1, tension: 200, friction: 26 },
          y: { x: initial.y ?? 0, v: 0, target: initial.y ?? 0, tension: 200, friction: 26 },
          x: { x: initial.x ?? 0, v: 0, target: initial.x ?? 0, tension: 300, friction: 20 },
          scale: { x: initial.scale ?? 1, v: 0, target: initial.scale ?? 1, tension: 260, friction: 22 },
          hoverY: { x: 0, v: 0, target: 0, tension: 300, friction: 22 },
          hoverScale: { x: 1, v: 0, target: 1, tension: 300, friction: 22 }
        },
        parallaxX: 0,
        parallaxY: 0,
        usePercentX: false,
        dirty: true
      };
      el._pbSpring = state;
      this.springs.push(state);
      return state;
    };

    this.setSpringTarget = (el, targets, config = {}) => {
      const s = this.getOrCreateSpring(el);
      Object.keys(targets).forEach((k) => {
        if (s.props[k]) {
          s.props[k].target = targets[k];
          if (config.tension) s.props[k].tension = config.tension;
          if (config.friction) s.props[k].friction = config.friction;
        }
      });
      s.dirty = true;
    };

    // 4. Initialize Hero Gated Elements & In-View Elements
    document.querySelectorAll("[data-hero-inview]").forEach((el) => {
      const yFrom = parseFloat(el.dataset.y || "28");
      const s = this.getOrCreateSpring(el, { opacity: 0, y: yFrom, scale: 0.98 });
      s.dirty = true;
    });

    document.querySelectorAll("[data-inview]").forEach((el) => {
      const yFrom = parseFloat(el.dataset.y || "30");
      const s = this.getOrCreateSpring(el, { opacity: 0, y: yFrom, scale: 0.98 });
      s.dirty = true;
    });

    // 5. Register Scroll-Progress Parallax Items ([data-parallax-y], [data-parallax-x])
    document.querySelectorAll("[data-parallax-y], [data-parallax-x]").forEach((el) => {
      const s = this.getOrCreateSpring(el);
      let yRange = null;
      let xRange = null;
      if (el.dataset.parallaxY) {
        const parts = el.dataset.parallaxY.split(",").map(Number);
        if (parts.length === 2) yRange = parts;
      }
      if (el.dataset.parallaxX) {
        const parts = el.dataset.parallaxX.split(",").map(Number);
        if (parts.length === 2) {
          xRange = parts;
          s.usePercentX = true;
        }
      }
      this.parallaxItems.push({ el, spring: s, yRange, xRange });
    });

    // 6. Register Hover Micro-Interactions (disabled on mobile <= 768px)
    const isDesktop = () => window.innerWidth > 768;

    document.querySelectorAll("[data-hover-lift]").forEach((el) => {
      const lift = parseFloat(el.dataset.hoverLift || "-8");
      this.getOrCreateSpring(el);
      el.addEventListener("pointerenter", () => {
        if (!isDesktop()) return;
        this.setSpringTarget(el, { hoverY: lift }, { tension: 300, friction: 22 });
        const arrow = el.querySelector("[data-hover-arrow]");
        if (arrow) {
          this.setSpringTarget(arrow, { x: 8, opacity: 1 }, { tension: 300, friction: 20 });
        }
      });
      el.addEventListener("pointerleave", () => {
        this.setSpringTarget(el, { hoverY: 0 }, { tension: 300, friction: 22 });
        const arrow = el.querySelector("[data-hover-arrow]");
        if (arrow) {
          this.setSpringTarget(arrow, { x: 0, opacity: 0.6 }, { tension: 300, friction: 20 });
        }
      });
    });

    document.querySelectorAll("[data-hover-arrow]").forEach((arrow) => {
      this.getOrCreateSpring(arrow, { opacity: 0.6, x: 0 });
    });

    document.querySelectorAll("[data-hover-scale]").forEach((el) => {
      const targetScale = parseFloat(el.dataset.hoverScale || "1.03");
      this.getOrCreateSpring(el);
      el.addEventListener("pointerenter", () => {
        if (!isDesktop()) return;
        this.setSpringTarget(el, { hoverScale: targetScale }, { tension: 300, friction: 22 });
      });
      el.addEventListener("pointerleave", () => {
        this.setSpringTarget(el, { hoverScale: 1 }, { tension: 300, friction: 22 });
      });
    });

    document.querySelectorAll("[data-hover-btn]").forEach((btn) => {
      const nudgeEl = btn.querySelector("[data-hover-nudge]");
      if (nudgeEl) {
        this.getOrCreateSpring(nudgeEl, { x: 0 });
        btn.addEventListener("pointerenter", () => {
          if (!isDesktop()) return;
          this.setSpringTarget(nudgeEl, { x: 5 }, { tension: 320, friction: 20 });
        });
        btn.addEventListener("pointerleave", () => {
          this.setSpringTarget(nudgeEl, { x: 0 }, { tension: 320, friction: 20 });
        });
      }
    });

    // 7. Helper functions to trigger Clip-Mask Lines & Words
    const revealLines = (container) => {
      if (container.dataset.revealed) return;
      container.dataset.revealed = "1";
      const stagger = parseInt(container.dataset.stagger || "120", 10);
      const baseDelay = parseInt(container.dataset.delay || "0", 10);
      container.querySelectorAll(".clip-line").forEach((line, idx) => {
        setTimeout(() => line.classList.add("revealed"), baseDelay + idx * stagger);
      });
    };

    const revealWords = (container) => {
      if (container.dataset.revealed) return;
      container.dataset.revealed = "1";
      const stagger = parseInt(container.dataset.stagger || "26", 10);
      const baseDelay = parseInt(container.dataset.delay || "0", 10);
      container.querySelectorAll(".clip-word-box").forEach((w, idx) => {
        setTimeout(() => w.classList.add("revealed"), baseDelay + idx * stagger);
      });
    };

    // 8. IntersectionObserver for Scroll-Triggered Reveals (plays once on enter)
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          const el = entry.target;

          if (el.hasAttribute("data-inview")) {
            const delay = parseInt(el.dataset.delay || "0", 10);
            setTimeout(() => {
              this.setSpringTarget(el, { opacity: 1, y: 0, scale: 1 }, { tension: 190, friction: 26 });
            }, delay);
          }

          if (el.hasAttribute("data-anim-lines")) {
            revealLines(el);
          }

          if (el.hasAttribute("data-anim-words")) {
            revealWords(el);
          }

          io.unobserve(el);
        });
      },
      { threshold: 0.14, rootMargin: "0px 0px -6% 0px" }
    );

    document.querySelectorAll("[data-inview], [data-anim-lines], [data-anim-words]").forEach((el) => {
      io.observe(el);
    });

    // 9. ScrollSpy IntersectionObserver to highlight active section in sticky navbar
    const spyObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const secKey = entry.target.dataset.scrollSection;
            if (secKey) {
              document.querySelectorAll(".nav-menu-btn").forEach((btn) => {
                btn.classList.toggle("active", btn.dataset.page === secKey);
              });
            }
          }
        });
      },
      { threshold: 0.28, rootMargin: "-15% 0px -45% 0px" }
    );

    document.querySelectorAll("[data-scroll-section]").forEach((sec) => {
      spyObserver.observe(sec);
    });

    // 9b. Fullscreen Menu Overlay (Burger Button + Close Button + Backdrop + Esc)
    const menuOverlay = document.getElementById("baseline-menu-overlay");
    const openMenu = () => {
      if (!menuOverlay) return;
      menuOverlay.classList.add("open");
      menuOverlay.setAttribute("aria-hidden", "false");
      if (this.lenis) this.lenis.stop();
    };
    const closeMenu = () => {
      if (!menuOverlay) return;
      menuOverlay.classList.remove("open");
      menuOverlay.setAttribute("aria-hidden", "true");
      if (this.lenis && !this.loaderActive) this.lenis.start();
    };

    document.getElementById("btn-open-menu")?.addEventListener("click", openMenu);
    document.getElementById("btn-close-menu")?.addEventListener("click", closeMenu);
    document.getElementById("bl-menu-backdrop")?.addEventListener("click", closeMenu);
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && menuOverlay?.classList.contains("open")) {
        closeMenu();
      }
    });

    document.querySelectorAll("[data-menu-scroll]").forEach((el) => {
      el.addEventListener("click", (e) => {
        e.preventDefault();
        const targetSel = el.dataset.menuScroll;
        closeMenu();
        const targetEl = targetSel ? document.querySelector(targetSel) : null;
        if (targetEl) {
          if (this.lenis) {
            this.lenis.scrollTo(targetEl, { offset: -24, duration: 1.15 });
          } else {
            targetEl.scrollIntoView({ behavior: "smooth" });
          }
        }
      });
    });

    // 9c. 3-Slide Hero Collection Carousel (Autoplay 3800ms + Dots)
    const HERO_SLIDES = [
      {
        img: "https://api.getlayers.ai/storage/v1/object/public/public/assets/baseline-88535e4000/2.webp",
        brand: "PICKVIZ PRO",
        title: "FEATURED COURT AI",
        cta: "Explore the kit →",
        target: "#match-demo"
      },
      {
        img: "https://api.getlayers.ai/storage/v1/object/public/public/assets/baseline-88535e4000/3.webp",
        brand: "COURT SERIES",
        title: "HAWK-EYE LINE VAR",
        cta: "View line VAR →",
        target: "#sec-line-var"
      },
      {
        img: "https://api.getlayers.ai/storage/v1/object/public/public/assets/baseline-88535e4000/5.webp",
        brand: "ACADEMY KIT",
        title: "17-PT POSE COACH",
        cta: "Browse AI coach →",
        target: "#sec-ai-coach"
      }
    ];
    let heroSlideIdx = 0;
    const slideCard = document.getElementById("bl-hero-slide-card");
    const slideImg = document.getElementById("bl-slide-img");
    const slideBrand = document.getElementById("bl-slide-brand");
    const slideTitle = document.getElementById("bl-slide-title");
    const slideCta = document.getElementById("bl-slide-cta");
    const dotBtns = document.querySelectorAll("[data-hero-slide]");

    const setHeroSlide = (idx) => {
      heroSlideIdx = (idx + HERO_SLIDES.length) % HERO_SLIDES.length;
      const s = HERO_SLIDES[heroSlideIdx];
      dotBtns.forEach((b, i) => b.classList.toggle("active", i === heroSlideIdx));
      if (slideCard) {
        slideCard.style.opacity = "0";
        slideCard.style.transform = "translateY(8px) scale(0.97)";
        setTimeout(() => {
          if (slideImg) slideImg.src = s.img;
          if (slideBrand) slideBrand.textContent = s.brand;
          if (slideTitle) slideTitle.textContent = s.title;
          if (slideCta) {
            slideCta.textContent = s.cta;
            slideCta.dataset.menuScroll = s.target;
            slideCta.setAttribute("href", s.target);
          }
          slideCard.style.opacity = "1";
          slideCard.style.transform = "translateY(0) scale(1)";
        }, 180);
      }
    };

    dotBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const idx = parseInt(btn.dataset.heroSlide || "0", 10);
        setHeroSlide(idx);
      });
    });

    setInterval(() => {
      if (!this.loaderActive) {
        setHeroSlide(heroSlideIdx + 1);
      }
    }, 3800);

    // 10. Opening Navy Curtain Loader (MIN_VISIBLE_MS = 1400, EXIT_MS = 850)
    const curtain = document.getElementById("baseline-intro-loader");
    const wordmark = document.getElementById("intro-wordmark");
    const fillBar = document.getElementById("intro-progress-fill");
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const MIN_VISIBLE_MS = reducedMotion ? 200 : 1400;
    const EXIT_MS = reducedMotion ? 100 : 850;

    const triggerHeroReveals = () => {
      document.querySelectorAll("[data-hero-lines]").forEach((el) => revealLines(el));
      document.querySelectorAll("[data-hero-words]").forEach((el) => revealWords(el));
      document.querySelectorAll("[data-hero-inview]").forEach((el) => {
        const delay = parseInt(el.dataset.delay || "0", 10);
        setTimeout(() => {
          this.setSpringTarget(el, { opacity: 1, y: 0, scale: 1 }, { tension: 200, friction: 26 });
        }, delay);
      });
    };

    if (curtain) {
      this.loaderActive = true;
      window.scrollTo(0, 0);

      if (wordmark) {
        this.getOrCreateSpring(wordmark, { opacity: 0, y: 16 });
        setTimeout(() => {
          this.setSpringTarget(wordmark, { opacity: 1, y: 0 }, { tension: 200, friction: 22 });
        }, 40);
      }

      if (fillBar) {
        setTimeout(() => fillBar.classList.add("filling"), 120);
      }

      setTimeout(() => {
        this.loaderActive = false;
        if (this.lenis) this.lenis.start();
        curtain.classList.add("curtain-exit");
        triggerHeroReveals();
        setTimeout(() => {
          curtain.remove();
        }, EXIT_MS + 50);
      }, MIN_VISIBLE_MS);
    } else {
      triggerHeroReveals();
    }
  }

  stepMotionEngine(now) {
    const dt = Math.min(0.032, Math.max(0.001, (now - (this.lastRafTime || now)) / 1000));
    this.lastRafTime = now;

    // 1. Step Lenis Smooth Scroll
    if (this.lenis) {
      this.lenis.raf(now);
    }

    // 2. Compute Scroll-Progress Parallax (viewport top=bottom -> 0, bottom=top -> 1)
    const vh = window.innerHeight || 1;
    if (this.parallaxItems) {
      for (let i = 0; i < this.parallaxItems.length; i++) {
        const item = this.parallaxItems[i];
        const rect = item.el.getBoundingClientRect();
        if (rect.bottom < -200 || rect.top > vh + 200) continue;
        const progress = Math.max(0, Math.min(1, (vh - rect.top) / (vh + rect.height)));
        if (item.yRange) {
          const [fromY, toY] = item.yRange;
          item.spring.parallaxY = fromY + (toY - fromY) * progress;
          item.spring.dirty = true;
        }
        if (item.xRange) {
          const [fromX, toX] = item.xRange;
          item.spring.parallaxX = fromX + (toX - fromX) * progress;
          item.spring.dirty = true;
        }
      }
    }

    // 3. Step Critically-Tunable Springs: v += (-tension*(x-target) - friction*v) * dt; x += v*dt
    if (this.springs) {
      for (let i = 0; i < this.springs.length; i++) {
        const s = this.springs[i];
        if (!s.dirty) continue;

        let active = false;
        const keys = ["opacity", "y", "x", "scale", "hoverY", "hoverScale"];
        for (let k = 0; k < keys.length; k++) {
          const p = s.props[keys[k]];
          const disp = p.x - p.target;
          if (Math.abs(disp) > 0.0005 || Math.abs(p.v) > 0.0005) {
            const accel = -p.tension * disp - p.friction * p.v;
            p.v += accel * dt;
            p.x += p.v * dt;
            active = true;
          } else {
            p.x = p.target;
            p.v = 0;
          }
        }

        const totalY = s.props.y.x + s.props.hoverY.x + (s.parallaxY || 0);
        const totalScale = s.props.scale.x * s.props.hoverScale.x;
        const xStr = s.usePercentX
          ? `${(s.parallaxX || 0).toFixed(2)}%`
          : `${(s.props.x.x + (s.parallaxX || 0)).toFixed(2)}px`;

        s.el.style.opacity = s.props.opacity.x.toFixed(3);
        s.el.style.transform = `translate3d(${xStr}, ${totalY.toFixed(2)}px, 0) scale(${totalScale.toFixed(4)})`;

        if (!active) {
          s.dirty = false;
        }
      }
    }
  }

  startLoop() {
    const step = (now) => {
      this.stepMotionEngine(now || performance.now());

      if (this.activePage === "home" || true) {
        const clip = CLIPS_DATA[this.activeClipKey] || CLIPS_DATA["match11"];
        let frameIdx = 0;

        if (this.demoVideoEl && !isNaN(this.demoVideoEl.duration) && this.demoVideoEl.duration > 0) {
          const progress = this.demoVideoEl.currentTime / this.demoVideoEl.duration;
          frameIdx = Math.min(clip.frames - 1, Math.floor(progress * clip.frames));
        } else {
          frameIdx = Math.floor((performance.now() / 33) % clip.frames);
        }

        let telemetry = null;
        if (this.realTelemetry && this.realTelemetry.length > 0) {
          const realFrame = this.realTelemetry[Math.min(this.realTelemetry.length - 1, frameIdx)];
          if (realFrame) {
            const mappedPlayers = (realFrame.players || []).map((p) => {
              const cft = p.court_ft || [10.0, 22.0];
              return {
                id: p.slot || `P${p.id}`,
                team: p.side === "near" ? "A" : "B",
                x: Math.max(0.02, Math.min(0.98, cft[0] / 20.0)),
                y: Math.max(0.02, Math.min(0.98, cft[1] / 44.0)),
              };
            });

            let mappedBall = null;
            if (realFrame.ball && realFrame.ball.court_ft) {
              const bft = realFrame.ball.court_ft;
              const spd = realFrame.ball.speed_px_per_frame || 12.0;
              const liveMph = (realFrame.ball.speed_mph && realFrame.ball.speed_mph > 0)
                ? Number(realFrame.ball.speed_mph).toFixed(1)
                : (spd * 1.95).toFixed(1);
              mappedBall = {
                x: Math.max(0.02, Math.min(0.98, bft[0] / 20.0)),
                y: Math.max(0.02, Math.min(0.98, bft[1] / 44.0)),
                speedPx: spd,
                speedMph: liveMph,
              };
            }

            telemetry = {
              frameIndex: frameIdx,
              players: mappedPlayers.length > 0 ? mappedPlayers : getFrameTelemetry(this.activeClipKey, frameIdx).players,
              ball: mappedBall || getFrameTelemetry(this.activeClipKey, frameIdx).ball,
              activeBounce: null,
            };
          }
        }

        if (!telemetry) {
          telemetry = getFrameTelemetry(this.activeClipKey, frameIdx);
        }

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
