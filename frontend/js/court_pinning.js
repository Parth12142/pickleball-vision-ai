/**
 * court_pinning.js — Interactive 4-Point Court Calibration & Pinning Workbench
 * Allows users to visually pin, drag, and fine-tune 4 USAP baseline court corners
 * on any video clip (portrait smartphone or landscape broadcast), computes
 * homography in real-time, saves calibration to backend, and triggers AI perception runs.
 */

export class CourtPinningWorkbench {
  constructor(options = {}) {
    this.containerEl = options.containerEl || document.getElementById("court-pinning-workbench");
    this.canvas = options.canvas || document.getElementById("court-pin-canvas");
    this.ctx = this.canvas ? this.canvas.getContext("2d") : null;
    this.onCalibrated = options.onCalibrated || null;
    this.onPipelineCompleted = options.onPipelineCompleted || null;

    this.activeClip = options.defaultClip || "match1.mp4";
    this.availableClips = [];

    // Image & dimensions
    this.frameImage = new Image();
    this.imageLoaded = false;
    this.nativeWidth = 1302;
    this.nativeHeight = 740;
    this.scale = 1.0;
    this.offsetX = 0;
    this.offsetY = 0;

    // 4 Draggable Corners in native image pixels for 9s rally (match1.mp4):
    // [0]: Far-Left  (0, 0) ft
    // [1]: Far-Right (20, 0) ft
    // [2]: Near-Right (20, 44) ft
    // [3]: Near-Left  (0, 44) ft
    this.corners = [
      { x: 425.0, y: 177.0, label: "P1 Far-Left (0, 0) ft", color: "#FF4D4D", code: "p1" },
      { x: 855.0, y: 177.0, label: "P2 Far-Right (20, 0) ft", color: "#00BFFF", code: "p2" },
      { x: 1080.0, y: 610.0, label: "P3 Near-Right (20, 44) ft", color: "#FFD700", code: "p3" },
      { x: 204.0, y: 602.0, label: "P4 Near-Left (0, 44) ft", color: "#00FF7F", code: "p4" },
    ];

    this.draggedPinIndex = null;
    this.hoverPinIndex = null;
    this.dragRadius = 18;
    this.isCalibrated = false;
    this.currentRmse = 0.0;
    this.homographyMatrix = null;
    this.pollInterval = null;

    this.bindEvents();
    this.loadClipsList().then(() => {
      this.loadClipCalibration(this.activeClip);
    });
  }

  async loadClipsList() {
    try {
      const res = await fetch("/api/clips");
      const data = await res.json();
      if (data.ok && Array.isArray(data.clips)) {
        this.availableClips = data.clips;
        this.renderClipSelectOptions();
      }
    } catch (err) {
      console.warn("[CourtPinning] Could not fetch clips list:", err);
    }
  }

  renderClipSelectOptions() {
    const select = document.getElementById("calib-clip-select");
    if (!select) return;
    select.innerHTML = this.availableClips
      .map((c) => {
        const isSel = c.name === this.activeClip ? "selected" : "";
        const tag = c.is_portrait ? "📱 9:16 Portrait" : "📺 16:9 Landscape";
        return `<option value="${c.name}" ${isSel}>${c.name} (${c.resolution} • ${tag})</option>`;
      })
      .join("");
  }

  async loadClipCalibration(clipName) {
    this.activeClip = clipName;
    this.imageLoaded = false;
    this.setLoading(true, `Loading frame 0 for ${clipName}...`);

    // 1. Fetch calibration data
    try {
      const res = await fetch(`/api/court-calibration?clip=${encodeURIComponent(clipName)}`);
      const data = await res.json();
      if (data.ok && Array.isArray(data.corners_px) && data.corners_px.length === 4) {
        data.corners_px.forEach((pt, i) => {
          this.corners[i].x = Math.round(pt[0]);
          this.corners[i].y = Math.round(pt[1]);
        });
        this.isCalibrated = data.is_saved;
        this.currentRmse = data.rmse_ft || 0.0;
        this.homographyMatrix = data.homography || null;
      }
    } catch (err) {
      console.warn("[CourtPinning] Error loading calibration:", err);
    }

    // 2. Load Frame 0 image
    const imgUrl = `/api/frame0?clip=${encodeURIComponent(clipName)}&t=${Date.now()}`;
    this.frameImage = new Image();
    this.frameImage.onload = () => {
      this.imageLoaded = true;
      this.nativeWidth = this.frameImage.naturalWidth || 1280;
      this.nativeHeight = this.frameImage.naturalHeight || 720;
      this.setLoading(false);
      this.resizeCanvas();
      this.updateCoordinateInputs();
      this.updateStatusBadge();
      this.redraw();
    };
    this.frameImage.onerror = () => {
      this.setLoading(false, "Failed to load video frame. Please check clip file.");
    };
    this.frameImage.src = imgUrl;
  }

  bindEvents() {
    if (!this.canvas) return;

    // Mouse events
    this.canvas.addEventListener("mousedown", (e) => this.onPointerDown(e));
    window.addEventListener("mousemove", (e) => this.onPointerMove(e));
    window.addEventListener("mouseup", (e) => this.onPointerUp(e));

    // Touch events for mobile/tablet
    this.canvas.addEventListener("touchstart", (e) => {
      if (e.touches.length > 0) {
        this.onPointerDown(e.touches[0]);
        e.preventDefault();
      }
    }, { passive: false });
    window.addEventListener("touchmove", (e) => {
      if (this.draggedPinIndex !== null && e.touches.length > 0) {
        this.onPointerMove(e.touches[0]);
        e.preventDefault();
      }
    }, { passive: false });
    window.addEventListener("touchend", () => this.onPointerUp());

    // Window resize
    window.addEventListener("resize", () => {
      this.resizeCanvas();
      this.redraw();
    });

    // Clip select dropdown
    const select = document.getElementById("calib-clip-select");
    if (select) {
      select.addEventListener("change", (e) => {
        this.loadClipCalibration(e.target.value);
      });
    }

    // Action buttons
    document.getElementById("btn-calib-recommend")?.addEventListener("click", () => {
      this.recommendCorners();
    });

    document.getElementById("btn-calib-save")?.addEventListener("click", () => {
      this.saveCalibration();
    });

    document.getElementById("btn-calib-run-pipeline")?.addEventListener("click", () => {
      this.triggerPipelineRun();
    });

    document.getElementById("btn-toggle-pin-court")?.addEventListener("click", () => {
      this.toggleWorkbench();
    });

    document.getElementById("btn-close-pin-court")?.addEventListener("click", () => {
      this.toggleWorkbench(false);
    });

    // Nudge buttons & coordinate inputs
    this.bindNudgeButtons();
  }

  bindNudgeButtons() {
    const pins = ["p1", "p2", "p3", "p4"];
    pins.forEach((pinCode, idx) => {
      const inputX = document.getElementById(`pin-${pinCode}-x`);
      const inputY = document.getElementById(`pin-${pinCode}-y`);

      inputX?.addEventListener("change", () => {
        const val = parseFloat(inputX.value);
        if (!isNaN(val)) {
          this.corners[idx].x = Math.max(-500, Math.min(this.nativeWidth + 500, val));
          this.resizeCanvas();
          this.redraw();
        }
      });

      inputY?.addEventListener("change", () => {
        const val = parseFloat(inputY.value);
        if (!isNaN(val)) {
          this.corners[idx].y = Math.max(-200, Math.min(this.nativeHeight + 200, val));
          this.resizeCanvas();
          this.redraw();
        }
      });

      // Nudge buttons: e.g. [data-nudge-pin="p1" data-axis="x" data-amount="-5"]
      document.querySelectorAll(`[data-nudge-pin="${pinCode}"]`).forEach((btn) => {
        btn.addEventListener("click", () => {
          const axis = btn.dataset.axis;
          const amt = parseFloat(btn.dataset.amount || 0);
          if (axis === "x") {
            this.corners[idx].x = Math.max(-500, Math.min(this.nativeWidth + 500, this.corners[idx].x + amt));
          } else if (axis === "y") {
            this.corners[idx].y = Math.max(-200, Math.min(this.nativeHeight + 200, this.corners[idx].y + amt));
          }
          this.updateCoordinateInputs();
          this.redraw();
        });
      });
    });
  }

  toggleWorkbench(forceState = null) {
    if (!this.containerEl) return;
    const isHidden = this.containerEl.classList.contains("hidden");
    const nextState = forceState !== null ? !forceState : !isHidden;

    if (nextState) {
      this.containerEl.classList.add("hidden");
    } else {
      this.containerEl.classList.remove("hidden");
      this.resizeCanvas();
      this.redraw();
      this.containerEl.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  getPointerCanvasPos(e) {
    if (!this.canvas) return { x: 0, y: 0 };
    const rect = this.canvas.getBoundingClientRect();
    const clientX = e.clientX ?? (e.touches && e.touches[0] ? e.touches[0].clientX : 0);
    const clientY = e.clientY ?? (e.touches && e.touches[0] ? e.touches[0].clientY : 0);

    const canvasX = clientX - rect.left;
    const canvasY = clientY - rect.top;

    // Convert to native image pixels
    const nativeX = (canvasX - this.offsetX) / this.scale;
    const nativeY = (canvasY - this.offsetY) / this.scale;

    return { canvasX, canvasY, nativeX, nativeY };
  }

  onPointerDown(e) {
    const { canvasX, canvasY } = this.getPointerCanvasPos(e);

    // Find closest pin within hit radius
    let closestIdx = null;
    let minDist = this.dragRadius * 1.5;

    this.corners.forEach((corner, idx) => {
      const scrX = corner.x * this.scale + this.offsetX;
      const scrY = corner.y * this.scale + this.offsetY;
      const dist = Math.hypot(canvasX - scrX, canvasY - scrY);
      if (dist < minDist) {
        minDist = dist;
        closestIdx = idx;
      }
    });

    if (closestIdx !== null) {
      this.draggedPinIndex = closestIdx;
      this.canvas.style.cursor = "grabbing";
    }
  }

  onPointerMove(e) {
    const { canvasX, canvasY, nativeX, nativeY } = this.getPointerCanvasPos(e);

    if (this.draggedPinIndex !== null) {
      // Allow dragging slightly past edges for wide-angle baseline corners
      const clampedX = Math.max(-500, Math.min(this.nativeWidth + 500, Math.round(nativeX)));
      const clampedY = Math.max(-200, Math.min(this.nativeHeight + 200, Math.round(nativeY)));

      this.corners[this.draggedPinIndex].x = clampedX;
      this.corners[this.draggedPinIndex].y = clampedY;

      this.updateCoordinateInputs();
      this.redraw();
      return;
    }

    // Check hover state
    let hoveredIdx = null;
    this.corners.forEach((corner, idx) => {
      const scrX = corner.x * this.scale + this.offsetX;
      const scrY = corner.y * this.scale + this.offsetY;
      const dist = Math.hypot(canvasX - scrX, canvasY - scrY);
      if (dist < this.dragRadius) {
        hoveredIdx = idx;
      }
    });

    if (hoveredIdx !== this.hoverPinIndex) {
      this.hoverPinIndex = hoveredIdx;
      this.canvas.style.cursor = hoveredIdx !== null ? "grab" : "crosshair";
      this.redraw();
    }
  }

  onPointerUp() {
    if (this.draggedPinIndex !== null) {
      this.draggedPinIndex = null;
      this.canvas.style.cursor = this.hoverPinIndex !== null ? "grab" : "crosshair";
      this.redraw();
    }
  }

  resizeCanvas() {
    if (!this.canvas || !this.imageLoaded) return;
    const parent = this.canvas.parentElement;
    if (!parent) return;

    const maxWidth = parent.clientWidth - 8;
    const maxHeight = 560;

    // Expand horizontal viewport gently only when corners sit outside [0, nativeWidth]
    const minX = Math.min(...this.corners.map((c) => c.x));
    const maxX = Math.max(...this.corners.map((c) => c.x));
    const padLeft = minX < 0 ? Math.abs(minX) + 56 : 0;
    const padRight = maxX > this.nativeWidth ? (maxX - this.nativeWidth) + 56 : 0;
    const totalW = this.nativeWidth + padLeft + padRight;

    const aspect = totalW / this.nativeHeight;
    let targetW = maxWidth;
    let targetH = maxWidth / aspect;

    if (targetH > maxHeight) {
      targetH = maxHeight;
      targetW = maxHeight * aspect;
    }

    this.canvas.width = Math.round(targetW);
    this.canvas.height = Math.round(targetH);

    this.scale = targetW / totalW;
    this.offsetX = padLeft * this.scale;
    this.offsetY = 0;
  }

  redraw() {
    if (!this.ctx || !this.imageLoaded) return;
    const ctx = this.ctx;
    const cw = this.canvas.width;
    const ch = this.canvas.height;

    ctx.clearRect(0, 0, cw, ch);
    ctx.fillStyle = "#080C12";
    ctx.fillRect(0, 0, cw, ch);

    // 1. Draw Frame 0 Image (centered cleanly when side margins exist)
    ctx.drawImage(
      this.frameImage,
      this.offsetX,
      this.offsetY,
      this.nativeWidth * this.scale,
      this.nativeHeight * this.scale
    );

    // 2. Compute screen coordinates of the 4 outer corners
    const pts = this.corners.map((c) => ({
      x: c.x * this.scale + this.offsetX,
      y: c.y * this.scale + this.offsetY,
    }));

    // 3. Draw Semi-Transparent Court Surface Fill
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    ctx.lineTo(pts[1].x, pts[1].y);
    ctx.lineTo(pts[2].x, pts[2].y);
    ctx.lineTo(pts[3].x, pts[3].y);
    ctx.closePath();
    ctx.fillStyle = "rgba(27, 86, 138, 0.28)";
    ctx.fill();

    // 4. Draw Regulation USAP Court Perspective Lines
    this.drawCourtLines(ctx, pts);

    // 5. Draw Outer Perimeter Boundary
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    ctx.lineTo(pts[1].x, pts[1].y);
    ctx.lineTo(pts[2].x, pts[2].y);
    ctx.lineTo(pts[3].x, pts[3].y);
    ctx.closePath();
    ctx.lineWidth = 3.5;
    ctx.strokeStyle = "#FFFFFF";
    ctx.stroke();

    // Subtle neon cyan glow
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "#00FFFF";
    ctx.stroke();

    // 6. Draw 4 Interactive Pin Handles
    this.corners.forEach((corner, idx) => {
      const scr = pts[idx];
      const isDragging = this.draggedPinIndex === idx;
      const isHovered = this.hoverPinIndex === idx;

      // Outer glow pulse
      ctx.beginPath();
      ctx.arc(scr.x, scr.y, isDragging ? 18 : (isHovered ? 15 : 12), 0, Math.PI * 2);
      ctx.fillStyle = `${corner.color}40`;
      ctx.fill();

      // Inner pin body
      ctx.beginPath();
      ctx.arc(scr.x, scr.y, isDragging ? 9 : 7, 0, Math.PI * 2);
      ctx.fillStyle = corner.color;
      ctx.fill();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = "#FFFFFF";
      ctx.stroke();

      // Pin Label & Tooltip Badge
      const isTop = idx < 2;
      const badgeY = isTop ? scr.y - 18 : scr.y + 24;

      ctx.font = "bold 11px Inter, sans-serif";
      const labelText = `${corner.code.toUpperCase()}: (${corner.x}, ${corner.y})`;
      const textWidth = ctx.measureText(labelText).width;

      ctx.fillStyle = "rgba(14, 17, 22, 0.88)";
      ctx.beginPath();
      ctx.roundRect(scr.x - textWidth / 2 - 6, badgeY - 12, textWidth + 12, 18, 6);
      ctx.fill();
      ctx.strokeStyle = corner.color;
      ctx.lineWidth = 1.2;
      ctx.stroke();

      ctx.fillStyle = "#FFFFFF";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(labelText, scr.x, badgeY - 3);
    });
  }

  _solveHomography(pts) {
    // Solve 8x8 linear system mapping USAP feet (0..20 x 0..44) -> screen pixels
    const src = [[0, 0], [20, 0], [20, 44], [0, 44]];
    const A = [];
    for (let i = 0; i < 4; i++) {
      const [u, v] = src[i];
      const { x, y } = pts[i];
      A.push([u, v, 1, 0, 0, 0, -u * x, -v * x, x]);
      A.push([0, 0, 0, u, v, 1, -u * y, -v * y, y]);
    }
    for (let col = 0; col < 8; col++) {
      let piv = col;
      for (let r = col + 1; r < 8; r++) {
        if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
      }
      if (Math.abs(A[piv][col]) < 1e-9) return null;
      [A[col], A[piv]] = [A[piv], A[col]];
      const div = A[col][col];
      for (let k = col; k < 9; k++) A[col][k] /= div;
      for (let r = 0; r < 8; r++) {
        if (r === col) continue;
        const factor = A[r][col];
        for (let k = col; k < 9; k++) A[r][k] -= factor * A[col][k];
      }
    }
    return [A[0][8], A[1][8], A[2][8], A[3][8], A[4][8], A[5][8], A[6][8], A[7][8], 1.0];
  }

  drawCourtLines(ctx, pts) {
    const H = this._solveHomography(pts);
    const proj = (u, v) => {
      if (!H) return pts[0];
      const d = H[6] * u + H[7] * v + H[8];
      return {
        x: (H[0] * u + H[1] * v + H[2]) / (d || 1),
        y: (H[3] * u + H[4] * v + H[5]) / (d || 1),
      };
    };

    const drawSeg = (a, b, color, width) => {
      const pA = proj(a[0], a[1]);
      const pB = proj(b[0], b[1]);
      ctx.beginPath();
      ctx.moveTo(pA.x, pA.y);
      ctx.lineTo(pB.x, pB.y);
      ctx.lineWidth = width;
      ctx.strokeStyle = color;
      ctx.stroke();
    };

    drawSeg([0, 22], [20, 22], "#FF4D00", 2.5); // Center Net Line (22 ft)
    drawSeg([0, 15], [20, 15], "#DFFF00", 2.0); // Far Kitchen Line (15 ft)
    drawSeg([0, 29], [20, 29], "#DFFF00", 2.0); // Near Kitchen Line (29 ft)
    drawSeg([10, 0], [10, 15], "#FFFFFF", 1.8); // Far Centerline
    drawSeg([10, 29], [10, 44], "#FFFFFF", 1.8); // Near Centerline
  }

  updateCoordinateInputs() {
    this.corners.forEach((c) => {
      const inputX = document.getElementById(`pin-${c.code}-x`);
      const inputY = document.getElementById(`pin-${c.code}-y`);
      if (inputX && document.activeElement !== inputX) inputX.value = Math.round(c.x);
      if (inputY && document.activeElement !== inputY) inputY.value = Math.round(c.y);
    });
  }

  updateStatusBadge() {
    const badge = document.getElementById("calib-status-badge");
    if (!badge) return;
    if (this.isCalibrated) {
      badge.textContent = `✅ Calibrated • RMSE: ${this.currentRmse.toFixed(4)} ft`;
      badge.className = "calib-badge success";
    } else {
      badge.textContent = "⚡ AI Recommended • Unsaved";
      badge.className = "calib-badge warning";
    }
  }

  setLoading(isLoading, message = "") {
    const loader = document.getElementById("pin-canvas-loader");
    const loaderText = document.getElementById("pin-loader-text");
    if (loader) {
      loader.style.display = isLoading ? "flex" : "none";
    }
    if (loaderText && message) {
      loaderText.textContent = message;
    }
  }

  async recommendCorners() {
    const w = this.nativeWidth;
    const h = this.nativeHeight;

    if (h > w) {
      // Portrait 9:16 phone clip (e.g. cut.mp4)
      this.corners[0].x = Math.round(w * 0.3403);
      this.corners[0].y = Math.round(h * 0.40625);
      this.corners[1].x = Math.round(w * 0.6597);
      this.corners[1].y = Math.round(h * 0.40625);
      this.corners[2].x = Math.round(w * 0.9444);
      this.corners[2].y = Math.round(h * 0.81250);
      this.corners[3].x = Math.round(w * 0.0556);
      this.corners[3].y = Math.round(h * 0.81250);
    } else {
      // Landscape 16:9 broadcast
      this.corners[0].x = Math.round(w * 0.3264);
      this.corners[0].y = Math.round(h * 0.2392);
      this.corners[1].x = Math.round(w * 0.6567);
      this.corners[1].y = Math.round(h * 0.2392);
      this.corners[2].x = Math.round(w * 0.8310);
      this.corners[2].y = Math.round(h * 0.8135);
      this.corners[3].x = Math.round(w * 0.1567);
      this.corners[3].y = Math.round(h * 0.8135);
    }

    this.isCalibrated = false;
    this.updateCoordinateInputs();
    this.updateStatusBadge();
    this.redraw();
  }

  async saveCalibration() {
    const btn = document.getElementById("btn-calib-save");
    if (btn) btn.disabled = true;

    const payload = {
      clip: this.activeClip,
      corners_px: this.corners.map((c) => [c.x, c.y]),
    };

    try {
      const res = await fetch("/api/court-calibration", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (data.ok) {
        this.isCalibrated = true;
        this.currentRmse = data.rmse_ft || 0.0;
        this.homographyMatrix = data.homography || null;
        this.updateStatusBadge();
        this.showToast(`Court corners saved for ${this.activeClip}! RMSE: ${this.currentRmse.toFixed(4)} ft.`);
        if (this.onCalibrated) {
          this.onCalibrated(this.activeClip, payload.corners_px);
        }
      } else {
        alert(`Error saving court: ${data.error || "Unknown error"}`);
      }
    } catch (err) {
      alert(`Network error saving calibration: ${err.message}`);
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  async triggerPipelineRun() {
    const btn = document.getElementById("btn-calib-run-pipeline");
    if (btn) btn.disabled = true;

    // First ensure calibration is saved
    await this.saveCalibration();

    const hud = document.getElementById("pipeline-progress-hud");
    const hudStatus = document.getElementById("pipeline-hud-status");
    const hudBar = document.getElementById("pipeline-hud-bar");

    if (hud) hud.style.display = "block";
    if (hudStatus) hudStatus.textContent = `Initiating Perception Pipeline for ${this.activeClip}...`;
    if (hudBar) hudBar.style.width = "15%";

    const clipMeta = this.availableClips.find((c) => c.name === this.activeClip);
    try {
      const res = await fetch("/api/run-pipeline", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          clip: this.activeClip || "match1.mp4",
          frames: clipMeta?.total_frames || 300,
          multiscale: true,
        }),
      });

      const data = await res.json();
      if (!data.ok && res.status !== 202) {
        alert(data.error || "Could not start pipeline run.");
        if (hud) hud.style.display = "none";
        if (btn) btn.disabled = false;
        return;
      }

      // Poll pipeline status
      this.pollPipelineStatus();
    } catch (err) {
      alert(`Failed to start pipeline: ${err.message}`);
      if (hud) hud.style.display = "none";
      if (btn) btn.disabled = false;
    }
  }

  pollPipelineStatus() {
    if (this.pollInterval) clearInterval(this.pollInterval);

    const hud = document.getElementById("pipeline-progress-hud");
    const hudStatus = document.getElementById("pipeline-hud-status");
    const hudBar = document.getElementById("pipeline-hud-bar");
    const btn = document.getElementById("btn-calib-run-pipeline");

    this.pollInterval = setInterval(async () => {
      try {
        const res = await fetch("/api/pipeline-status");
        const status = await res.json();

        if (status.running) {
          if (hudStatus) hudStatus.textContent = status.status || "Processing...";
          if (hudBar) hudBar.style.width = `${Math.max(20, status.progress_pct || 40)}%`;
        } else if (status.status === "completed") {
          clearInterval(this.pollInterval);
          this.pollInterval = null;
          if (hudBar) hudBar.style.width = "100%";
          if (hudStatus) hudStatus.textContent = `Analysis Completed! (${status.result.frames_processed} frames, ${status.result.processing_fps} FPS)`;

          this.showToast(`Pipeline completed! Court RMSE: ${status.result.court_rmse_ft} ft. Reloading preview...`);

          setTimeout(() => {
            if (hud) hud.style.display = "none";
            if (btn) btn.disabled = false;
            if (this.onPipelineCompleted) {
              this.onPipelineCompleted(status.result);
            }
          }, 1200);
        } else if (status.status === "error") {
          clearInterval(this.pollInterval);
          this.pollInterval = null;
          if (hudStatus) hudStatus.textContent = `Error: ${status.error}`;
          if (btn) btn.disabled = false;
        }
      } catch {
        // ignore network hiccup during poll
      }
    }, 800);
  }

  showToast(msg) {
    const toast = document.getElementById("calib-toast");
    if (!toast) return;
    toast.textContent = msg;
    toast.style.display = "block";
    toast.style.opacity = "1";
    setTimeout(() => {
      toast.style.opacity = "0";
      setTimeout(() => { toast.style.display = "none"; }, 300);
    }, 3500);
  }
}
