/**
 * court_canvas.js
 * 1. WorkbenchCourtRenderer: Interactive 2D USAP Pickleball Court Twin (20x44 ft)
 * 2. PickleballBgAnimator: Full-page 60 FPS live pickleball rally & bounce ripple background
 * 3. AuthPaddleDuelAnimator: Dedicated Sign Up / Login Paddle-vs-Paddle Kitchen Volley Duel animation
 * 4. SubscriptionArenaAnimator: Dedicated Subscription Page Stadium Spotlight & Golden Ball Orbits
 */

export class WorkbenchCourtRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.lens = "rally";
    this.trail = [];
    this.maxTrail = 20;

    this._resizeObserver = new ResizeObserver(() => this.resize());
    if (this.canvas.parentElement) {
      this._resizeObserver.observe(this.canvas.parentElement);
    }
    this.resize();
  }

  setLens(lensName) {
    this.lens = lensName;
  }

  clearTrail() {
    this.trail = [];
  }

  resize() {
    const parent = this.canvas.parentElement;
    if (!parent) return;
    const dpr = window.devicePixelRatio || 1;
    const w = parent.clientWidth;
    const h = parent.clientHeight;
    this.canvas.width = w * dpr;
    this.canvas.height = h * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.width = w;
    this.height = h;
  }

  getCourtBounds() {
    const padX = 28;
    const padY = 22;
    const availW = Math.max(120, this.width - padX * 2);
    const availH = Math.max(200, this.height - padY * 2);
    const ratio = 20 / 44;

    let cw = availH * ratio;
    let ch = availH;
    if (cw > availW) {
      cw = availW;
      ch = cw / ratio;
    }
    const cx = (this.width - cw) / 2;
    const cy = (this.height - ch) / 2;
    return { x: cx, y: cy, w: cw, h: ch };
  }

  render(telemetry) {
    if (!this.width || !this.height) this.resize();
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    ctx.fillStyle = "#0B1929";
    ctx.fillRect(0, 0, this.width, this.height);

    const b = this.getCourtBounds();
    this.drawUSAPCourt(ctx, b);

    if (!telemetry) return;

    if (this.lens === "heatmap") {
      this.drawCoverageHeatmap(ctx, b, telemetry.players);
    }
    if (this.lens === "kitchen") {
      this.drawKitchenWatchdog(ctx, b, telemetry.players);
    }
    if (this.lens === "var" && telemetry.activeBounce) {
      this.drawBounceVAR(ctx, b, telemetry.activeBounce);
    }

    this.drawPlayers(ctx, b, telemetry.players);
    this.drawBallAndTrail(ctx, b, telemetry.ball);
    this.drawHUD(ctx, telemetry);
  }

  drawUSAPCourt(ctx, b) {
    const { x, y, w, h } = b;
    const nvzFrac = 7 / 44;
    const netY = y + h * 0.5;
    const nvzTop = netY - h * nvzFrac;
    const nvzBot = netY + h * nvzFrac;

    ctx.fillStyle = "#1B568A";
    ctx.fillRect(x, y, w, h);

    ctx.fillStyle = "#0F3556";
    ctx.fillRect(x, nvzTop, w, nvzBot - nvzTop);

    if (this.lens === "kitchen") {
      ctx.fillStyle = "rgba(223, 255, 0, 0.14)";
      ctx.fillRect(x, nvzTop, w, nvzBot - nvzTop);
    }

    ctx.strokeStyle = "#F8FAFC";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);

    ctx.beginPath();
    ctx.moveTo(x, nvzTop);
    ctx.lineTo(x + w, nvzTop);
    ctx.moveTo(x, nvzBot);
    ctx.lineTo(x + w, nvzBot);
    ctx.stroke();

    const midX = x + w * 0.5;
    ctx.beginPath();
    ctx.moveTo(midX, y);
    ctx.lineTo(midX, nvzTop);
    ctx.moveTo(midX, nvzBot);
    ctx.lineTo(midX, y + h);
    ctx.stroke();

    ctx.save();
    ctx.strokeStyle = "#DFFF00";
    ctx.lineWidth = 2.5;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(x - 8, netY);
    ctx.lineTo(x + w + 8, netY);
    ctx.stroke();
    ctx.restore();

    ctx.fillStyle = "rgba(255, 255, 255, 0.28)";
    ctx.font = "600 9px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText("7 FT KITCHEN (NVZ)", midX, nvzTop + (netY - nvzTop) * 0.55);
    ctx.fillText("7 FT KITCHEN (NVZ)", midX, netY + (nvzBot - netY) * 0.55);
  }

  drawCoverageHeatmap(ctx, b, players) {
    for (const p of players) {
      const px = b.x + p.x * b.w;
      const py = b.y + p.y * b.h;
      const r = b.w * 0.34;
      const grad = ctx.createRadialGradient(px, py, 4, px, py, r);
      if (p.team === "A") {
        grad.addColorStop(0, "rgba(223, 255, 0, 0.48)");
        grad.addColorStop(0.55, "rgba(34, 197, 94, 0.22)");
        grad.addColorStop(1, "rgba(15, 53, 86, 0)");
      } else {
        grad.addColorStop(0, "rgba(56, 189, 248, 0.48)");
        grad.addColorStop(0.55, "rgba(59, 130, 246, 0.22)");
        grad.addColorStop(1, "rgba(15, 53, 86, 0)");
      }
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  drawKitchenWatchdog(ctx, b, players) {
    const nvzFrac = 7 / 44;
    const netY = b.y + b.h * 0.5;
    const nvzTop = netY - b.h * nvzFrac;
    const nvzBot = netY + b.h * nvzFrac;

    ctx.save();
    for (const p of players) {
      const px = b.x + p.x * b.w;
      const py = b.y + p.y * b.h;
      const targetLineY = p.team === "A" ? nvzBot : nvzTop;

      ctx.strokeStyle = "rgba(223, 255, 0, 0.7)";
      ctx.lineWidth = 1.25;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(px, targetLineY);
      ctx.stroke();

      const distFt = Math.abs(p.y - (p.team === "A" ? 0.5 + nvzFrac : 0.5 - nvzFrac)) * 44;
      const distMm = Math.round(distFt * 304.8);
      ctx.setLineDash([]);
      ctx.fillStyle = "#0D1219";
      ctx.fillRect(px - 28, (py + targetLineY) / 2 - 8, 56, 15);
      ctx.fillStyle = "#DFFF00";
      ctx.font = "700 9px 'JetBrains Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText(`+${distMm}mm`, px, (py + targetLineY) / 2 + 3);
    }
    ctx.restore();
  }

  drawBounceVAR(ctx, b, bounce) {
    const bx = b.x + bounce.x * b.w;
    const by = b.y + bounce.y * b.h;
    const isIn = bounce.verdict === "IN";
    const col = isIn ? "#22C55E" : "#EF4444";

    ctx.save();
    ctx.strokeStyle = col;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(bx, by, 15, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.ellipse(bx, by, 7, 4.5, 0.25, 0, Math.PI * 2);
    ctx.fill();

    const tagW = 114;
    const tagH = 22;
    const tx = Math.min(Math.max(b.x + 4, bx - tagW / 2), b.x + b.w - tagW - 4);
    const ty = by - 32;
    ctx.fillStyle = "#0D1219";
    ctx.strokeStyle = col;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.roundRect(tx, ty, tagW, tagH, 5);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = col;
    ctx.font = "700 10px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    const sign = bounce.marginMm >= 0 ? "+" : "";
    ctx.fillText(`${bounce.verdict} (${sign}${bounce.marginMm}mm)`, tx + tagW / 2, ty + 14);
    ctx.restore();
  }

  drawPlayers(ctx, b, players) {
    for (const p of players) {
      const px = b.x + p.x * b.w;
      const py = b.y + p.y * b.h;
      const isTeamA = p.team === "A";

      ctx.fillStyle = isTeamA ? "rgba(34, 197, 94, 0.25)" : "rgba(255, 255, 255, 0.22)";
      ctx.beginPath();
      ctx.arc(px, py, 14, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = isTeamA ? "#22C55E" : "#F8FAFC";
      ctx.strokeStyle = "#0D1219";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(px, py, 9.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "#0D1219";
      ctx.font = "700 9px 'JetBrains Mono', monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(p.id, px, py + 0.5);
    }
    ctx.textBaseline = "alphabetic";
  }

  drawBallAndTrail(ctx, b, ball) {
    if (!ball) return;
    const bx = b.x + ball.x * b.w;
    const by = b.y + ball.y * b.h;

    this.trail.push({ x: bx, y: by, time: performance.now() });
    if (this.trail.length > this.maxTrail) {
      this.trail.shift();
    }

    // 1. Draw glowing projectile trail
    if (this.trail.length > 1) {
      for (let i = 1; i < this.trail.length; i++) {
        const p0 = this.trail[i - 1];
        const p1 = this.trail[i];
        const progress = i / this.trail.length;
        const alpha = progress * 0.9;

        ctx.strokeStyle = `rgba(204, 255, 0, ${alpha})`;
        ctx.lineWidth = 1.5 + progress * 3.5;
        ctx.beginPath();
        ctx.moveTo(p0.x, p0.y);
        ctx.lineTo(p1.x, p1.y);
        ctx.stroke();

        // Motion blur glow
        if (i === this.trail.length - 1) {
          ctx.strokeStyle = `rgba(255, 255, 255, 0.7)`;
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.moveTo(p0.x, p0.y);
          ctx.lineTo(p1.x, p1.y);
          ctx.stroke();
        }
      }
    }

    // 2. Animated contact shockwave ripple (every few frames or based on speed)
    const t = performance.now() / 1000;
    const waveRadius = 10 + (Math.sin(t * 8) * 0.5 + 0.5) * 12;
    const waveAlpha = Math.max(0, 0.7 - (waveRadius / 22) * 0.6);
    ctx.strokeStyle = `rgba(204, 255, 0, ${waveAlpha})`;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(bx, by, waveRadius, 0, Math.PI * 2);
    ctx.stroke();

    // 3. Pro Optic Yellow Pickleball Body
    ctx.beginPath();
    ctx.arc(bx, by, 6.5, 0, Math.PI * 2);
    ctx.fillStyle = "#CCFF00";
    ctx.fill();
    ctx.strokeStyle = "#0A111A";
    ctx.lineWidth = 1.8;
    ctx.stroke();

    // 4. Perforated ball dots
    ctx.fillStyle = "#99CC00";
    const dotOffsets = [[-2, -2], [2, -2], [0, 2], [-3, 1], [3, 1]];
    dotOffsets.forEach(([dx, dy]) => {
      ctx.beginPath();
      ctx.arc(bx + dx, by + dy, 0.9, 0, Math.PI * 2);
      ctx.fill();
    });

    // 5. Live Ball Velocity Pill
    if (ball.speedMph) {
      ctx.save();
      ctx.font = "bold 9px 'JetBrains Mono', monospace";
      const spdText = `${ball.speedMph} MPH`;
      const tw = ctx.measureText(spdText).width;
      const tagX = bx + 10;
      const tagY = by - 8;

      ctx.fillStyle = "rgba(10, 17, 26, 0.88)";
      ctx.beginPath();
      ctx.roundRect(tagX, tagY - 9, tw + 8, 14, 4);
      ctx.fill();
      ctx.strokeStyle = "#CCFF00";
      ctx.lineWidth = 1.0;
      ctx.stroke();

      ctx.fillStyle = "#CCFF00";
      ctx.fillText(spdText, tagX + 4, tagY + 1.5);
      ctx.restore();
    }
  }

  drawHUD(ctx, telemetry) {
    ctx.save();
    ctx.fillStyle = "rgba(13, 18, 25, 0.85)";
    ctx.beginPath();
    ctx.roundRect(12, 12, 168, 24, 6);
    ctx.fill();

    ctx.fillStyle = "#DFFF00";
    ctx.font = "700 10px 'JetBrains Mono', monospace";
    ctx.textAlign = "left";
    ctx.fillText(
      `${telemetry.ball.shotType} • ${telemetry.ball.speedMph} MPH`,
      20,
      27
    );
    ctx.restore();
  }
}

/**
 * Helper to draw a realistic 40-hole perforated pickleball at (x, y)
 */
function drawPerforatedBall(ctx, x, y, r, rotAngle) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(rotAngle);

  const glow = ctx.createRadialGradient(0, 0, r * 0.2, 0, 0, r * 2.1);
  glow.addColorStop(0, "rgba(212, 247, 0, 0.48)");
  glow.addColorStop(1, "rgba(212, 247, 0, 0)");
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(0, 0, r * 2.1, 0, Math.PI * 2);
  ctx.fill();

  const sphereGrad = ctx.createRadialGradient(-r * 0.3, -r * 0.3, r * 0.1, 0, 0, r);
  sphereGrad.addColorStop(0, "#F5FF78");
  sphereGrad.addColorStop(0.65, "#DFFF00");
  sphereGrad.addColorStop(1, "#9FB800");
  ctx.fillStyle = sphereGrad;
  ctx.strokeStyle = "rgba(14, 17, 22, 0.5)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(0, 0, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  const holes = [
    { dx: 0, dy: 0, hr: r * 0.16 },
    { dx: r * 0.48, dy: 0, hr: r * 0.14 },
    { dx: -r * 0.48, dy: 0, hr: r * 0.14 },
    { dx: 0, dy: r * 0.48, hr: r * 0.14 },
    { dx: 0, dy: -r * 0.48, hr: r * 0.14 },
    { dx: r * 0.34, dy: r * 0.34, hr: r * 0.12 },
    { dx: -r * 0.34, dy: -r * 0.34, hr: r * 0.12 },
    { dx: r * 0.34, dy: -r * 0.34, hr: r * 0.12 },
    { dx: -r * 0.34, dy: r * 0.34, hr: r * 0.12 }
  ];
  ctx.fillStyle = "rgba(14, 17, 22, 0.34)";
  for (const h of holes) {
    ctx.beginPath();
    ctx.arc(h.dx, h.dy, h.hr, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

/**
 * Full-Screen Live Pickleball Background Animation
 */
export class PickleballBgAnimator {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.ripples = [];
    this.balls = [
      { phase: 0.0, speed: 0.0052, radius: 16, angle: 0, trail: [], lastBounceSide: -1 },
      { phase: 0.52, speed: 0.0040, radius: 12, angle: 1.4, trail: [], lastBounceSide: -1 }
    ];

    window.addEventListener("resize", () => this.resize());
    this.resize();
    this.animate();
  }

  resize() {
    if (!this.canvas) return;
    const dpr = window.devicePixelRatio || 1;
    this.width = window.innerWidth;
    this.height = window.innerHeight;
    this.canvas.width = this.width * dpr;
    this.canvas.height = this.height * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  drawPerspectiveCourt(ctx, w, h, t) {
    ctx.save();
    const cx = w * 0.52;
    const cy = h * 0.54;
    const topW = Math.min(w * 0.62, 820);
    const botW = Math.min(w * 0.92, 1280);
    const courtH = Math.min(h * 0.72, 620);

    const topY = cy - courtH * 0.48;
    const botY = cy + courtH * 0.52;
    const netY = (topY + botY) * 0.5;
    const nvzOffset = courtH * 0.16;
    const lerpW = (yFrac) => topW + (botW - topW) * yFrac;

    const isDark = document.documentElement.getAttribute("data-theme") === "dark";
    ctx.strokeStyle = isDark ? "rgba(56, 189, 248, 0.16)" : "rgba(27, 86, 138, 0.12)";
    ctx.lineWidth = 1.6;

    ctx.beginPath();
    ctx.moveTo(cx - topW / 2, topY);
    ctx.lineTo(cx + topW / 2, topY);
    ctx.lineTo(cx + botW / 2, botY);
    ctx.lineTo(cx - botW / 2, botY);
    ctx.closePath();
    ctx.stroke();

    const nvzTopFrac = (netY - nvzOffset - topY) / courtH;
    const nvzBotFrac = (netY + nvzOffset - topY) / courtH;
    const wTopNVZ = lerpW(nvzTopFrac);
    const wBotNVZ = lerpW(nvzBotFrac);

    ctx.fillStyle = isDark ? "rgba(56, 189, 248, 0.05)" : "rgba(27, 86, 138, 0.04)";
    ctx.beginPath();
    ctx.moveTo(cx - wTopNVZ / 2, netY - nvzOffset);
    ctx.lineTo(cx + wTopNVZ / 2, netY - nvzOffset);
    ctx.lineTo(cx + wBotNVZ / 2, netY + nvzOffset);
    ctx.lineTo(cx - wBotNVZ / 2, netY + nvzOffset);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(cx, topY);
    ctx.lineTo(cx, netY - nvzOffset);
    ctx.moveTo(cx, netY + nvzOffset);
    ctx.lineTo(cx, botY);
    ctx.stroke();

    const wNet = lerpW(0.5);
    ctx.strokeStyle = isDark ? "rgba(204, 255, 0, 0.28)" : "rgba(255, 77, 0, 0.24)";
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 6]);
    ctx.lineDashOffset = -t * 18;
    ctx.beginPath();
    ctx.moveTo(cx - wNet / 2 - 18, netY);
    ctx.lineTo(cx + wNet / 2 + 18, netY);
    ctx.stroke();
    ctx.restore();
  }

  animate() {
    const step = (now) => {
      const t = now * 0.001;
      const ctx = this.ctx;
      const w = this.width;
      const h = this.height;

      ctx.clearRect(0, 0, w, h);
      this.drawPerspectiveCourt(ctx, w, h, t);

      for (let i = this.ripples.length - 1; i >= 0; i--) {
        const rp = this.ripples[i];
        rp.radius += 1.4;
        rp.alpha -= 0.014;
        if (rp.alpha <= 0) {
          this.ripples.splice(i, 1);
          continue;
        }
        ctx.save();
        ctx.strokeStyle = `rgba(255, 77, 0, ${rp.alpha})`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.ellipse(rp.x, rp.y, rp.radius * 1.4, rp.radius * 0.65, 0, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }

      for (let idx = 0; idx < this.balls.length; idx++) {
        const b = this.balls[idx];
        b.phase = (b.phase + b.speed) % 1;
        b.angle += 0.06;

        const pingPong = Math.sin(b.phase * Math.PI * 2);
        const arcBounce = Math.abs(Math.cos(b.phase * Math.PI * 4));
        const bx = w * 0.5 + pingPong * (w * (idx === 0 ? 0.34 : 0.25));
        const baseGroundY = h * (idx === 0 ? 0.62 : 0.38) + Math.cos(b.phase * Math.PI * 2) * (h * 0.14);
        const by = baseGroundY - arcBounce * 115;

        const bounceBucket = Math.floor(b.phase * 4);
        if (arcBounce < 0.08 && b.lastBounceSide !== bounceBucket) {
          b.lastBounceSide = bounceBucket;
          this.ripples.push({ x: bx, y: baseGroundY, radius: 6, alpha: 0.65 });
        }

        b.trail.push({ x: bx, y: by });
        if (b.trail.length > 26) b.trail.shift();

        if (b.trail.length > 2) {
          for (let i = 1; i < b.trail.length; i++) {
            const p0 = b.trail[i - 1];
            const p1 = b.trail[i];
            const frac = i / b.trail.length;
            ctx.strokeStyle = `rgba(255, 77, 0, ${frac * 0.32})`;
            ctx.lineWidth = 1 + frac * 3.2;
            ctx.beginPath();
            ctx.moveTo(p0.x, p0.y);
            ctx.lineTo(p1.x, p1.y);
            ctx.stroke();
          }
        }

        drawPerforatedBall(ctx, bx, by, b.radius, b.angle);
      }

      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
}

/**
 * Dedicated Sign Up / Login Page Animation:
 * Two Carbon-Fiber Pickleball Paddles in a High-Speed Kitchen Volley Duel
 * with Impact Shockwaves, Spin RPM, and Ball Velocity Vectors.
 */
export class AuthPaddleDuelAnimator {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.shocks = [];
    this.trail = [];
    this.ballAngle = 0;
    this.lastHitDir = 0;
    this.currentSpeedMph = "44.8";
    this.currentSpinRpm = "1,860";

    this._resizeObserver = new ResizeObserver(() => this.resize());
    if (this.canvas.parentElement) {
      this._resizeObserver.observe(this.canvas.parentElement);
    }
    this.resize();
    this.animate();
  }

  resize() {
    const parent = this.canvas.parentElement;
    if (!parent) return;
    const dpr = window.devicePixelRatio || 1;
    this.width = Math.max(280, parent.clientWidth);
    this.height = Math.max(360, parent.clientHeight);
    this.canvas.width = this.width * dpr;
    this.canvas.height = this.height * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  drawCarbonPaddle(ctx, x, y, tiltRad, accentColor, label) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(tiltRad);

    // Paddle Handle
    ctx.fillStyle = "#1E293B";
    ctx.strokeStyle = "#475569";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(-7, 36, 14, 38, 5);
    ctx.fill();
    ctx.stroke();

    // Grip Tape Ridges
    ctx.strokeStyle = accentColor;
    ctx.lineWidth = 1.5;
    for (let gy = 44; gy <= 66; gy += 7) {
      ctx.beginPath();
      ctx.moveTo(-6, gy);
      ctx.lineTo(6, gy);
      ctx.stroke();
    }

    // Carbon-Fiber Paddle Face
    ctx.fillStyle = "#0F172A";
    ctx.strokeStyle = accentColor;
    ctx.lineWidth = 3.2;
    ctx.beginPath();
    ctx.roundRect(-34, -44, 68, 82, 22);
    ctx.fill();
    ctx.stroke();

    // Honeycomb / Carbon Weave Texture Lines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
    ctx.lineWidth = 1;
    for (let i = -24; i <= 24; i += 12) {
      ctx.beginPath();
      ctx.moveTo(i, -36);
      ctx.lineTo(i, 30);
      ctx.stroke();
    }
    for (let j = -30; j <= 24; j += 12) {
      ctx.beginPath();
      ctx.moveTo(-26, j);
      ctx.lineTo(26, j);
      ctx.stroke();
    }

    // Center Brand Emblem on Paddle Face
    ctx.fillStyle = accentColor;
    ctx.font = "800 10px 'Inter', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(label, 0, -4);

    ctx.restore();
  }

  animate() {
    const step = (now) => {
      const t = now * 0.0018;
      const ctx = this.ctx;
      const w = this.width;
      const h = this.height;

      if (w > 0 && h > 0) {
        ctx.clearRect(0, 0, w, h);

        // Stadium Dark Court Background
        const bgGrad = ctx.createLinearGradient(0, 0, w, h);
        bgGrad.addColorStop(0, "#0B1320");
        bgGrad.addColorStop(0.5, "#112238");
        bgGrad.addColorStop(1, "#0B1320");
        ctx.fillStyle = bgGrad;
        ctx.fillRect(0, 0, w, h);

        // Court Kitchen Zone & Center Net
        const midX = w * 0.5;
        const nvzW = w * 0.36;
        ctx.fillStyle = "rgba(27, 86, 138, 0.25)";
        ctx.fillRect(midX - nvzW / 2, 28, nvzW, h - 56);

        ctx.strokeStyle = "rgba(255, 255, 255, 0.18)";
        ctx.lineWidth = 1.5;
        ctx.strokeRect(24, 28, w - 48, h - 56);

        // Net Line
        ctx.save();
        ctx.strokeStyle = "#DFFF00";
        ctx.lineWidth = 2.5;
        ctx.setLineDash([6, 5]);
        ctx.beginPath();
        ctx.moveTo(midX, 18);
        ctx.lineTo(midX, h - 18);
        ctx.stroke();
        ctx.restore();

        // Paddle-to-Paddle Rally Kinematics
        const leftX = w * 0.17;
        const rightX = w * 0.83;
        const rallySin = Math.sin(t * Math.PI);
        const dir = Math.cos(t * Math.PI) >= 0 ? 1 : -1;

        const bx = midX + rallySin * (rightX - leftX - 46) * 0.5;
        const by = h * 0.5 + Math.sin(t * 2.3) * (h * 0.22) - Math.abs(Math.cos(t * Math.PI)) * 28;

        // Detect paddle impact inflection
        if (dir !== this.lastHitDir) {
          this.lastHitDir = dir;
          const hitX = dir === 1 ? leftX + 22 : rightX - 22;
          const hitColor = dir === 1 ? "#FF4D00" : "#DFFF00";
          this.shocks.push({ x: hitX, y: by, r: 10, alpha: 0.95, color: hitColor });
          this.currentSpeedMph = (41.5 + Math.random() * 11.2).toFixed(1);
          this.currentSpinRpm = (1650 + Math.floor(Math.random() * 520)).toLocaleString();
        }

        // Left & Right Paddle Positions + Swing Angles
        const leftPaddleY = h * 0.5 + Math.sin(t * 2.3 - 0.2) * (h * 0.20);
        const rightPaddleY = h * 0.5 + Math.sin(t * 2.3 + 0.2) * (h * 0.20);
        const leftSwing = -0.18 + Math.max(0, -rallySin) * 0.38;
        const rightSwing = 0.18 - Math.max(0, rallySin) * 0.38;

        // Draw Shockwave Rings on Paddle Hit
        for (let i = this.shocks.length - 1; i >= 0; i--) {
          const s = this.shocks[i];
          s.r += 2.2;
          s.alpha -= 0.028;
          if (s.alpha <= 0) {
            this.shocks.splice(i, 1);
            continue;
          }
          ctx.save();
          ctx.strokeStyle = s.color;
          ctx.globalAlpha = s.alpha;
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
          ctx.stroke();
          ctx.restore();
        }

        // Draw Paddles
        this.drawCarbonPaddle(ctx, leftX, leftPaddleY, leftSwing, "#FF4D00", "PRO");
        this.drawCarbonPaddle(ctx, rightX, rightPaddleY, rightSwing, "#DFFF00", "AI");

        // Ball Trail
        this.trail.push({ x: bx, y: by });
        if (this.trail.length > 22) this.trail.shift();

        for (let i = 1; i < this.trail.length; i++) {
          const p0 = this.trail[i - 1];
          const p1 = this.trail[i];
          const frac = i / this.trail.length;
          ctx.strokeStyle = `rgba(223, 255, 0, ${frac * 0.75})`;
          ctx.lineWidth = 1.5 + frac * 4;
          ctx.beginPath();
          ctx.moveTo(p0.x, p0.y);
          ctx.lineTo(p1.x, p1.y);
          ctx.stroke();
        }

        // Spinning 40-Hole Pickleball
        this.ballAngle += 0.12 * dir;
        drawPerforatedBall(ctx, bx, by, 14, this.ballAngle);

        // Live Telemetry HUD Pill at Top
        ctx.fillStyle = "rgba(14, 17, 22, 0.82)";
        ctx.beginPath();
        ctx.roundRect(24, 16, w - 48, 34, 999);
        ctx.fill();

        ctx.fillStyle = "#DFFF00";
        ctx.font = "700 11px 'JetBrains Mono', monospace";
        ctx.textAlign = "center";
        ctx.fillText(
          `LIVE KITCHEN VOLLEY DUEL • ${this.currentSpeedMph} MPH • SPIN ${this.currentSpinRpm} RPM`,
          midX,
          37
        );
      }

      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
}
