/**
 * court_visualizer.js — 60 FPS HTML5 Canvas Background & VAR Visualizer
 * Handcrafted Pro-Sports Palette: Yellow-Green (#BAFF39) • Dim Grey (#6E6E6E) • Crisp White (#FFFFFF)
 */

export class BackgroundCourtAnimator {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.balls = [
      { x: 0, y: 0, phase: 0.1, speed: 0.005, trail: [] },
      { x: 0, y: 0, phase: 0.65, speed: 0.0042, trail: [] },
    ];
    this.ripples = [];

    window.addEventListener("resize", () => this.resize());
    this.resize();
    this.animate();
  }

  resize() {
    this.width = window.innerWidth;
    this.height = window.innerHeight;
    this.canvas.width = this.width;
    this.canvas.height = this.height;
  }

  drawCourt(ctx, w, h, t) {
    const cx = w * 0.5;
    const cy = h * 0.55;
    const courtW = Math.min(w * 0.85, 1100);
    const courtH = Math.min(h * 0.7, 650);

    ctx.save();
    // Dim Grey (#6E6E6E) Outer Court Perimeter
    ctx.strokeStyle = "rgba(110, 110, 110, 0.22)";
    ctx.lineWidth = 1.5;

    ctx.beginPath();
    ctx.rect(cx - courtW * 0.5, cy - courtH * 0.5, courtW, courtH);
    ctx.stroke();

    // 7-ft Kitchen Non-Volley Zone lines (7/44 ratio)
    const kitchenDist = courtH * (7 / 44);
    ctx.strokeStyle = "rgba(186, 255, 57, 0.25)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(cx - courtW * 0.5, cy - kitchenDist);
    ctx.lineTo(cx + courtW * 0.5, cy - kitchenDist);
    ctx.moveTo(cx - courtW * 0.5, cy + kitchenDist);
    ctx.lineTo(cx + courtW * 0.5, cy + kitchenDist);
    ctx.stroke();

    // Kitchen subtle tint
    ctx.fillStyle = "rgba(186, 255, 57, 0.015)";
    ctx.fillRect(cx - courtW * 0.5, cy - kitchenDist, courtW, kitchenDist * 2);

    // Center Net Line (Crisp White #FFFFFF with dashed pulse)
    ctx.strokeStyle = "#FFFFFF";
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 8]);
    ctx.lineDashOffset = -t * 15;
    ctx.beginPath();
    ctx.moveTo(cx - courtW * 0.5 - 20, cy);
    ctx.lineTo(cx + courtW * 0.5 + 20, cy);
    ctx.stroke();
    ctx.setLineDash([]);

    // Center service line in Dim Grey (#6E6E6E)
    ctx.strokeStyle = "rgba(110, 110, 110, 0.35)";
    ctx.beginPath();
    ctx.moveTo(cx, cy - courtH * 0.5);
    ctx.lineTo(cx, cy - kitchenDist);
    ctx.moveTo(cx, cy + kitchenDist);
    ctx.lineTo(cx, cy + courtH * 0.5);
    ctx.stroke();

    ctx.restore();
  }

  animate() {
    const step = (timestamp) => {
      const t = timestamp * 0.001;
      const ctx = this.ctx;
      const w = this.width;
      const h = this.height;

      ctx.clearRect(0, 0, w, h);
      this.drawCourt(ctx, w, h, t);

      // Render expanding ground ripples in Yellow-Green (#BAFF39)
      for (let i = this.ripples.length - 1; i >= 0; i--) {
        const rp = this.ripples[i];
        rp.radius += 1.8;
        rp.alpha -= 0.018;
        if (rp.alpha <= 0) {
          this.ripples.splice(i, 1);
          continue;
        }
        ctx.save();
        ctx.strokeStyle = `rgba(186, 255, 57, ${rp.alpha})`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.ellipse(rp.x, rp.y, rp.radius * 1.5, rp.radius * 0.6, 0, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }

      // Render live animated pickleballs with Yellow-Green (#BAFF39) trails
      this.balls.forEach((b) => {
        b.phase = (b.phase + b.speed) % 1;
        const courtW = Math.min(w * 0.8, 1000);
        const courtH = Math.min(h * 0.65, 580);
        const bx = w * 0.5 + Math.sin(b.phase * Math.PI * 2) * (courtW * 0.42);
        const groundY = h * 0.55 + Math.cos(b.phase * Math.PI * 2) * (courtH * 0.38);
        const arc = Math.abs(Math.sin(b.phase * Math.PI * 4)) * 75;
        const by = groundY - arc;

        // Bounce impact shockwave
        if (arc < 5 && (!b.lastBounce || Math.abs(b.phase - b.lastBounce) > 0.15)) {
          b.lastBounce = b.phase;
          this.ripples.push({ x: bx, y: groundY, radius: 4, alpha: 0.65 });
        }

        b.trail.push({ x: bx, y: by });
        if (b.trail.length > 20) b.trail.shift();

        // Trail in Yellow-Green (#BAFF39)
        if (b.trail.length > 1) {
          ctx.save();
          for (let j = 1; j < b.trail.length; j++) {
            const frac = j / b.trail.length;
            ctx.strokeStyle = `rgba(186, 255, 57, ${frac * 0.4})`;
            ctx.lineWidth = frac * 3.5;
            ctx.beginPath();
            ctx.moveTo(b.trail[j - 1].x, b.trail[j - 1].y);
            ctx.lineTo(b.trail[j].x, b.trail[j].y);
            ctx.stroke();
          }
          ctx.restore();
        }

        // Optic Yellow-Green Pickleball
        ctx.save();
        ctx.fillStyle = "#BAFF39";
        ctx.shadowColor = "#BAFF39";
        ctx.shadowBlur = 12;
        ctx.beginPath();
        ctx.arc(bx, by, 6.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      });

      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
}

/**
 * Interactive Electronic Line Calling (ELC) Hawk-Eye Zoom Station
 * Handcrafted palette: Yellow-Green (#BAFF39) • Dim Grey (#6E6E6E) • White (#FFFFFF)
 */
export class HawkEyeVARVisualizer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.activeCall = null;
  }

  render(callData) {
    this.activeCall = callData;
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    ctx.clearRect(0, 0, w, h);

    // Deep Tournament Surface
    ctx.fillStyle = "#12151A";
    ctx.fillRect(0, 0, w, h);

    // Architectural Grid Lines in Dim Grey (#6E6E6E)
    ctx.strokeStyle = "rgba(110, 110, 110, 0.18)";
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 32) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }

    // Court Line (2-inch regulation white court line in Crisp White #FFFFFF)
    const lineX = w * 0.55;
    ctx.fillStyle = "#FFFFFF";
    ctx.shadowColor = "rgba(255, 255, 255, 0.4)";
    ctx.shadowBlur = 10;
    ctx.fillRect(lineX - 16, 0, 32, h);
    ctx.shadowBlur = 0;

    // Technical Line Label in Dim Grey (#6E6E6E)
    ctx.fillStyle = "#6E6E6E";
    ctx.font = "800 11px 'JetBrains Mono', monospace";
    ctx.fillText("USAP REGULATION LINE (2 IN / 50 MM)", lineX + 24, 28);

    // Ball Impact Ellipse
    const isIn = callData.verdict === "IN";
    const ballColor = isIn ? "#BAFF39" : "#EF4444";
    const ballX = lineX - 16 + (isIn ? 8 : -28) + (callData.margin_mm * 0.4);
    const ballY = h * 0.52;

    // Compression Shadow & Glow
    ctx.fillStyle = ballColor;
    ctx.shadowColor = ballColor;
    ctx.shadowBlur = 24;
    ctx.beginPath();
    ctx.ellipse(ballX, ballY, 32, 22, 0.1, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;

    // Ball core
    ctx.fillStyle = "#FFFFFF";
    ctx.beginPath();
    ctx.ellipse(ballX, ballY, 24, 16, 0.1, 0, Math.PI * 2);
    ctx.fill();

    // Measurement Dimension Caliper in Dim Grey (#6E6E6E) with Accent-YG endpoints
    ctx.strokeStyle = "#6E6E6E";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(ballX, ballY + 45);
    ctx.lineTo(lineX - 16, ballY + 45);
    ctx.stroke();
    ctx.setLineDash([]);

    // Caliper end marks
    ctx.strokeStyle = ballColor;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(ballX, ballY + 40);
    ctx.lineTo(ballX, ballY + 50);
    ctx.moveTo(lineX - 16, ballY + 40);
    ctx.lineTo(lineX - 16, ballY + 50);
    ctx.stroke();

    // Distance Pill in #0D0F12 with Dim Grey border
    const tagW = 120;
    const tagH = 28;
    ctx.fillStyle = "#0D0F12";
    ctx.strokeStyle = ballColor;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.roundRect((ballX + lineX) * 0.5 - tagW * 0.5, ballY + 54, tagW, tagH, 6);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = ballColor;
    ctx.font = "800 12px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const sign = callData.margin_mm > 0 ? "+" : "";
    ctx.fillText(`${callData.verdict} (${sign}${callData.margin_mm} mm)`, (ballX + lineX) * 0.5, ballY + 68);

    // Callout Badge at Top-Left
    ctx.fillStyle = "rgba(13, 15, 18, 0.94)";
    ctx.strokeStyle = "#6E6E6E";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(16, 16, 190, 52, 6);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#6E6E6E";
    ctx.font = "700 10px 'JetBrains Mono', monospace";
    ctx.textAlign = "left";
    ctx.fillText("HAWK-EYE SUB-PIXEL VAR", 26, 34);

    ctx.fillStyle = ballColor;
    ctx.font = "800 15px 'Outfit', sans-serif";
    ctx.fillText(`BOUNCE #${callData.bounce_idx}: ${callData.verdict}`, 26, 54);
  }
}
