/**
 * world.js — Bruno Simon-Inspired 3D Playable Open-World Pickleball Complex
 *
 * Self-contained 60 FPS 3D Perspective Projection & Vehicle Physics Engine.
 * Zero external network dependencies — guaranteed instant load & 60 FPS.
 */

import { AchievementManager } from "./achievements.js";
import { WORLD_LANDMARKS, LandmarkModalManager } from "./landmarks.js";

class PickleballOpenWorld {
  constructor() {
    this.canvas = document.getElementById("world-canvas");
    this.ctx = this.canvas.getContext("2d");
    this.minimapCanvas = document.getElementById("minimap-canvas");
    this.minimapCtx = this.minimapCanvas ? this.minimapCanvas.getContext("2d") : null;

    this.achievements = new AchievementManager();
    this.modals = new LandmarkModalManager(this.achievements);

    this.isStarted = false;
    this.isNightMode = false;
    this.time = 0;

    // Rover Physics State
    this.car = {
      x: 0,
      y: 0,
      z: 20,
      vx: 0,
      vz: 0,
      vy: 0,
      speed: 0,
      angle: Math.PI, // Facing North (-Z toward Center Court)
      angularVel: 0,
      steering: 0,
      wheelSpin: 0,
      tiltRoll: 0,
      tiltPitch: 0,
      onGround: true,
      totalDistance: 0,
      boosting: false,
    };

    // Chase Camera State
    this.cam = {
      x: 0,
      y: 9.5,
      z: 34,
      lookX: 0,
      lookY: 1.2,
      lookZ: 14,
      fov: 520,
    };

    // Input State
    this.keys = {
      forward: false,
      backward: false,
      left: false,
      right: false,
      boost: false,
      jump: false,
    };

    // Environment Props (Trees, Ramps, Bleachers, Particles, Tire Marks)
    this.trees = [];
    this.ramps = [
      { x: -18, z: 4, w: 4.5, l: 7, h: 2.2, dir: -1 },
      { x: 18, z: -4, w: 4.5, l: 7, h: 2.2, dir: 1 },
    ];
    this.particles = [];
    this.tireMarks = [];
    this.hoveredLandmark = null;

    this.initEnvironment();
    this.bindEvents();
    this.resize();
    requestAnimationFrame((ts) => this.loop(ts));
  }

  initEnvironment() {
    // Generate decorative stadium perimeter trees
    const treeRing = [
      [-52, -45], [-38, -50], [-20, -52], [20, -52], [38, -50], [52, -45],
      [-56, -20], [-56, 0], [-56, 22], [-50, 44], [-28, 52], [28, 52],
      [50, 44], [56, 22], [56, 0], [56, -20],
      [-14, -44], [14, -44], // Framing the secret Golden Paddle in North Woods
    ];
    treeRing.forEach(([tx, tz], idx) => {
      this.trees.push({
        x: tx,
        z: tz,
        scale: 0.85 + (idx % 3) * 0.22,
      });
    });
  }

  bindEvents() {
    window.addEventListener("resize", () => this.resize());

    // Start overlay button
    const startOverlay = document.getElementById("start-overlay");
    const btnStart = document.getElementById("btn-start-drive");
    const startDrive = () => {
      this.isStarted = true;
      startOverlay?.classList.add("hidden");
    };
    btnStart?.addEventListener("click", startDrive);

    // Top HUD Controls
    document.getElementById("btn-night-mode")?.addEventListener("click", () => {
      this.toggleNightMode();
    });
    document.getElementById("btn-reset-car")?.addEventListener("click", () => {
      this.resetCar();
    });

    // Achievements Modal
    const achModal = document.getElementById("achievements-modal");
    document.getElementById("btn-open-achievements")?.addEventListener("click", () => {
      this.achievements.updateUI();
      achModal?.classList.remove("hidden");
    });
    document.getElementById("btn-close-achievements")?.addEventListener("click", () => {
      achModal?.classList.add("hidden");
    });

    // Zone Prompt Click
    document.getElementById("zone-prompt")?.addEventListener("click", () => {
      if (this.hoveredLandmark) {
        this.modals.open(this.hoveredLandmark);
      }
    });

    // Keyboard Controls
    window.addEventListener("keydown", (e) => {
      const code = e.code;
      if (
        !this.isStarted &&
        ["KeyW", "KeyA", "KeyS", "KeyD", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Enter", "Space"].includes(code)
      ) {
        startDrive();
      }

      if (code === "KeyW" || code === "ArrowUp") this.keys.forward = true;
      if (code === "KeyS" || code === "ArrowDown") this.keys.backward = true;
      if (code === "KeyA" || code === "ArrowLeft") this.keys.left = true;
      if (code === "KeyD" || code === "ArrowRight") this.keys.right = true;
      if (code === "ShiftLeft" || code === "ShiftRight") this.keys.boost = true;

      if (code === "Space") {
        this.triggerJump();
      }
      if (code === "KeyR") {
        this.resetCar();
      }
      if (code === "KeyN") {
        this.toggleNightMode();
      }
      if ((code === "Enter" || code === "KeyE") && this.hoveredLandmark) {
        if (document.getElementById("landmark-modal")?.classList.contains("hidden")) {
          this.modals.open(this.hoveredLandmark);
        } else {
          this.modals.close();
        }
      }
    });

    window.addEventListener("keyup", (e) => {
      const code = e.code;
      if (code === "KeyW" || code === "ArrowUp") this.keys.forward = false;
      if (code === "KeyS" || code === "ArrowDown") this.keys.backward = false;
      if (code === "KeyA" || code === "ArrowLeft") this.keys.left = false;
      if (code === "KeyD" || code === "ArrowRight") this.keys.right = false;
      if (code === "ShiftLeft" || code === "ShiftRight") this.keys.boost = false;
    });
  }

  toggleNightMode() {
    this.isNightMode = !this.isNightMode;
    const btn = document.getElementById("btn-night-mode");
    if (btn) {
      btn.textContent = this.isNightMode ? "☀️ Day Mode (N)" : "🌙 Night Mode (N)";
    }
    if (this.isNightMode) {
      this.achievements.unlock("night_owl");
    }
  }

  resetCar() {
    this.car.x = 0;
    this.car.y = 1.5;
    this.car.z = 20;
    this.car.vx = 0;
    this.car.vz = 0;
    this.car.vy = 0;
    this.car.speed = 0;
    this.car.angle = Math.PI;
  }

  triggerJump() {
    if (this.car.onGround) {
      this.car.vy = 13.5;
      this.car.onGround = false;
      this.achievements.unlock("atp_jump");
      // Spawn jump burst particles
      for (let i = 0; i < 12; i++) {
        this.particles.push({
          x: this.car.x + (Math.random() - 0.5) * 1.5,
          y: 0.2,
          z: this.car.z + (Math.random() - 0.5) * 1.5,
          vx: (Math.random() - 0.5) * 6,
          vy: 2 + Math.random() * 4,
          vz: (Math.random() - 0.5) * 6,
          life: 1.0,
          color: "#CCFF00",
          size: 4,
        });
      }
    }
  }

  resize() {
    this.width = window.innerWidth;
    this.height = window.innerHeight;
    this.canvas.width = this.width;
    this.canvas.height = this.height;
    this.cam.fov = Math.min(this.width, this.height) * 0.72;
  }

  /**
   * Project 3D world coordinates (x, y, z) into 2D canvas screen coordinates
   */
  project(wx, wy, wz) {
    // Translate relative to camera
    const dx = wx - this.cam.x;
    const dy = wy - this.cam.y;
    const dz = wz - this.cam.z;

    // Camera basis vectors looking at (lookX, lookY, lookZ)
    const fx = this.cam.lookX - this.cam.x;
    const fy = this.cam.lookY - this.cam.y;
    const fz = this.cam.lookZ - this.cam.z;
    const flen = Math.hypot(fx, fy, fz) || 1;
    const forwardX = fx / flen;
    const forwardY = fy / flen;
    const forwardZ = fz / flen;

    // Right vector = Forward x WorldUp(0,1,0) -> Wait: standard right-handed or screen X right
    // With camera at +Z looking toward -Z: forward = (0, 0, -1).
    // Screen Right (+X) should be World +X: cross(forward, up) = (0,0,-1) x (0,1,0) = (1, 0, 0) -> Wait:
    // fy*0 - fz*1 = -fz. If fz = -1, -fz = +1!
    let rx = -forwardZ;
    let ry = 0;
    let rz = forwardX;
    const rlen = Math.hypot(rx, rz) || 1;
    rx /= rlen;
    rz /= rlen;

    // Up vector = Right x Forward
    const ux = ry * forwardZ - rz * forwardY;
    const uy = rz * forwardX - rx * forwardZ;
    const uz = rx * forwardY - ry * forwardX;

    // Transform point into camera space
    const cx = dx * rx + dy * ry + dz * rz;
    const cy = dx * ux + dy * uy + dz * uz;
    const cz = dx * forwardX + dy * forwardY + dz * forwardZ;

    if (cz <= 0.8) {
      return null; // Behind near clip plane
    }

    const scale = this.cam.fov / cz;
    const sx = this.width * 0.5 + cx * scale;
    const sy = this.height * 0.5 - cy * scale;

    return { x: sx, y: sy, depth: cz, scale };
  }

  updatePhysics(dt) {
    const car = this.car;

    // Acceleration & Boost
    const maxSpeed = this.keys.boost ? 34.0 : 21.0;
    const accelRate = this.keys.boost ? 30.0 : 18.0;
    car.boosting = this.keys.boost && this.keys.forward;

    let throttle = 0;
    if (this.keys.forward) throttle += 1;
    if (this.keys.backward) throttle -= 0.75;

    // Steering
    let steerInput = 0;
    if (this.keys.left) steerInput -= 1;
    if (this.keys.right) steerInput += 1;

    car.steering += (steerInput - car.steering) * Math.min(1, dt * 10);

    // Forward direction vector: angle=PI means facing -Z (North)
    const dirX = Math.sin(car.angle);
    const dirZ = Math.cos(car.angle);

    if (throttle !== 0) {
      car.vx += dirX * throttle * accelRate * dt;
      car.vz += dirZ * throttle * accelRate * dt;
    }

    // Ground friction / drag
    const drag = Math.pow(0.18, dt);
    car.vx *= drag;
    car.vz *= drag;

    // Clamp max speed
    const currentSpeed = Math.hypot(car.vx, car.vz);
    if (currentSpeed > maxSpeed) {
      car.vx = (car.vx / currentSpeed) * maxSpeed;
      car.vz = (car.vz / currentSpeed) * maxSpeed;
    }

    // Signed speed along heading
    const forwardDot = car.vx * dirX + car.vz * dirZ;
    car.speed = forwardDot;

    // Turn rate proportional to speed
    if (Math.abs(car.speed) > 0.4) {
      const turnDir = car.speed >= 0 ? -1 : 1;
      car.angle += car.steering * turnDir * 2.45 * dt * Math.min(1, Math.abs(car.speed) / 8);
    }

    // Lateral grip (reduce sideways sliding while keeping fun drift)
    const rightX = Math.cos(car.angle);
    const rightZ = -Math.sin(car.angle);
    const lateralVel = car.vx * rightX + car.vz * rightZ;
    car.vx -= rightX * lateralVel * Math.min(1, dt * 6.5);
    car.vz -= rightZ * lateralVel * Math.min(1, dt * 6.5);

    // Update position
    car.x += car.vx * dt;
    car.z += car.vz * dt;

    // Check Ramps for Launch
    let groundHeight = 0;
    for (const r of this.ramps) {
      if (Math.abs(car.x - r.x) < r.w * 0.5 && Math.abs(car.z - r.z) < r.l * 0.5) {
        const progress = ((car.z - r.z) * r.dir + r.l * 0.5) / r.l;
        groundHeight = Math.max(0, Math.min(1, progress)) * r.h;
        if (progress > 0.85 && Math.abs(car.speed) > 10 && car.onGround) {
          car.vy = Math.abs(car.speed) * 0.55;
          car.onGround = false;
          this.achievements.unlock("atp_jump");
        }
      }
    }

    // Vertical Gravity & Suspension
    if (!car.onGround || car.y > groundHeight) {
      car.vy -= 28.0 * dt;
      car.y += car.vy * dt;
      if (car.y <= groundHeight) {
        car.y = groundHeight;
        car.vy = 0;
        car.onGround = true;
      }
    } else {
      car.y = groundHeight;
    }

    // World boundary clamp (Stadium perimeter fence)
    const limit = 54;
    if (car.x < -limit) { car.x = -limit; car.vx *= -0.4; }
    if (car.x > limit) { car.x = limit; car.vx *= -0.4; }
    if (car.z < -limit) { car.z = -limit; car.vz *= -0.4; }
    if (car.z > limit) { car.z = limit; car.vz *= -0.4; }

    // Visual wheel spin & chassis tilt
    car.wheelSpin += car.speed * dt * 1.8;
    car.tiltRoll += (car.steering * (car.speed / 20) * 0.18 - car.tiltRoll) * dt * 8;
    car.tiltPitch += (throttle * -0.08 - car.tiltPitch) * dt * 8;

    // Odometer & Speed Achievements
    const stepDist = Math.hypot(car.vx * dt, car.vz * dt);
    car.totalDistance += stepDist;
    if (car.totalDistance > 15) {
      this.achievements.unlock("first_drive");
    }

    const mph = Math.round(Math.abs(car.speed) * 1.45);
    if (mph >= 45) {
      this.achievements.unlock("speed_demon");
    }

    // Update Speedometer HUD
    const speedEl = document.getElementById("speed-val");
    const gearEl = document.getElementById("gear-val");
    if (speedEl) speedEl.textContent = String(mph);
    if (gearEl) {
      if (!car.onGround) gearEl.textContent = "AIR";
      else if (car.boosting && mph > 5) gearEl.textContent = "TURBO";
      else if (car.speed < -0.5) gearEl.textContent = "R";
      else if (mph === 0) gearEl.textContent = "P";
      else gearEl.textContent = "D";
    }

    // Spawn exhaust particles & tire marks
    if (Math.abs(car.speed) > 2 && car.onGround) {
      if (Math.random() < (car.boosting ? 0.9 : 0.35)) {
        this.particles.push({
          x: car.x - dirX * 1.6 + (Math.random() - 0.5) * 0.5,
          y: car.y + 0.3,
          z: car.z - dirZ * 1.6 + (Math.random() - 0.5) * 0.5,
          vx: -dirX * (2 + Math.random() * 3),
          vy: 1.2 + Math.random() * 1.5,
          vz: -dirZ * (2 + Math.random() * 3),
          life: 0.65,
          color: car.boosting ? "#FF5722" : "#CCFF00",
          size: car.boosting ? 5.5 : 3.2,
        });
      }

      if (Math.abs(lateralVel) > 2.2 || car.boosting) {
        this.tireMarks.push({
          x: car.x,
          z: car.z,
          angle: car.angle,
          alpha: 0.45,
        });
        if (this.tireMarks.length > 90) this.tireMarks.shift();
      }
    }

    // Update Particles
    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.z += p.vz * dt;
      p.life -= dt * 1.4;
      if (p.life <= 0) this.particles.splice(i, 1);
    }

    // Smooth Spring Chase Camera
    const targetCamX = car.x - dirX * 13.5;
    const targetCamZ = car.z - dirZ * 13.5;
    const targetCamY = car.y + 8.5 + Math.abs(car.speed) * 0.08;

    this.cam.x += (targetCamX - this.cam.x) * Math.min(1, dt * 6.5);
    this.cam.z += (targetCamZ - this.cam.z) * Math.min(1, dt * 6.5);
    this.cam.y += (targetCamY - this.cam.y) * Math.min(1, dt * 6.5);

    this.cam.lookX += (car.x + dirX * 4.5 - this.cam.lookX) * Math.min(1, dt * 10);
    this.cam.lookZ += (car.z + dirZ * 4.5 - this.cam.lookZ) * Math.min(1, dt * 10);
    this.cam.lookY += (car.y + 1.0 - this.cam.lookY) * Math.min(1, dt * 10);

    // Check Landmark Proximity
    this.checkLandmarks();
  }

  checkLandmarks() {
    let closest = null;
    let minDist = Infinity;

    for (const lm of WORLD_LANDMARKS) {
      const d = Math.hypot(this.car.x - lm.x, this.car.z - lm.z);
      if (d < lm.radius && d < minDist) {
        closest = lm;
        minDist = d;
      }
    }

    const promptEl = document.getElementById("zone-prompt");
    const nameEl = document.getElementById("zone-prompt-name");

    if (closest) {
      this.hoveredLandmark = closest;
      this.achievements.recordZoneVisit(closest.id);
      if (nameEl) nameEl.textContent = closest.name;
      promptEl?.classList.remove("hidden");
    } else {
      this.hoveredLandmark = null;
      promptEl?.classList.add("hidden");
    }
  }

  /**
   * Helper to draw a filled 3D polygon on the ground or in space
   */
  drawPoly3D(points3D, fillStyle, strokeStyle = null, lineWidth = 1.5) {
    const projected = [];
    for (const pt of points3D) {
      const p = this.project(pt[0], pt[1], pt[2]);
      if (!p) return false;
      projected.push(p);
    }

    const ctx = this.ctx;
    ctx.beginPath();
    ctx.moveTo(projected[0].x, projected[0].y);
    for (let i = 1; i < projected.length; i++) {
      ctx.lineTo(projected[i].x, projected[i].y);
    }
    ctx.closePath();

    if (fillStyle) {
      ctx.fillStyle = fillStyle;
      ctx.fill();
    }
    if (strokeStyle) {
      ctx.strokeStyle = strokeStyle;
      ctx.lineWidth = lineWidth;
      ctx.stroke();
    }
    return true;
  }

  renderWorld() {
    const ctx = this.ctx;

    // 1. Sky & Horizon Gradient
    const skyGrad = ctx.createLinearGradient(0, 0, 0, this.height * 0.55);
    if (this.isNightMode) {
      skyGrad.addColorStop(0, "#050811");
      skyGrad.addColorStop(1, "#0F172A");
    } else {
      skyGrad.addColorStop(0, "#0E294B");
      skyGrad.addColorStop(1, "#1E4E79");
    }
    ctx.fillStyle = skyGrad;
    ctx.fillRect(0, 0, this.width, this.height);

    // 2. Ground Plane Fill
    const groundGrad = ctx.createLinearGradient(0, this.height * 0.35, 0, this.height);
    if (this.isNightMode) {
      groundGrad.addColorStop(0, "#090F1B");
      groundGrad.addColorStop(1, "#111C2D");
    } else {
      groundGrad.addColorStop(0, "#14261D");
      groundGrad.addColorStop(1, "#1B3528");
    }
    ctx.fillStyle = groundGrad;
    ctx.fillRect(0, this.height * 0.32, this.width, this.height * 0.68);

    // 3. Stadium Ground Grid Lines
    const gridColor = this.isNightMode ? "rgba(56, 189, 248, 0.08)" : "rgba(204, 255, 0, 0.09)";
    for (let g = -55; g <= 55; g += 5) {
      this.drawPoly3D(
        [[g, 0, -55], [g, 0, 55]],
        null,
        gridColor,
        1
      );
      this.drawPoly3D(
        [[-55, 0, g], [55, 0, g]],
        null,
        gridColor,
        1
      );
    }

    // 4. Paved Stadium Roads Connecting Landmarks
    const roadFill = this.isNightMode ? "#162032" : "#243044";
    const roadBorder = this.isNightMode ? "#26354D" : "#384966";
    // North-South Main Boulevard
    this.drawPoly3D([[-5, 0.01, -46], [5, 0.01, -46], [5, 0.01, 44], [-5, 0.01, 44]], roadFill, roadBorder, 1.5);
    // Cross Avenues to Corner Landmarks
    this.drawPoly3D([[-40, 0.01, -26], [40, 0.01, -26], [40, 0.01, -18], [-40, 0.01, -18]], roadFill, roadBorder, 1.5);
    this.drawPoly3D([[-40, 0.01, 20], [40, 0.01, 20], [40, 0.01, 28], [-40, 0.01, 28]], roadFill, roadBorder, 1.5);

    // 5. Tire Marks
    for (const tm of this.tireMarks) {
      this.drawPoly3D(
        [
          [tm.x - 0.6, 0.03, tm.z - 0.6],
          [tm.x + 0.6, 0.03, tm.z + 0.6],
        ],
        null,
        `rgba(15, 23, 42, ${tm.alpha})`,
        3
      );
    }

    // 6. Center Court (Official USAP 20x44 ft scaled to world units: 11x22)
    this.renderCenterCourt();

    // 7. Jump Ramps
    for (const r of this.ramps) {
      const x0 = r.x - r.w * 0.5;
      const x1 = r.x + r.w * 0.5;
      const zLow = r.dir === 1 ? r.z - r.l * 0.5 : r.z + r.l * 0.5;
      const zHigh = r.dir === 1 ? r.z + r.l * 0.5 : r.z - r.l * 0.5;
      this.drawPoly3D(
        [
          [x0, 0.05, zLow],
          [x1, 0.05, zLow],
          [x1, r.h, zHigh],
          [x0, r.h, zHigh],
        ],
        "#FF5722",
        "#CCFF00",
        2
      );
    }

    // 8. Landmark Beacon Pads
    for (const lm of WORLD_LANDMARKS) {
      this.renderLandmarkPad(lm);
    }

    // 9. Depth-Sorted 3D Objects (Trees, Pillars, Animated Rally Ball, Player Rover, Particles)
    const renderables = [];

    // Trees
    for (const t of this.trees) {
      const p = this.project(t.x, 0, t.z);
      if (p) {
        renderables.push({
          depth: p.depth,
          draw: () => this.drawTree(t),
        });
      }
    }

    // Landmark 3D Monuments & Labels
    for (const lm of WORLD_LANDMARKS) {
      const p = this.project(lm.x, 2, lm.z);
      if (p) {
        renderables.push({
          depth: p.depth,
          draw: () => this.drawLandmarkMonument(lm, p),
        });
      }
    }

    // Live Animated Rally Pickleball over Center Court
    const ballZ = Math.sin(this.time * 2.6) * 8.5;
    const ballX = Math.cos(this.time * 1.3) * 3.2;
    const ballY = 1.4 + Math.abs(Math.cos(this.time * 2.6)) * 3.2;
    const ballProj = this.project(ballX, ballY, ballZ);
    if (ballProj) {
      renderables.push({
        depth: ballProj.depth,
        draw: () => {
          // Ball shadow on court
          const sh = this.project(ballX, 0.08, ballZ);
          if (sh) {
            ctx.fillStyle = "rgba(0,0,0,0.45)";
            ctx.beginPath();
            ctx.arc(sh.x, sh.y, Math.max(3, 0.45 * sh.scale), 0, Math.PI * 2);
            ctx.fill();
          }
          // Optic Yellow Pickleball
          ctx.fillStyle = "#CCFF00";
          ctx.shadowColor = "#CCFF00";
          ctx.shadowBlur = 12;
          ctx.beginPath();
          ctx.arc(ballProj.x, ballProj.y, Math.max(4, 0.42 * ballProj.scale), 0, Math.PI * 2);
          ctx.fill();
          ctx.shadowBlur = 0;
        },
      });
    }

    // Exhaust Particles
    for (const pt of this.particles) {
      const pp = this.project(pt.x, pt.y, pt.z);
      if (pp) {
        renderables.push({
          depth: pp.depth,
          draw: () => {
            ctx.fillStyle = pt.color;
            ctx.globalAlpha = Math.max(0, pt.life);
            ctx.beginPath();
            ctx.arc(pp.x, pp.y, Math.max(2, (pt.size * pp.scale) / 35), 0, Math.PI * 2);
            ctx.fill();
            ctx.globalAlpha = 1.0;
          },
        });
      }
    }

    // Player Pickleball Rover
    const carProj = this.project(this.car.x, this.car.y, this.car.z);
    if (carProj) {
      renderables.push({
        depth: carProj.depth,
        draw: () => this.drawPlayerRover(),
      });
    }

    // Sort back-to-front (Painter's Algorithm)
    renderables.sort((a, b) => b.depth - a.depth);
    for (const item of renderables) {
      item.draw();
    }
  }

  renderCenterCourt() {
    // Outer Pro Green Apron
    this.drawPoly3D(
      [[-9.5, 0.02, -16], [9.5, 0.02, -16], [9.5, 0.02, 16], [-9.5, 0.02, 16]],
      "#15803D",
      "#FFFFFF",
      2
    );

    // Inner USAP Tournament Blue Court (11 x 22 world units)
    this.drawPoly3D(
      [[-5.5, 0.04, -11], [5.5, 0.04, -11], [5.5, 0.04, 11], [-5.5, 0.04, 11]],
      "#1D4ED8",
      "#FFFFFF",
      2.5
    );

    // 7-ft Non-Volley Zone (Kitchen) in terracotta / coral tint
    this.drawPoly3D(
      [[-5.5, 0.05, -3.5], [5.5, 0.05, -3.5], [5.5, 0.05, 3.5], [-5.5, 0.05, 3.5]],
      "#0284C7",
      "#FFFFFF",
      2.2
    );

    // Center Service Lines
    this.drawPoly3D([[0, 0.06, -11], [0, 0.06, -3.5]], null, "#FFFFFF", 2);
    this.drawPoly3D([[0, 0.06, 3.5], [0, 0.06, 11]], null, "#FFFFFF", 2);

    // 3D Pickleball Net & Posts at Z = 0
    this.drawPoly3D(
      [[-6.0, 0, 0], [6.0, 0, 0], [6.0, 1.6, 0], [-6.0, 1.6, 0]],
      "rgba(15, 23, 42, 0.72)",
      "#FFFFFF",
      2.5
    );
  }

  renderLandmarkPad(lm) {
    const isHovered = this.hoveredLandmark && this.hoveredLandmark.id === lm.id;
    const pulse = 1 + Math.sin(this.time * 4) * 0.06;
    const r = lm.radius * 0.65 * pulse;

    // Octagonal glowing zone pad on ground
    const pts = [];
    const sides = 12;
    for (let i = 0; i < sides; i++) {
      const a = (i / sides) * Math.PI * 2;
      pts.push([lm.x + Math.cos(a) * r, 0.05, lm.z + Math.sin(a) * r]);
    }

    this.drawPoly3D(
      pts,
      isHovered ? "rgba(204, 255, 0, 0.28)" : "rgba(15, 23, 42, 0.55)",
      isHovered ? "#CCFF00" : lm.color,
      isHovered ? 3.5 : 2.2
    );
  }

  drawLandmarkMonument(lm, proj) {
    const ctx = this.ctx;
    const floatY = 2.8 + Math.sin(this.time * 3 + lm.x) * 0.35;
    const topProj = this.project(lm.x, floatY, lm.z);
    if (!topProj) return;

    // Draw glowing holo-pillar or Golden Paddle
    if (lm.id === "golden_egg") {
      ctx.save();
      ctx.translate(topProj.x, topProj.y);
      ctx.fillStyle = "#FFD700";
      ctx.shadowColor = "#FFD700";
      ctx.shadowBlur = 18;
      ctx.beginPath();
      ctx.arc(0, 0, Math.max(8, 1.5 * topProj.scale), 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // Floating 3D Billboard Sign
    const isNear = this.hoveredLandmark && this.hoveredLandmark.id === lm.id;
    const fontSize = Math.max(11, Math.min(18, Math.round(0.95 * topProj.scale)));
    ctx.save();
    ctx.font = `800 ${fontSize}px "Outfit", sans-serif`;
    const text = lm.name;
    const padX = 12;
    const tw = ctx.measureText(text).width + padX * 2;
    const th = fontSize + 14;

    ctx.fillStyle = isNear ? "#CCFF00" : "rgba(11, 17, 30, 0.88)";
    ctx.strokeStyle = lm.color;
    ctx.lineWidth = 2;

    const bx = topProj.x - tw * 0.5;
    const by = topProj.y - th * 0.5;
    ctx.beginPath();
    ctx.roundRect(bx, by, tw, th, 8);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = isNear ? "#080D1A" : "#FFFFFF";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, topProj.x, topProj.y);
    ctx.restore();
  }

  drawTree(t) {
    // Trunk
    const base = this.project(t.x, 0, t.z);
    const top = this.project(t.x, 4.8 * t.scale, t.z);
    if (!base || !top) return;

    const ctx = this.ctx;
    const w = Math.max(4, 1.8 * base.scale * t.scale);

    // Foliage cone
    ctx.fillStyle = this.isNightMode ? "#0F392B" : "#15803D";
    ctx.strokeStyle = "#CCFF00";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(top.x, top.y);
    ctx.lineTo(base.x - w, base.y);
    ctx.lineTo(base.x + w, base.y);
    ctx.closePath();
    ctx.fill();
  }

  drawPlayerRover() {
    const car = this.car;
    const cos = Math.cos(car.angle);
    const sin = Math.sin(car.angle);

    // Transform local car coordinates (lx, ly, lz) to world coordinates
    const toWorld = (lx, ly, lz) => {
      const wx = car.x + lx * cos + lz * sin;
      const wy = car.y + ly;
      const wz = car.z - lx * sin + lz * cos;
      return [wx, wy, wz];
    };

    // 1. Rover Ground Shadow
    this.drawPoly3D(
      [
        [-1.15, 0.04, -1.9].map((v, i) => toWorld(-1.15, 0, -1.9)[i]),
        toWorld(1.15, 0.04, -1.9),
        toWorld(1.15, 0.04, 1.9),
        toWorld(-1.15, 0.04, 1.9),
      ],
      "rgba(0, 0, 0, 0.52)"
    );

    // 2. Night Mode Headlight Cones
    if (this.isNightMode) {
      this.drawPoly3D(
        [
          toWorld(-0.8, 0.06, 1.8),
          toWorld(0.8, 0.06, 1.8),
          toWorld(3.8, 0.06, 14.0),
          toWorld(-3.8, 0.06, 14.0),
        ],
        "rgba(204, 255, 0, 0.18)"
      );
    }

    // 3. Lower Chassis Body (Optic Yellow & Carbon Slate)
    const bodyBottom = 0.35;
    const bodyTop = 1.05;
    // Rear Face
    this.drawPoly3D(
      [
        toWorld(-1.05, bodyBottom, -1.8),
        toWorld(1.05, bodyBottom, -1.8),
        toWorld(0.95, bodyTop, -1.7),
        toWorld(-0.95, bodyTop, -1.7),
      ],
      "#1E293B",
      "#CCFF00",
      2
    );
    // Left Side
    this.drawPoly3D(
      [
        toWorld(-1.05, bodyBottom, -1.8),
        toWorld(-1.05, bodyBottom, 1.8),
        toWorld(-0.95, bodyTop, 1.6),
        toWorld(-0.95, bodyTop, -1.7),
      ],
      "#A3E635",
      "#0F172A",
      1.5
    );
    // Right Side
    this.drawPoly3D(
      [
        toWorld(1.05, bodyBottom, -1.8),
        toWorld(1.05, bodyBottom, 1.8),
        toWorld(0.95, bodyTop, 1.6),
        toWorld(0.95, bodyTop, -1.7),
      ],
      "#84CC16",
      "#0F172A",
      1.5
    );
    // Hood / Top Deck
    this.drawPoly3D(
      [
        toWorld(-0.95, bodyTop, -1.7),
        toWorld(0.95, bodyTop, -1.7),
        toWorld(0.95, bodyTop, 1.6),
        toWorld(-0.95, bodyTop, 1.6),
      ],
      "#CCFF00",
      "#0F172A",
      2
    );

    // 4. Cockpit Glass Cabin
    this.drawPoly3D(
      [
        toWorld(-0.75, bodyTop, -0.7),
        toWorld(0.75, bodyTop, -0.7),
        toWorld(0.65, 1.65, 0.4),
        toWorld(-0.65, 1.65, 0.4),
      ],
      "#0284C7",
      "#38BDF8",
      1.5
    );

    // 5. Rear Sport Wing / Spoiler (Court Orange)
    this.drawPoly3D(
      [
        toWorld(-1.15, 1.35, -1.85),
        toWorld(1.15, 1.35, -1.85),
        toWorld(1.15, 1.45, -1.45),
        toWorld(-1.15, 1.45, -1.45),
      ],
      "#FF5722",
      "#FFFFFF",
      1.5
    );
  }

  renderMinimap() {
    const ctx = this.minimapCtx;
    if (!ctx) return;

    const w = this.minimapCanvas.width;
    const h = this.minimapCanvas.height;
    ctx.clearRect(0, 0, w, h);

    // Background radar circle
    ctx.fillStyle = "#090F1B";
    ctx.fillRect(0, 0, w, h);

    const toMap = (wx, wz) => [
      w * 0.5 + (wx / 60) * (w * 0.45),
      h * 0.5 + (wz / 60) * (h * 0.45),
    ];

    // Center Court rectangle on radar
    const [cx0, cz0] = toMap(-5.5, -11);
    const [cx1, cz1] = toMap(5.5, 11);
    ctx.fillStyle = "#1D4ED8";
    ctx.strokeStyle = "#FFFFFF";
    ctx.lineWidth = 1;
    ctx.fillRect(cx0, cz0, cx1 - cx0, cz1 - cz0);
    ctx.strokeRect(cx0, cz0, cx1 - cx0, cz1 - cz0);

    // Landmarks dots
    for (const lm of WORLD_LANDMARKS) {
      const [lx, lz] = toMap(lm.x, lm.z);
      ctx.fillStyle = lm.color;
      ctx.beginPath();
      ctx.arc(lx, lz, 4.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Player Rover Arrow
    const [px, pz] = toMap(this.car.x, this.car.z);
    ctx.save();
    ctx.translate(px, pz);
    ctx.rotate(-this.car.angle + Math.PI);
    ctx.fillStyle = "#CCFF00";
    ctx.beginPath();
    ctx.moveTo(0, -7);
    ctx.lineTo(5, 6);
    ctx.lineTo(-5, 6);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  loop(timestamp) {
    const t = timestamp * 0.001;
    const dt = Math.min(0.05, t - (this.lastTime || t));
    this.lastTime = t;
    this.time = t;

    this.updatePhysics(dt);
    this.renderWorld();
    this.renderMinimap();

    requestAnimationFrame((ts) => this.loop(ts));
  }
}

window.addEventListener("DOMContentLoaded", () => {
  window.pickleballWorld = new PickleballOpenWorld();
});
