# 📚 Pickleball Vision AI — Engineering & Architecture Documentation

Welcome to the central documentation hub for **Pickleball Vision AI**. This repository houses all technical specifications, architectural designs, dataset preparation pipelines, model training workflows, and execution logs organized week-by-week according to the [5-Week Master Implementation Plan](file:///home/pce118/Desktop/pickleball-vision-ai-main/5_WEEK_TASK_MASTER_LIST.md).

---

## 🗺️ 5-Week Implementation Roadmap

| Phase | Timeline | Primary Objective | Status | Documentation |
| :--- | :--- | :--- | :---: | :--- |
| **Phase 1 (Week 1)** | Days 1–5 (Nov 3–16, 2026) | **Base Perception Engine & Colab A100 Training** (15 Tasks) | ✅ **100% Complete** | [`docs/week1/README.md`](file:///home/pce118/Desktop/pickleball-vision-ai-main/docs/week1/README.md) |
| **Phase 1 (Week 2)** | Days 6–10 (Nov 17–23, 2026) | **Base Model Fine-Tuning & Tracking Stabilization** (7 Tasks) | ⏳ **In Progress (W2-01 Done)** | [`docs/week2/README.md`](file:///home/pce118/Desktop/pickleball-vision-ai-main/docs/week2/README.md) |
| **Phase 2 (Week 3)** | Days 11–13 (Nov 24–26, 2026) | **Court Minimap & Parabolic Bounce Kinematics** (5 Tasks) | 📅 Planned | `docs/week3/README.md` |
| **Phase 3 (Week 4)** | Days 14–18 (Nov 27–Dec 03, 2026) | **Rules Engine & Autonomous Scoring (USAP 0-0-2)** (6 Tasks) | 📅 Planned | `docs/week4/README.md` |
| **Phase 4 (Week 5)** | Days 19–23 (Dec 04–10, 2026) | **AI Coach Biomechanics & Broadcast Graphics** (6 Tasks) | 📅 Planned | `docs/week5/README.md` |

---

## 📁 Documentation Structure

```
docs/
├── README.md                      # Documentation index (this file)
└── week1/
    └── README.md                  # Week 1 Complete Guide (Zero to Current State)

FUTURE_IDEAS.md                    # Future ideas and feature backlog (e.g. stroke speed attribution)
```

---

## 🛡️ Core Project Guardrail
> **Phase 1 & Phase 2 Strictly Enforce Zero Hardcoded Rules:**  
> Weeks 1–3 focus exclusively on spatial vision, object detection, metric tracking, and court geometry. Zero referee scoring logic (0-0-2), line calls, or kitchen fault penalties exist in the codebase until Phase 3 (Week 4).
