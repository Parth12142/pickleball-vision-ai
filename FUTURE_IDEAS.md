# 💡 Pickleball Vision AI — Future Ideas & Feature Backlog

This document serves as the master backlog for prospective features, advanced AI enhancements, and architectural proposals to be incorporated into future releases without disrupting current milestones.

---

## 📌 Proposed Features Backlog

### 1. Player Ball-Hit & Shot Speed Tracking (Rally Stroke Attribution)

- **Status**: 💡 Planned for Future Integration (Phase 4 / Week 5 Alignment)
- **Target Workstream**: AI Coach Biomechanics & Broadcast Telemetry ([`W5-01`](file:///home/pce118/Desktop/pickleball-vision-ai-main/5_WEEK_TASK_MASTER_LIST.md), [`W5-02`](file:///home/pce118/Desktop/pickleball-vision-ai-main/5_WEEK_TASK_MASTER_LIST.md), [`W5-04`](file:///home/pce118/Desktop/pickleball-vision-ai-main/5_WEEK_TASK_MASTER_LIST.md))

#### 🎯 Concept & User Goal
Track the exact sequence of player hits across each rally with ball exit speeds:
- Log which player made contact: **P1**, **P2**, **P3**, or **P4**.
- Calculate instantaneous ball exit speed in **MPH** at paddle contact.
- Record the exact video timestamp and frame index.
- Export to clean, standalone tables (Markdown & CSV) for match review and broadcast graphics.

#### 🧠 The Core Technical Challenge: "The Middle Ball"
In doubles pickleball, partners often stand close together at the non-volley line (kitchen) or converge in the middle of the court. When one player extends their arm to hit the ball:
- Whole-body bounding boxes overlap.
- Simple centroid or bounding-box distance can falsely assign the hit to the adjacent teammate whose torso center happens to be closer than the hitting player's body center.

#### 🛠️ Recommended Implementation Architecture
To achieve 99%+ attribution accuracy without touching or degrading base perception models:
1. **Additive 17-Keypoint Skeletal Pose Layer (YOLO-Pose / MediaPipe)**:
   - Tracks wrist keypoints (Left Wrist #9, Right Wrist #10) representing the paddle hand.
   - Measures wrist speed surge and arm extension at impact frame.
   - Resolves partner ambiguity even when players stand shoulder-to-shoulder.
2. **Dedicated Paddle Detector Model (`class = paddle`)**:
   - Directly detects the paddle surface to establish exact paddle-to-ball contact points.
3. **Contact Proximity-Climax & Launch Vector**:
   - Identifies the frame where ball-to-wrist physical distance reaches its true minimum ($d < 50\text{ px}$).
   - Validates exit velocity vector along the rebound angle $\vec{v}_{\text{exit}}$.
4. **Standalone Clean Output**:
   - Auto-generates clean Markdown tables (`outputs/player_hit_speeds.md`) and CSVs (`outputs/player_hit_speeds.csv`) containing only the verified hit sequence.

---

## 📋 Idea Template for New Additions

When adding new ideas to this file, use the following structure:
- **Feature Name**: Short descriptive title
- **Status / Target Week**: Phase alignment (e.g. Phase 2, 3, or 4)
- **Concept & User Goal**: What problem does this solve?
- **Technical Architecture**: Models, math, or pipeline modules required
- **Integration Guardrails**: Ensuring zero regression on existing base perception models
