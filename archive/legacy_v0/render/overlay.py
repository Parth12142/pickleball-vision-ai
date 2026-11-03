import cv2

COLORS = [  # BGR colours, one per player
    (0, 165, 255),   # orange
    (80, 80, 255),   # red
    (255, 200, 0),   # cyan/blue
    (200, 80, 200),  # purple
    (0, 220, 0),     # green
    (0, 255, 255),   # yellow
]

def color_for(track_id):
    return COLORS[track_id % len(COLORS)]

def draw_player(frame, box, track_id, label=True):
    x1, y1, x2, y2 = map(int, box)
    color = color_for(track_id)
    cx = (x1 + x2) // 2
    w = x2 - x1
    axes = (int(w * 0.75), int(w * 0.22))      # wide, flat ellipse on the ground

    # soft filled glow
    overlay = frame.copy()
    cv2.ellipse(overlay, (cx, y2), axes, 0, 0, 360, color, -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    # solid ring
    cv2.ellipse(frame, (cx, y2), axes, 0, 0, 360, color, 3)

    if label:
        cv2.putText(frame, f"P{track_id}", (cx - 15, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame