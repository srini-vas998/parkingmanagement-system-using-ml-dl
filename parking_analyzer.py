"""
Parking Analyzer - FINAL VERSION
Small beautiful status panel + shadow rejection.
"""

import cv2
import numpy as np
import time
from typing import List
import config


class ParkingAnalyzer:

    def __init__(self, slots: List):
        self.slots       = slots
        self.model       = None
        self.fps_time    = time.time()
        self.fps         = 0.0
        self.frame_count = 0
        self.history     = [0] * len(slots)

        print(f"✅ ParkingAnalyzer initialized: {len(slots)} slots")
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(config.YOLO_MODEL)
            print(f"✅ YOLO loaded: {config.YOLO_MODEL}")
        except Exception as e:
            print(f"⚠️  YOLO not available: {e}")
            self.model = None

    def analyze_frame(self, frame):
        self.frame_count += 1
        now           = time.time()
        self.fps      = 1.0 / max(now - self.fps_time, 0.001)
        self.fps_time = now

        vehicle_boxes = self._detect_vehicles(frame)
        occupancy     = self._check_occupancy(frame, vehicle_boxes)

        occupied = sum(occupancy)
        empty    = len(self.slots) - occupied
        total    = len(self.slots)

        result_frame = self._draw(frame.copy(), occupancy, vehicle_boxes)

        return {
            "frame":     result_frame,
            "occupied":  occupied,
            "empty":     empty,
            "total":     total,
            "fps":       self.fps,
            "occupancy": occupancy
        }

    # ═══════════════════════════════════════════
    # DETECT VEHICLES
    # ═══════════════════════════════════════════

    def _detect_vehicles(self, frame):
        if self.model is None:
            return []

        fh, fw        = frame.shape[:2]
        vehicle_boxes = []

        try:
            results = self.model(
                frame,
                conf=0.10,
                verbose=False,
                imgsz=640
            )

            for r in results:
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    bw   = x2 - x1
                    bh   = y2 - y1
                    conf = float(b.conf[0])

                    if (fw * 0.01 < bw < fw * 0.65 and
                            fh * 0.01 < bh < fh * 0.75):
                        vehicle_boxes.append((x1, y1, x2, y2, conf))

        except Exception as e:
            print(f"   Detection error: {e}")

        return vehicle_boxes

    # ═══════════════════════════════════════════
    # CHECK OCCUPANCY
    # YOLO primary + shadow rejection
    # ═══════════════════════════════════════════

    def _check_occupancy(self, frame, vehicle_boxes):
        occupancy = [0] * len(self.slots)
        fh, fw    = frame.shape[:2]

        for i, slot in enumerate(self.slots):
            sx1 = min(p[0] for p in slot)
            sy1 = min(p[1] for p in slot)
            sx2 = max(p[0] for p in slot)
            sy2 = max(p[1] for p in slot)
            slot_area = max(1, (sx2-sx1) * (sy2-sy1))

            # Crop slot ROI
            roi = frame[
                max(0,  sy1):min(fh, sy2),
                max(0,  sx1):min(fw, sx2)
            ]

            if roi.size == 0:
                continue

            # ── SHADOW REJECTION ──────────────────────────
            # Shadow = dark + low saturation variation
            # + low brightness variation
            # Real car = has color, texture, edges
            hsv      = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mean_val = float(np.mean(hsv[:, :, 2]))
            std_sat  = float(np.std(hsv[:, :, 1]))
            std_val  = float(np.std(hsv[:, :, 2]))

            if mean_val < 75 and std_sat < 12 and std_val < 18:
                # This is a shadow — mark empty and skip
                occupancy[i] = 0
                continue

            # ── YOLO CHECK — primary method ───────────────
            yolo_hit = False
            for vx1, vy1, vx2, vy2, conf in vehicle_boxes:
                ix1 = max(sx1, vx1)
                iy1 = max(sy1, vy1)
                ix2 = min(sx2, vx2)
                iy2 = min(sy2, vy2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2-ix1) * (iy2-iy1)
                    if inter / slot_area > 0.08:
                        yolo_hit = True
                        break

            if yolo_hit:
                occupancy[i] = 1
                continue

            # ── FALLBACK: color + edge check ──────────────
            # Only triggers if YOLO missed something
            # Requires BOTH high color variance AND edges
            # Prevents shadow false positives
            std_b = float(np.std(roi[:, :, 0]))
            std_g = float(np.std(roi[:, :, 1]))
            std_r = float(np.std(roi[:, :, 2]))
            color_var = (std_b + std_g + std_r) / 3.0

            gray         = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges        = cv2.Canny(gray, 40, 120)
            edge_density = np.sum(edges > 0) / max(1, edges.size)

            # Strict threshold — must have BOTH
            # color variation AND edge density
            if color_var > 40 and edge_density > 0.10:
                occupancy[i] = 1

        # Smooth — prevents flickering
        for i in range(len(self.slots)):
            self.history[i] = max(0, min(4,
                self.history[i] + (2 if occupancy[i] else -1)))
            occupancy[i] = 1 if self.history[i] >= 2 else 0

        return occupancy

    # ═══════════════════════════════════════════
    # DRAW — Small beautiful panel
    # ═══════════════════════════════════════════

    def _draw(self, frame, occupancy, vehicle_boxes):
        fh, fw = frame.shape[:2]
        overlay = frame.copy()

        # Fill slots with transparent color
        for slot, occ in zip(self.slots, occupancy):
            pts   = np.array(slot, dtype=np.int32).reshape((-1, 1, 2))
            color = config.COLOR_OCCUPIED if occ else config.COLOR_EMPTY
            cv2.fillPoly(overlay, [pts], color)

        cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)

        # Draw slot borders and small labels
        for i, (slot, occ) in enumerate(zip(self.slots, occupancy)):
            pts   = np.array(slot, dtype=np.int32).reshape((-1, 1, 2))
            color = config.COLOR_OCCUPIED if occ else config.COLOR_EMPTY
            label = "OCC" if occ else "FREE"

            cv2.polylines(frame, [pts], True, color, 2)

            cx = int(np.mean([p[0] for p in slot]))
            cy = int(np.mean([p[1] for p in slot]))

            if config.SHOW_SLOT_IDS:
                cv2.putText(frame, f"#{i+1}",
                            (cx-12, cy-5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.32, (255, 255, 255), 1)

            cv2.putText(frame, label,
                        (cx-12, cy+9),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.30, color, 1)

        # YOLO bounding boxes
        if config.SHOW_DETECTION_BOXES:
            for vx1, vy1, vx2, vy2, conf in vehicle_boxes:
                cv2.rectangle(frame,
                              (vx1, vy1), (vx2, vy2),
                              config.COLOR_DETECTION, 1)
                if config.SHOW_CONFIDENCE:
                    cv2.putText(frame, f"{conf:.2f}",
                                (vx1, vy1-4),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.32, config.COLOR_DETECTION, 1)

        # ── COMPACT BEAUTIFUL STATUS PANEL ──────────────
        occupied  = sum(occupancy)
        empty     = len(self.slots) - occupied
        total     = len(self.slots)
        pct       = int(occupied / max(1, total) * 100)

        # Panel size
        pw, ph = 195, 118

        # Semi-transparent dark background
        panel_bg = frame[0:ph, 0:pw].copy()
        cv2.rectangle(panel_bg, (0, 0), (pw, ph), (15, 15, 15), -1)
        cv2.addWeighted(panel_bg, 0.80, frame[0:ph, 0:pw], 0.20,
                        0, frame[0:ph, 0:pw])
        cv2.rectangle(frame[0:ph, 0:pw], (0, 0), (pw, ph), (15, 15, 15), -1)

        # Colored top accent bar
        cv2.rectangle(frame, (0, 0), (pw, 3), (0, 200, 255), -1)

        # Title
        cv2.putText(frame, "PARKING STATUS",
                    (7, 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 200, 255), 1)

        # Thin divider
        cv2.line(frame, (7, 21), (pw-7, 21), (50, 50, 50), 1)

        # Stats rows
        rows = [
            (f"Slots    {total}",    (180, 180, 180), 35),
            (f"FPS      {self.fps:.1f}", (180, 180, 180), 49),
            (f"Occupied {occupied}", (80,  80,  255), 63),
            (f"Empty    {empty}",    (80,  255, 80),  77),
        ]
        for text, color, y in rows:
            cv2.putText(frame, text, (7, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, color, 1)

        # Thin divider
        cv2.line(frame, (7, 83), (pw-7, 83), (50, 50, 50), 1)

        # Occupancy progress bar
        bar_x  = 7
        bar_y  = 90
        bar_w  = pw - 14
        bar_h  = 10

        # Background
        cv2.rectangle(frame,
                      (bar_x, bar_y),
                      (bar_x+bar_w, bar_y+bar_h),
                      (45, 45, 45), -1)

        # Fill
        fill = int(bar_w * pct / 100)
        bar_color = (60,  60, 255) if pct > 80 else \
                    (60, 180, 255) if pct > 50 else \
                    (60, 255,  60)

        if fill > 0:
            cv2.rectangle(frame,
                          (bar_x, bar_y),
                          (bar_x+fill, bar_y+bar_h),
                          bar_color, -1)

        # Percentage text on bar
        cv2.putText(frame, f"{pct}%",
                    (bar_x+2, bar_y+9),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.30, (255, 255, 255), 1)

        # Outer border
        cv2.rectangle(frame, (0, 0), (pw, ph), (60, 60, 60), 1)

        return frame