"""
Smart Slot Detector - Per-video slot files.
"""

import cv2
import numpy as np
import json
from typing import List
import config


class SmartSlotDetector:

    def __init__(self):
        self.slots = []

    def detect_slots_from_video(self, video_source) -> List:
        print("\n" + "="*60)
        print("🎯 AUTOMATIC SLOT DETECTION")
        print("="*60)

        print("\n📷 Collecting frames...")
        frames = self._collect_frames(video_source, n=60)
        if not frames:
            raise RuntimeError("Cannot read frames")

        h, w = frames[0].shape[:2]
        print(f"   Frame size: {w} x {h}")

        print("\n🖼️  Building background...")
        bg = self._median_background(frames)

        print("\n🚗 Finding parking zones...")
        parking_zones = self._find_parking_zones(frames, w, h)
        print(f"   Found {len(parking_zones)} zones")

        print("\n📏 Detecting parking lines...")
        self.slots = self._detect_slots_from_lines(bg, parking_zones, w, h)
        print(f"   Line detection: {len(self.slots)} slots")

        if len(self.slots) < 3:
            print("⚠️  Using YOLO fallback...")
            self.slots = self._yolo_fallback(frames, w, h)

        self.slots = self._deduplicate(self.slots)
        self.slots = self._remove_overlapping(self.slots)

        self._save(self.slots)
        print(f"\n✅ DONE: {len(self.slots)} slots")
        print("="*60)
        return self.slots

    def load_slots(self) -> bool:
        filepath = config.SLOT_DATA_PATH
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            self.slots = [[tuple(p) for p in slot] for slot in data]
            print(f"📂 Loaded {len(self.slots)} slots from {filepath}")
            return True
        except Exception:
            return False

    def get_slots(self):
        return self.slots

    def visualize_detection(self, frame):
        vis = frame.copy()
        for i, slot in enumerate(self.slots):
            pts = np.array(slot, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, (0, 255, 255), 2)
            cx = int(np.mean([p[0] for p in slot]))
            cy = int(np.mean([p[1] for p in slot]))
            cv2.putText(vis, f"#{i+1}", (cx-15, cy+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(vis, f"AUTO-DETECTED: {len(self.slots)} SLOTS",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 255), 3)
        cv2.putText(vis, "Press ANY KEY to start monitoring",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        return vis

    def _collect_frames(self, src, n=60):
        frames = []
        attempt = 0
        while len(frames) < n and attempt < n * 4:
            ret, frame = src.read_frame()
            attempt += 1
            if not ret or frame is None:
                break
            if attempt % 3 == 0:
                frames.append(frame)
        print(f"   Collected {len(frames)} frames")
        return frames

    def _median_background(self, frames):
        sample = frames[::3]
        stack  = np.stack(sample, axis=0).astype(np.uint8)
        bg     = np.median(stack, axis=0).astype(np.uint8)
        print(f"   Background from {len(sample)} frames")
        return bg

    def _find_parking_zones(self, frames, fw, fh):
        try:
            from ultralytics import YOLO
            model = YOLO(config.YOLO_MODEL)
        except Exception as e:
            print(f"   YOLO failed: {e}")
            return [(0, fh, 0, fw)]

        all_boxes = []
        for frame in frames[::5]:
            results = model(frame, conf=0.10, verbose=False, imgsz=640)
            for r in results:
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    bw, bh = x2-x1, y2-y1
                    if (fw*0.02 < bw < fw*0.40 and
                            fh*0.02 < bh < fh*0.50 and
                            bw < fw*0.6 and bh < fh*0.6):
                        all_boxes.append((x1, y1, x2, y2))

        if not all_boxes:
            return [(0, fh, 0, fw)]

        heights   = [b[3]-b[1] for b in all_boxes]
        med_h     = float(np.median(heights))
        ys        = [(b[1]+b[3])//2 for b in all_boxes]
        ys_sorted = sorted(set(ys))

        groups = [[ys_sorted[0]]]
        for y in ys_sorted[1:]:
            if y - groups[-1][-1] < med_h * 1.2:
                groups[-1].append(y)
            else:
                groups.append([y])

        zones = []
        for group in groups:
            if len(group) < 2:
                continue
            pad = int(med_h * 0.8)
            zt  = max(0,  min(group) - pad)
            zb  = min(fh, max(group) + pad)
            zone_boxes = [b for b in all_boxes
                          if zt <= (b[1]+b[3])//2 <= zb]
            if not zone_boxes:
                continue
            zl = max(0,  min(b[0] for b in zone_boxes) - 20)
            zr = min(fw, max(b[2] for b in zone_boxes) + 20)
            zones.append((zt, zb, zl, zr))
            print(f"   Zone: y=[{zt}→{zb}] x=[{zl}→{zr}]")

        return zones if zones else [(0, fh, 0, fw)]

    def _detect_slots_from_lines(self, bg, parking_zones, fw, fh):
        all_slots = []

        for zone_idx, zone in enumerate(parking_zones):
            zt, zb, zl, zr = zone
            zh = zb - zt
            zw = zr - zl
            if zh < 20 or zw < 20:
                continue

            zone_img = bg[zt:zb, zl:zr]
            gray     = cv2.cvtColor(zone_img, cv2.COLOR_BGR2GRAY)
            blurred  = cv2.GaussianBlur(gray, (3, 3), 0)

            binary = cv2.adaptiveThreshold(
                blurred, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=15, C=-8)

            hsv        = cv2.cvtColor(zone_img, cv2.COLOR_BGR2HSV)
            yellow     = cv2.inRange(hsv,
                                     np.array([15, 40, 40]),
                                     np.array([40, 255, 255]))
            white      = cv2.inRange(hsv,
                                     np.array([0, 0, 160]),
                                     np.array([180, 40, 255]))
            color_mask = cv2.bitwise_or(yellow, white)
            combined   = cv2.bitwise_or(binary, color_mask)
            kernel     = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            combined   = cv2.morphologyEx(combined,
                                          cv2.MORPH_CLOSE, kernel)
            edges      = cv2.Canny(combined, 30, 100)

            lines = cv2.HoughLinesP(
                edges, 1, np.pi/180,
                threshold=30,
                minLineLength=int(min(zh, zw) * 0.15),
                maxLineGap=25)

            if lines is None:
                print(f"   Zone {zone_idx+1}: no lines found")
                continue

            h_ys, v_xs = [], []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
                if angle < 25 or angle > 155:
                    h_ys.append((y1+y2)//2)
                elif 65 < angle < 115:
                    v_xs.append((x1+x2)//2)

            h_ys_c = self._cluster(h_ys, 12) if h_ys else []
            v_xs_c = self._cluster(v_xs, 12) if v_xs else []
            h_ys_c.sort()
            v_xs_c.sort()

            print(f"   Zone {zone_idx+1}: "
                  f"{len(h_ys_c)} h-lines {len(v_xs_c)} v-lines")

            slots_found = 0
            if len(h_ys_c) >= 2 and len(v_xs_c) >= 2:
                for r in range(len(h_ys_c)-1):
                    for c in range(len(v_xs_c)-1):
                        x1 = v_xs_c[c]   + zl
                        x2 = v_xs_c[c+1] + zl
                        y1 = h_ys_c[r]   + zt
                        y2 = h_ys_c[r+1] + zt
                        sw, sh = x2-x1, y2-y1
                        if (fw*0.02 < sw < fw*0.35 and
                                fh*0.02 < sh < fh*0.45):
                            all_slots.append([
                                (x1, y1), (x2, y1),
                                (x2, y2), (x1, y2)])
                            slots_found += 1
            elif len(v_xs_c) >= 2:
                for c in range(len(v_xs_c)-1):
                    x1 = v_xs_c[c]   + zl
                    x2 = v_xs_c[c+1] + zl
                    if fw*0.02 < x2-x1 < fw*0.35:
                        all_slots.append([
                            (x1, zt), (x2, zt),
                            (x2, zb), (x1, zb)])
                        slots_found += 1
            elif len(h_ys_c) >= 2:
                for r in range(len(h_ys_c)-1):
                    y1 = h_ys_c[r]   + zt
                    y2 = h_ys_c[r+1] + zt
                    if fh*0.02 < y2-y1 < fh*0.45:
                        all_slots.append([
                            (zl, y1), (zr, y1),
                            (zr, y2), (zl, y2)])
                        slots_found += 1

            print(f"   Zone {zone_idx+1}: {slots_found} slots built")

        return all_slots

    def _cluster(self, values, gap=12):
        if not values:
            return []
        vals     = sorted(values)
        clusters = [[vals[0]]]
        for v in vals[1:]:
            if v - clusters[-1][-1] < gap:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        return [int(np.mean(c)) for c in clusters]

    def _yolo_fallback(self, frames, fw, fh):
        print("   YOLO fallback running...")
        try:
            from ultralytics import YOLO
            model = YOLO(config.YOLO_MODEL)
        except Exception:
            return []

        raw_boxes = []
        for frame in frames:
            results = model(frame, conf=0.10, verbose=False, imgsz=640)
            for r in results:
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    bw, bh = x2-x1, y2-y1
                    if (fw*0.02 < bw < fw*0.35 and
                            fh*0.02 < bh < fh*0.45 and
                            bw < fw*0.5 and bh < fh*0.6):
                        raw_boxes.append((x1, y1, x2, y2))

        merged = self._merge_nearby(raw_boxes, 50)
        if not merged:
            return []

        def cy(b): return (b[1]+b[3])//2
        sorted_b = sorted(merged, key=cy)
        heights  = [b[3]-b[1] for b in merged]
        med_h    = float(np.median(heights))
        row_gap  = med_h * 0.5

        rows = []
        for box in sorted_b:
            placed = False
            for row in rows:
                if abs(np.mean([(b[1]+b[3])//2
                               for b in row]) - cy(box)) < row_gap:
                    row.append(box)
                    placed = True
                    break
            if not placed:
                rows.append([box])

        rows      = [r for r in rows if len(r) >= 2]
        all_slots = []

        for row in rows:
            row.sort(key=lambda b: b[0])
            sw   = int(np.median([b[2]-b[0] for b in row]))
            pad  = 10
            y1   = max(0,  int(np.median([b[1] for b in row])) - pad)
            y2   = min(fh, int(np.median([b[3] for b in row])) + pad)
            x1   = row[0][0]
            x2   = row[-1][2]
            span = x2 - x1
            if span <= 0 or sw <= 0:
                continue
            nc  = len(row)
            nw  = max(1, round(span/sw))
            n   = nc if abs(nc-nw) <= 1 else min(nc, nw)
            esw = span / n
            for i in range(n):
                sx1 = max(0,  int(x1+i*esw))
                sx2 = min(fw, int(x1+(i+1)*esw))
                if sx2 > sx1 and y2 > y1:
                    all_slots.append([
                        (sx1, y1), (sx2, y1),
                        (sx2, y2), (sx1, y2)])
        return all_slots

    def _merge_nearby(self, boxes, dist):
        if not boxes:
            return []
        centers = [((x1+x2)//2, (y1+y2)//2, x1, y1, x2, y2)
                   for x1, y1, x2, y2 in boxes]
        used   = [False] * len(centers)
        merged = []
        for i, ci in enumerate(centers):
            if used[i]:
                continue
            cluster = [ci]
            used[i] = True
            for j, cj in enumerate(centers):
                if used[j]:
                    continue
                if np.sqrt((ci[0]-cj[0])**2 +
                           (ci[1]-cj[1])**2) < dist:
                    cluster.append(cj)
                    used[j] = True
            merged.append((
                int(np.mean([c[2] for c in cluster])),
                int(np.mean([c[3] for c in cluster])),
                int(np.mean([c[4] for c in cluster])),
                int(np.mean([c[5] for c in cluster]))))
        return merged

    def _deduplicate(self, slots):
        if not slots:
            return slots
        centers = [((s[0][0]+s[2][0])//2, (s[0][1]+s[2][1])//2)
                   for s in slots]
        keep = [True] * len(slots)
        for i in range(len(slots)):
            if not keep[i]:
                continue
            for j in range(i+1, len(slots)):
                if not keep[j]:
                    continue
                d = np.sqrt((centers[i][0]-centers[j][0])**2 +
                            (centers[i][1]-centers[j][1])**2)
                if d < 30:
                    keep[j] = False
        return [s for s, k in zip(slots, keep) if k]

    def _remove_overlapping(self, slots):
        if not slots:
            return slots

        def area(s):
            return max(1, abs(s[1][0]-s[0][0]) *
                       abs(s[2][1]-s[0][1]))

        def inter(a, b):
            ix1 = max(a[0][0], b[0][0])
            iy1 = max(a[0][1], b[0][1])
            ix2 = min(a[2][0], b[2][0])
            iy2 = min(a[2][1], b[2][1])
            if ix2 > ix1 and iy2 > iy1:
                return (ix2-ix1) * (iy2-iy1)
            return 0

        keep = [True] * len(slots)
        for i in range(len(slots)):
            if not keep[i]:
                continue
            for j in range(i+1, len(slots)):
                if not keep[j]:
                    continue
                ov = inter(slots[i], slots[j])
                sm = min(area(slots[i]), area(slots[j]))
                if ov / sm > 0.40:
                    if area(slots[i]) >= area(slots[j]):
                        keep[j] = False
                    else:
                        keep[i] = False

        result  = [s for s, k in zip(slots, keep) if k]
        removed = len(slots) - len(result)
        if removed > 0:
            print(f"   Removed {removed} overlapping slots")
        return result

    def _save(self, slots):
        path = config.SLOT_DATA_PATH
        with open(path, "w") as f:
            json.dump(slots, f, indent=2)
        print(f"💾 Saved {len(slots)} slots → {path}")