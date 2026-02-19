"""
Parking Slot Editor - Manual polygon drawing
Always starts FRESH for each video.
Saves to file named after the video automatically.
Controls:
  Left click  = add corner (4 clicks = 1 slot)
  Right click = undo last point
  S           = save
  Z           = undo last slot
  C           = clear current points
  ESC / Q     = save and quit
"""

import cv2
import numpy as np
import json
import config


class SlotEditor:

    def __init__(self):
        self.slots       = []
        self.current_pts = []
        self.frame       = None
        self.display     = None
        self.window_name = "SLOT EDITOR"

    def run(self):
        slot_path = config.SLOT_DATA_PATH

        print("\n" + "="*55)
        print("📂 SLOT EDITOR")
        print("="*55)
        print(f"   Video  : {config.VIDEO_SOURCES[config.ACTIVE_SOURCE].get('path', config.ACTIVE_SOURCE)}")
        print(f"   Saving : {slot_path}")
        print("   Starting FRESH — no old polygons loaded")
        print("="*55)

        from video_source_manager import VideoSourceManager
        src = VideoSourceManager()
        if not src.connect():
            print("❌ Cannot connect to video source")
            return

        # Skip first 30 frames for stable frame
        ret, frame = False, None
        for _ in range(30):
            ret, frame = src.read_frame()
        src.release()

        if not ret or frame is None:
            print("❌ Cannot read frame from video")
            return

        self.frame = frame.copy()
        self._refresh_display()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        print("\nINSTRUCTIONS:")
        print("  LEFT CLICK  = add corner (4 per slot)")
        print("  RIGHT CLICK = undo last point")
        print("  S           = save")
        print("  Z           = undo last slot")
        print("  C           = clear current points")
        print("  ESC / Q     = save and quit")
        print(f"\n  Saving to: {slot_path}\n")

        while True:
            cv2.imshow(self.window_name, self.display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('s'), ord('S')):
                self._save(slot_path)

            elif key in (ord('z'), ord('Z')):
                if self.slots:
                    self.slots.pop()
                    self.current_pts = []
                    self._refresh_display()
                    print(f"↩️  Undone. Slots: {len(self.slots)}")
                else:
                    print("   Nothing to undo")

            elif key in (ord('c'), ord('C')):
                self.current_pts = []
                self._refresh_display()
                print("🗑️  Cleared current points")

            elif key in (27, ord('q'), ord('Q')):
                self._save(slot_path)
                print("👋 Exiting editor")
                break

        cv2.destroyAllWindows()

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_pts.append((x, y))
            print(f"  Point {len(self.current_pts)}/4: ({x}, {y})")

            if len(self.current_pts) == 4:
                pts = self._order_points(self.current_pts)
                self.slots.append(pts)
                print(f"✅ Slot #{len(self.slots)} saved!")
                self.current_pts = []

            self._refresh_display()

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.current_pts:
                self.current_pts.pop()
                self._refresh_display()
                print(f"  Undone. Points: {len(self.current_pts)}/4")

    def _order_points(self, pts):
        pts    = sorted(pts, key=lambda p: p[1])
        top    = sorted(pts[:2], key=lambda p: p[0])
        bottom = sorted(pts[2:], key=lambda p: p[0])
        return [top[0], top[1], bottom[1], bottom[0]]

    def _refresh_display(self):
        vis = self.frame.copy()

        for i, slot in enumerate(self.slots):
            pts     = np.array(slot, dtype=np.int32).reshape((-1, 1, 2))
            overlay = vis.copy()
            cv2.fillPoly(overlay, [pts], (0, 200, 0))
            cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            cx = int(np.mean([p[0] for p in slot]))
            cy = int(np.mean([p[1] for p in slot]))
            cv2.putText(vis, f"#{i+1}", (cx-15, cy+5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

        for pt in self.current_pts:
            cv2.circle(vis, pt, 6, (0, 255, 255), -1)

        for i in range(len(self.current_pts)-1):
            cv2.line(vis,
                     self.current_pts[i],
                     self.current_pts[i+1],
                     (0, 255, 255), 2)

        info = [
            f"Slots: {len(self.slots)}  |  "
            f"Points: {len(self.current_pts)}/4  |  "
            f"File: {config.SLOT_DATA_PATH}",
            "LEFT CLICK=add  RIGHT CLICK=undo  "
            "S=save  Z=undo slot  C=clear  ESC=quit"
        ]
        for i, txt in enumerate(info):
            cv2.putText(vis, txt, (10, 25+i*28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 4)
            cv2.putText(vis, txt, (10, 25+i*28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 255), 2)

        self.display = vis

    def _save(self, path):
        with open(path, "w") as f:
            json.dump(self.slots, f, indent=2)
        print(f"💾 Saved {len(self.slots)} slots → {path}")


if __name__ == "__main__":
    editor = SlotEditor()
    editor.run()