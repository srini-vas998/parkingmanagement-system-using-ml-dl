"""
Parking Management System - FINAL VERSION
"""

import cv2
from video_source_manager import VideoSourceManager
from smart_slot_detector import SmartSlotDetector
from parking_analyzer import ParkingAnalyzer

print("\n" + "="*70)
print("🚗 PARKING MANAGEMENT SYSTEM")
print("="*70)

print("🔌 Loading Parking Lot...")
src = VideoSourceManager()

try:
    connected = src.connect()
    if not connected:
        input("Press Enter to exit...")
        exit()
    print("✅ Connected!")
except Exception as e:
    print(f"❌ Video error: {e}")
    input("Press Enter to exit...")
    exit()

det = SmartSlotDetector()

if det.load_slots():
    slots = det.get_slots()
    print(f"✅ Loaded {len(slots)} slots from file")
else:
    print("🔍 Detecting parking slots...")
    try:
        slots = det.detect_slots_from_video(src)
    except Exception as e:
        print(f"❌ Detection error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
        exit()

    if not slots or len(slots) == 0:
        print("❌ No slots detected!")
        input("Press Enter to exit...")
        exit()

    src.release()
    src2 = VideoSourceManager()
    src2.connect()
    ret, frame = src2.read_frame()
    if ret and frame is not None:
        vis = det.visualize_detection(frame)
        cv2.imshow("DETECTED SLOTS - Press ANY KEY", vis)
        print("\n👀 Press ANY KEY to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    src2.release()

    src = VideoSourceManager()
    src.connect()

print(f"\n🤖 Initializing analyzer...")
try:
    ana = ParkingAnalyzer(slots)
except Exception as e:
    print(f"❌ Analyzer error: {e}")
    input("Press Enter to exit...")
    exit()

print("\n" + "="*70)
print("▶️  LIVE MONITORING")
print("="*70)
print("Controls: ESC=Exit | P=Pause | R=Re-detect")
print("="*70 + "\n")

frame_num = 0
paused    = False

try:
    while True:
        if not paused:
            ret, frame = src.read_frame()

            if not ret or frame is None:
                print("🔄 Video ended, restarting...")
                src.release()
                src = VideoSourceManager()
                src.connect()
                continue

            frame_num += 1
            result = ana.analyze_frame(frame)

            if result and result["frame"] is not None:
                if frame_num % 30 == 0:
                    print(f"📊 Frame {frame_num:5d} | "
                          f"Occupied: {result['occupied']:2d}/{result['total']:2d} | "
                          f"Empty: {result['empty']:2d} | "
                          f"FPS: {result['fps']:.1f}")

                cv2.putText(result["frame"],
                            f"Frame: {frame_num}",
                            (1050, 700),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2)

                cv2.imshow("Parking Management System", result["frame"])
            else:
                cv2.imshow("Parking Management System", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            print("\n👋 Exiting...")
            break
        elif key == ord('p'):
            paused = not paused
            print("⏸️  PAUSED" if paused else "▶️  RESUMED")
        elif key == ord('r'):
            print("\n🔄 Re-detecting slots...")
            src.release()
            src = VideoSourceManager()
            src.connect()
            slots = det.detect_slots_from_video(src)
            if slots:
                ana = ParkingAnalyzer(slots)
                src.release()
                src = VideoSourceManager()
                src.connect()
                frame_num = 0

except KeyboardInterrupt:
    print("\n⚠️  Interrupted")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    src.release()
    cv2.destroyAllWindows()
    print("\n✅ Stopped")
    input("Press Enter to exit...")
