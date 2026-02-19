"""Configuration"""

import os

ACTIVE_SOURCE = "file"

VIDEO_SOURCES = {
    "file": {
        "path": r"C:\Users\SADI SRINIVAS\OneDrive\Desktop\APMS_V2\data\parking_video_fixed.mp4"
    },
    "webcam": {
        "index": 0
    },
    "rtsp": {
        "url": "rtsp://username:password@192.168.1.100:554/stream"
    },
    "http": {
        "url": "http://192.168.1.100:8080/video"
    }
}

YOLO_MODEL = "yolov8l.pt"
VEHICLE_CLASSES = [2, 3, 5, 7]
CONFIDENCE_THRESHOLD = 0.30
IOU_THRESHOLD = 0.5

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
SKIP_FRAMES = 0
TEMPORAL_WINDOW = 3
OCCUPANCY_THRESHOLD = 0.5

EDGE_DENSITY_MIN = 0.08
EDGE_DENSITY_MAX = 0.35
CANNY_LOW = 50
CANNY_HIGH = 150

SHOW_DETECTION_BOXES = True
SHOW_CONFIDENCE = True
SHOW_SLOT_IDS = True

COLOR_OCCUPIED = (0, 0, 255)
COLOR_EMPTY = (0, 255, 0)
COLOR_DETECTION = (255, 0, 0)
COLOR_TEXT = (255, 255, 255)

ENABLE_MOTION_DETECTION = False
SAVE_SLOT_DATA = True
SHOW_FPS = True


def get_slot_data_path():
    src = VIDEO_SOURCES.get(ACTIVE_SOURCE, {})
    if ACTIVE_SOURCE == "file":
        path     = src.get("path", "default")
        basename = os.path.splitext(os.path.basename(path))[0]
        return f"slots_{basename}.json"
    elif ACTIVE_SOURCE == "webcam":
        index = src.get("index", 0)
        return f"slots_webcam_{index}.json"
    elif ACTIVE_SOURCE == "rtsp":
        return "slots_rtsp.json"
    elif ACTIVE_SOURCE == "http":
        return "slots_http.json"
    return "slots_default.json"


SLOT_DATA_PATH = get_slot_data_path()