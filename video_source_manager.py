"""
Video Source Manager - FINAL VERSION
Supports: MP4, AVI, WEBM, RTSP, HTTP, Webcam
"""

import cv2
import time
import config


class VideoSourceManager:

    def __init__(self):
        self.cap         = None
        self.source      = None
        self.source_type = None

    def connect(self):
        src = config.VIDEO_SOURCES.get(config.ACTIVE_SOURCE, {})
        self.source_type = config.ACTIVE_SOURCE

        if self.source_type == "webcam":
            idx = src.get("index", 0)
            self.source = idx
            return self._open_webcam(idx)

        elif self.source_type == "rtsp":
            url = src.get("url", "")
            self.source = url
            return self._open_rtsp(url)

        elif self.source_type == "http":
            url = src.get("url", "")
            self.source = url
            return self._open_http(url)

        else:
            path = src.get("path", "")
            self.source = path
            return self._open_file(path)

    def _open_file(self, path):
        try:
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.cap = cap
                    print(f"✅ Video loaded: {path}")
                    return True
                cap.release()
        except Exception as e:
            print(f"   Error: {e}")
        print(f"❌ Video error: Cannot open video: {path}")
        return False

    def _open_webcam(self, index):
        try:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
                self.cap = cap
                print(f"✅ Webcam {index} connected")
                return True
            cap.release()
        except Exception as e:
            print(f"   Error: {e}")
        print(f"❌ Webcam error: Cannot open webcam {index}")
        return False

    def _open_rtsp(self, url):
        try:
            cap = cv2.VideoCapture(url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.cap = cap
                    print(f"✅ RTSP connected: {url}")
                    return True
            cap.release()
        except Exception as e:
            print(f"   Error: {e}")
        print(f"❌ RTSP error: Cannot connect to {url}")
        return False

    def _open_http(self, url):
        try:
            cap = cv2.VideoCapture(url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    self.cap = cap
                    print(f"✅ HTTP connected: {url}")
                    return True
            cap.release()
        except Exception as e:
            print(f"   Error: {e}")
        print(f"❌ HTTP error: Cannot connect to {url}")
        return False

    def read_frame(self):
        if self.cap is None or not self.cap.isOpened():
            if not self._reconnect():
                return False, None

        if self.source_type in ("rtsp", "webcam"):
            self.cap.grab()

        ret, frame = self.cap.read()

        if not ret and self.source_type == "file":
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        if not ret or frame is None:
            return False, None

        frame = cv2.resize(frame,
                           (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        return True, frame

    def _reconnect(self):
        print("🔄 Reconnecting...")
        if self.cap:
            self.cap.release()
        time.sleep(2)
        return self.connect()

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def get_fps(self):
        if self.cap:
            return self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        return 25.0
