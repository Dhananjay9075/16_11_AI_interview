import time
import threading
import cv2
import numpy as np
from dataclasses import dataclass
import torch
from datetime import datetime
import os
from django.conf import settings
from interview_app_11.models import InterviewSession, WarningLog


@dataclass
class _VideoCapture:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
        
    def isOpened(self) -> bool:
        return self.cap.isOpened()
    
    def read(self):
        return self.cap.read()
    
    def release(self):
        if self.cap:
            self.cap.release()


class RealVideoCamera:
    """Real camera implementation that accesses the webcam."""

    def __init__(self, session_id):
        self.session_id = session_id
        print(f"🎥 Initializing camera for session {session_id}")
        self.video = _VideoCapture()
        print(f"🎥 Camera object created: {self.video}")
        print(f"🎥 Camera isOpened: {self.video.isOpened()}")
        
        self._last_warning_state = {
            "no_person_warning_active": False,
            "multiple_people": False,
            "phone_detected": False,
            "no_person": False,
            "low_concentration": False,
            "tab_switched": False,
            "excessive_noise": False,
            "multiple_speakers": False,
        }
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._frame_thread = None
        self._running = False
        
        # Load YOLOv8n model
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov8n', pretrained=True)

        # Start frame capture thread
        self._start_frame_capture()

    def _start_frame_capture(self):
        """Start background thread to continuously capture frames."""
        if not self.video.isOpened():
            print(f"❌ Camera not available for session {self.session_id}")
            # Try to reinitialize camera
            try:
                self.video = _VideoCapture()
                if not self.video.isOpened():
                    print(f"❌ Camera still not available after retry for session {self.session_id}")
                    return
            except Exception as e:
                print(f"❌ Error reinitializing camera: {e}")
                return
            
        self._running = True
        self._frame_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self._frame_thread.start()
        print(f"✅ Camera started for session {self.session_id}")

    def _capture_frames(self):
        """Background thread to continuously capture frames."""
        frame_count = 0
        while self._running:
            try:
                ret, frame = self.video.read()
                if ret:
                    with self._frame_lock:
                        self._latest_frame = frame.copy()
                    frame_count += 1

                    # --- YOLO FRAME ANALYSIS ---
                    results = self.yolo_model(frame)
                    labels = [self.yolo_model.names[int(cls)] for cls in results.pred[0][:, -1]] if len(results.pred) > 0 else []
                    warning = None
                    if labels.count('person') > 1:
                        warning = 'multiple_people'
                    elif 'person' not in labels:
                        warning = 'no_person'
                    elif 'cell phone' in labels or 'mobile phone' in labels:
                        warning = 'phone_detected'
                    if warning:
                        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                        fname = f"{self.session_id}_{warning}_{ts}.jpg"
                        img_dir = os.path.join(settings.MEDIA_ROOT, "proctoring_snaps")
                        os.makedirs(img_dir, exist_ok=True)
                        img_path = os.path.join(img_dir, fname)
                        cv2.imwrite(img_path, frame)
                        try:
                            session = InterviewSession.objects.get(id=self.session_id)
                            WarningLog.objects.create(session=session, warning_type=warning, snapshot=fname)
                        except Exception as e:
                            print("[Proctoring] Failed to log warning:", e)
                        self._last_warning_state[warning] = True
                        threading.Timer(2, lambda: self._last_warning_state.update({warning: False})).start()
                    if frame_count % 30 == 1:  # Log every 30th frame
                        print(f"📹 Captured frame #{frame_count} for session {self.session_id}")
                else:
                    print(f"❌ Failed to read frame for session {self.session_id}")
                    time.sleep(0.1)
            except Exception as e:
                print(f"❌ Error capturing frame: {e}")
                time.sleep(0.1)

    def get_frame(self) -> bytes:
        """Get the latest frame as JPEG bytes."""
        try:
            with self._frame_lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame.copy()
                    print(f"📸 Serving real camera frame for session {self.session_id}")
                else:
                    # Fallback: create a black frame with session info
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    text = f"Session: {str(self.session_id)[:8]}"
                    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, "Camera Loading...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    print(f"⚠️ Serving fallback frame (no camera data) for session {self.session_id}")
            
            # Add timestamp overlay
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Encode as JPEG with lower quality for faster processing
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if ok:
                print(f"✅ Frame encoded successfully, size: {len(buf.tobytes())} bytes")
                return buf.tobytes()
            else:
                print(f"❌ Failed to encode frame")
                return b''
        except Exception as e:
            print(f"❌ Error in get_frame: {e}")
            # Return a simple black frame as fallback
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            ok, buf = cv2.imencode('.jpg', frame)
            return buf.tobytes() if ok else b''

    def get_latest_warnings(self) -> dict:
        return self._last_warning_state

    def set_tab_switch_status(self, hidden: bool) -> None:
        self._last_warning_state["tab_switched"] = bool(hidden)

    def cleanup(self) -> None:
        """Clean up camera resources."""
        print(f"🧹 Cleaning up camera for session {self.session_id}")
        self._running = False
        if self._frame_thread and self._frame_thread.is_alive():
            self._frame_thread.join(timeout=2.0)
        if self.video:
            self.video.release()
        print(f"✅ Camera cleanup completed for session {self.session_id}")
