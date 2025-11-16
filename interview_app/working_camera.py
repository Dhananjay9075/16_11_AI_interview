import time
import threading
import cv2
import numpy as np


class _VideoCapture:
    def __init__(self, camera_index=0):
        print(f"🎥 Attempting to open camera {camera_index}")
        
        # Try DirectShow backend first (works on Windows)
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        
        # If DirectShow fails, try default backend
        if not self.cap.isOpened():
            print(f"❌ DirectShow failed for camera {camera_index}, trying default...")
            self.cap = cv2.VideoCapture(camera_index)
        
        # Try different camera indices if the first one fails
        if not self.cap.isOpened():
            print(f"❌ Camera {camera_index} not available, trying alternatives...")
            for i in range(1, 5):
                print(f"🎥 Trying camera {i} with DirectShow...")
                self.cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if self.cap.isOpened():
                    print(f"✅ Successfully opened camera {i}")
                    break
                else:
                    print(f"❌ DirectShow failed for camera {i}, trying default...")
                    self.cap = cv2.VideoCapture(i)
                    if self.cap.isOpened():
                        print(f"✅ Successfully opened camera {i} with default backend")
                        break
            else:
                print(f"❌ No cameras found, creating dummy camera")
                self.cap = None
        
        if self.cap and self.cap.isOpened():
            # Set optimal camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
            print(f"✅ Camera configured: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} @ {self.cap.get(cv2.CAP_PROP_FPS)}fps")
        
    def isOpened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()
    
    def read(self):
        if self.cap:
            return self.cap.read()
        return False, None
    
    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None


class WorkingVideoCamera:
    """Working camera implementation with real webcam access and YOLO integration."""

    def __init__(self, session_id):
        self.session_id = session_id
        print(f"🎥 Initializing working camera for session {session_id}")
        
        self.video = _VideoCapture()
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
        self._frame_count = 0
        
        # Start frame capture thread if camera is available
        if self.video.isOpened():
            self._start_frame_capture()

    def _start_frame_capture(self):
        """Start background thread to continuously capture frames."""
        self._running = True
        self._frame_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self._frame_thread.start()
        print(f"✅ Camera frame capture started for session {self.session_id}")
        
        # Give the camera thread a moment to capture the first frame
        time.sleep(0.1)

    def _capture_frames(self):
        """Background thread to continuously capture frames."""
        while self._running:
            try:
                ret, frame = self.video.read()
                if ret and frame is not None:
                    with self._frame_lock:
                        self._latest_frame = frame.copy()
                    self._frame_count += 1
                    
                    # Log every 60th frame (about every 2 seconds at 30fps)
                    if self._frame_count % 60 == 1:
                        print(f"📹 Captured frame #{self._frame_count} for session {self.session_id}")
                else:
                    print(f"❌ Failed to read frame for session {self.session_id}")
                    time.sleep(0.1)
            except Exception as e:
                print(f"❌ Error capturing frame: {e}")
                time.sleep(0.1)

    def get_frame(self) -> bytes:
        """Get the latest frame as JPEG bytes."""
        try:
            if not self.video.isOpened():
                print(f"⚠️ Camera not available, serving fallback frame for session {self.session_id}")
                return self._create_fallback_frame()
            
            # Wait up to 1 second for the first frame
            attempts = 0
            while attempts < 10:
                with self._frame_lock:
                    if self._latest_frame is not None:
                        frame = self._latest_frame.copy()
                        if attempts > 0:
                            print(f"📸 Serving real camera frame for session {self.session_id} (waited {attempts*100}ms)")
                        break
                
                # If no frame yet, wait 100ms and try again
                time.sleep(0.1)
                attempts += 1
            else:
                print(f"⚠️ No camera frames available after waiting 1s, serving fallback for session {self.session_id}")
                return self._create_fallback_frame()
            
            # Add timestamp overlay
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add session info overlay
            session_text = f"Session: {str(self.session_id)[:8]}"
            cv2.putText(frame, session_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Encode as JPEG
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                return buf.tobytes()
            else:
                print(f"❌ Failed to encode frame")
                return self._create_fallback_frame()
                
        except Exception as e:
            print(f"❌ Error in get_frame: {e}")
            return self._create_fallback_frame()

    def _create_fallback_frame(self) -> bytes:
        """Create a fallback frame when camera is not available."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add session info
        session_text = f"Session: {str(self.session_id)[:8]}"
        cv2.putText(frame, session_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add status message
        if not self.video.isOpened():
            cv2.putText(frame, "Camera Not Available", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "Please check camera connection", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Camera Initializing...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Add timestamp
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        ok, buf = cv2.imencode('.jpg', frame)
        return buf.tobytes() if ok else b''

    def get_latest_warnings(self) -> dict:
        return self._last_warning_state

    def set_tab_switch_status(self, hidden: bool) -> None:
        self._last_warning_state["tab_switched"] = bool(hidden)

    def cleanup(self) -> None:
        """Clean up camera resources."""
        print(f"🧹 Cleaning up working camera for session {self.session_id}")
        self._running = False
        if self._frame_thread and self._frame_thread.is_alive():
            self._frame_thread.join(timeout=2.0)
        if self.video:
            self.video.release()
        print(f"✅ Working camera cleanup completed for session {self.session_id}")
