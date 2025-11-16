import time
import cv2
import numpy as np
import os
from datetime import datetime
from django.conf import settings


class _VideoCapture:
    def __init__(self, camera_index=0):
        print(f"🎥 Attempting to open camera {camera_index}")
        self.cap = None
        
        try:
            # Try DirectShow backend first (works on Windows)
            self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            
            # If DirectShow fails, try default backend
            if not self.cap.isOpened():
                print(f"❌ DirectShow failed for camera {camera_index}, trying default...")
                self.cap = cv2.VideoCapture(camera_index)
            
            if not self.cap.isOpened():
                print(f"❌ Failed to open camera {camera_index}")
                # Try different camera indices
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
                    print(f"❌ No cameras found - will use fallback frames")
                    self.cap = None
                    return
            
            if self.cap and self.cap.isOpened():
                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print(f"✅ Camera configured successfully")
                
                # Give camera a moment to initialize
                time.sleep(0.2)
                
                # Test reading a frame to ensure it's working
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    print(f"✅ Camera test frame captured: {test_frame.shape}")
                else:
                    print(f"⚠️ Camera test frame failed, but camera is opened")
        except Exception as e:
            print(f"❌ Camera initialization error: {e}")
            self.cap = None
        
    def isOpened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()
    
    def read(self):
        if self.cap and self.cap.isOpened():
            return self.cap.read()
        return False, None
    
    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None


class SimpleRealVideoCamera:
    """Simple synchronous camera implementation."""

    def __init__(self, session_id):
        self.session_id = session_id
        print(f"🎥 Initializing simple camera for session {session_id}")
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
        # NEW: frame buffer and thread control
        self._frame_lock = None
        self._latest_frame = None
        self._running = False
        
        # Warning tracking for counts
        self._warning_counts = {}
        self._last_warning_logged = {}  # Track last time each warning was logged to avoid spam
        
        # Cached fallback frame to avoid regenerating
        self._cached_fallback_frame = None
        self._fallback_frame_cache_time = 0
        self._fallback_print_time = 0  # Track when we last printed fallback warning
        
        # YOLO model - NOT loaded during initialization (only during technical interview)
        self._yolo = None
        self._yolo_loaded = False
        self._proctoring_active = False  # Flag to track if proctoring/warning capture is active
        
        # NEW: start basic frame capture loop (without YOLO) if camera available
        # This allows camera to work for identity verification without YOLO
        try:
            import threading
            import cv2
            if self.video.isOpened():
                self._frame_lock = threading.Lock()
                self._running = True
                # Start basic frame capture only (no YOLO detection yet)
                self._detector_thread = threading.Thread(target=self._capture_and_detect_loop, daemon=True)
                self._detector_thread.start()
                print(f"✅ Basic camera frame capture started for session {self.session_id} (YOLO not loaded yet)")
        except Exception as e:
            print(f"⚠️ Failed to start frame capture loop: {e}")
    
    def activate_yolo_proctoring(self):
        """Activate YOLO model and start proctoring warnings when technical interview starts"""
        if self._yolo_loaded:
            print(f"✅ YOLO already loaded for session {self.session_id}")
            self._proctoring_active = True
            return True
        
        try:
            from ultralytics import YOLO
            try:
                # Load YOLO model only when technical interview starts
                self._yolo = YOLO('yolov8n.pt')
                self._yolo_loaded = True
                self._proctoring_active = True
                print(f"✅ YOLOv8 model loaded and proctoring activated for session {self.session_id}")
                return True
            except Exception as e:
                print(f"⚠️ Could not load yolov8n.pt: {e}")
                self._yolo = None
                return False
        except Exception as e:
            print(f"ℹ️ ultralytics not installed; falling back to Haar cascade. {e}")
            return False

    def _capture_and_detect_loop(self):
        """Continuously capture frames, update latest frame, and run simple face-based warnings."""
        # Load Haar cascade once (fallback if YOLO not available)
        try:
            import cv2
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            print(f"⚠️ Failed to load Haar cascade: {e}")
            face_cascade = None
        
        import time
        import time as _t
        import threading
        import queue
        
        last_gray = None
        last_motion_ts = _t.time()
        LOW_MOTION_WINDOW_SEC = 8.0
        MOTION_LOW_THRESH = 2.5  # mean absolute diff threshold (tune if needed)
        
        # Background thread for saving snapshots and logging to DB (non-blocking)
        warning_queue = queue.Queue(maxsize=10)  # Limit queue size
        
        def _background_warning_logger():
            """Background thread to save snapshots and log to database without blocking detection."""
            while self._running:
                try:
                    # Get warning data from queue (with short timeout to prevent blocking)
                    warning_data = warning_queue.get(timeout=0.5)
                    if warning_data is None:
                        break
                    warning_type, frame_copy, snapshot_filename = warning_data
                    
                    # Save snapshot only if proctoring is active (technical interview started)
                    # Skip snapshot saving during identity verification
                    if self._proctoring_active and frame_copy is not None and snapshot_filename:
                        try:
                            img_dir = os.path.join(settings.MEDIA_ROOT, "proctoring_snaps")
                            os.makedirs(img_dir, exist_ok=True)
                            img_path = os.path.join(img_dir, snapshot_filename)
                            cv2.imwrite(img_path, frame_copy)
                            print(f"📸 Snapshot saved: {snapshot_filename}")
                        except Exception as e:
                            print(f"[Proctoring] Failed to save snapshot: {e}")
                    elif not self._proctoring_active:
                        # Proctoring not active - skip snapshot saving
                        print(f"⚠️ Skipping snapshot save - proctoring not active yet (still in identity verification)")
                    
                    # Log to database (with timeout to prevent blocking)
                    # Only log warnings if proctoring is active (technical interview started)
                    if self._proctoring_active:
                        try:
                            from .models import InterviewSession, WarningLog
                            from django.db import transaction
                            # Use select_for_update with nowait to prevent blocking
                            try:
                                session = InterviewSession.objects.select_for_update(nowait=True).get(id=self.session_id)
                                with transaction.atomic():
                                    WarningLog.objects.create(
                                        session=session,
                                        warning_type=warning_type,
                                        snapshot=snapshot_filename
                                    )
                            except Exception as db_error:
                                # Skip database logging if it would block - snapshot is still saved
                                # Only print if it's not the expected "no column named snapshot" error
                                if 'snapshot' not in str(db_error).lower():
                                    pass  # Other errors are silently skipped
                        except Exception as e:
                            # Silently continue - snapshot was saved, DB logging is optional
                            pass
                    else:
                        # Proctoring not active yet (still in identity verification), don't save warnings
                        print(f"⚠️ Warning detected but proctoring not active yet - skipping snapshot save")
                    
                    warning_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[Proctoring] Background logger error: {e}")
        
        logger_thread = threading.Thread(target=_background_warning_logger, daemon=True)
        logger_thread.start()
        self._warning_logger_queue = warning_queue
        
        # YOLO detection frequency - run every ~1.5 seconds to significantly reduce CPU usage
        last_yolo_time = _t.time()
        YOLO_INTERVAL = 1.5  # Run YOLO every 1.5 seconds (reduced from 0.5s for better performance)
        
        # Add debouncing for absence warning - require continuous absence for 3 seconds
        absence_start_time = None
        ABSENCE_THRESHOLD_SEC = 3.0  # Must be absent for 3 seconds before warning
        
        while self._running:
            try:
                # Check if video is still opened, try to reopen if needed (but don't spam logs)
                if not self.video.isOpened():
                    # Only print warning every 30 seconds to reduce log spam
                    now_check = _t.time()
                    if not hasattr(self, '_last_reopen_attempt_time'):
                        self._last_reopen_attempt_time = 0
                    if now_check - self._last_reopen_attempt_time > 30:
                        print(f"⚠️ Camera closed for session {str(self.session_id)[:8]}, attempting to reopen...")
                        self._last_reopen_attempt_time = now_check
                    
                    try:
                        if hasattr(self.video, 'cap') and self.video.cap:
                            self.video.release()
                        self.video = _VideoCapture()
                        if not self.video.isOpened():
                            time.sleep(2.0)  # Wait longer before retry
                            continue
                        else:
                            print(f"✅ Camera successfully reopened for session {str(self.session_id)[:8]}")
                    except Exception as e:
                        if now_check - self._last_reopen_attempt_time <= 30:
                            # Only log if we haven't logged recently
                            pass
                        time.sleep(2.0)
                        continue
                
                ret, frame = self.video.read()
                if not ret or frame is None:
                    time.sleep(0.1)  # Increased sleep to reduce CPU when camera fails
                    continue
                # Save latest frame (only update if lock is available quickly)
                if self._frame_lock is not None:
                    try:
                        # Use non-blocking lock acquisition to prevent blocking
                        if self._frame_lock.acquire(blocking=False):
                            try:
                                self._latest_frame = frame.copy()
                            finally:
                                self._frame_lock.release()
                        else:
                            # Lock is held, skip frame update this cycle (reduces blocking)
                            pass
                    except Exception:
                        pass
                
                # Run detections - YOLO only every 0.5s to reduce CPU usage
                has_person = False
                multiple_people = False
                phone_detected = False
                
                now_ts = _t.time()
                should_run_yolo = (now_ts - last_yolo_time) >= YOLO_INTERVAL

                # Only run YOLO if proctoring is active (technical interview has started)
                # Skip YOLO during identity verification
                if should_run_yolo and self._yolo is not None and self._proctoring_active:
                    try:
                        # Run YOLO with lower confidence for sensitivity; small imgsz for speed
                        results = self._yolo.predict(source=frame, imgsz=480, conf=0.35, iou=0.45, verbose=False)
                        if results and len(results) > 0:
                            r0 = results[0]
                            # Map class indices to names
                            names = r0.names if hasattr(r0, 'names') else getattr(self._yolo, 'names', {})
                            cls = r0.boxes.cls.cpu().numpy().tolist() if hasattr(r0, 'boxes') and r0.boxes is not None else []
                            labels = [str(names.get(int(c), str(int(c)))) for c in cls]
                            person_count = sum(1 for l in labels if l.lower() == 'person')
                            phone_count = sum(1 for l in labels if 'phone' in l.lower())
                            has_person = person_count >= 1
                            multiple_people = person_count >= 2
                            phone_detected = phone_count >= 1
                        last_yolo_time = now_ts
                    except Exception as e:
                        # Fall back to Haar if YOLO fails mid-run
                        print(f"⚠️ YOLO detection error; falling back to Haar: {e}")
                        self._yolo = None

                if self._yolo is None or not should_run_yolo:
                    # Haar-based fallback (always runs, or when YOLO skipped)
                    faces = []
                    try:
                        if face_cascade is not None:
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    except Exception:
                        faces = []
                    face_count = len(faces) if faces is not None else 0
                    # Only update if YOLO didn't run (to preserve YOLO results)
                    if self._yolo is None or not should_run_yolo:
                        has_person = face_count >= 1
                        multiple_people = face_count >= 2
                    # phone_detected remains False in Haar fallback

                # Update warnings from detections with logging - ensure state is always fresh
                prev_no_person = self._last_warning_state.get("no_person", False)
                prev_warning_active = self._last_warning_state.get("no_person_warning_active", False)
                prev_multiple = self._last_warning_state.get("multiple_people", False)
                prev_phone = self._last_warning_state.get("phone_detected", False)
                
                # Debounced absence detection - require continuous absence for threshold duration
                now_abs = _t.time()
                if not has_person:
                    # Person is absent - start/continue tracking absence duration
                    if absence_start_time is None:
                        absence_start_time = now_abs
                    absence_duration = now_abs - absence_start_time
                    
                    # Only activate warning if absent for threshold duration
                    if absence_duration >= ABSENCE_THRESHOLD_SEC:
                        self._last_warning_state["no_person"] = True
                        self._last_warning_state["no_person_warning_active"] = True
                    else:
                        # Still within threshold - don't activate warning yet
                        self._last_warning_state["no_person"] = False
                        self._last_warning_state["no_person_warning_active"] = False
                else:
                    # Person is present - reset absence tracking immediately
                    absence_start_time = None
                    self._last_warning_state["no_person"] = False
                    self._last_warning_state["no_person_warning_active"] = False
                
                # Update other warning states
                self._last_warning_state["multiple_people"] = multiple_people
                self._last_warning_state["phone_detected"] = phone_detected
                
                # Clear warnings when conditions are no longer met
                if has_person and prev_warning_active:
                    # Person returned - clear no_person warnings
                    self._last_warning_logged.pop('no_person', None)
                if not multiple_people and prev_multiple:
                    # Multiple people cleared
                    self._last_warning_state["multiple_people"] = False
                if not phone_detected and prev_phone:
                    # Phone cleared
                    self._last_warning_state["phone_detected"] = False
                
                # Log warnings to database with snapshots when they occur (non-blocking via queue)
                self._log_warning_with_snapshot_async('multiple_people', multiple_people, prev_multiple, frame)
                self._log_warning_with_snapshot_async('phone_detected', phone_detected, prev_phone, frame)
                
                # Handle no_person warning - only log when warning becomes active (not just when person absent)
                current_warning_active = self._last_warning_state.get("no_person_warning_active", False)
                if current_warning_active and not prev_warning_active:
                    # Warning just became active - log it
                    self._log_warning_with_snapshot_async('no_person', True, False, frame, rate_limit_seconds=15)
                elif has_person and prev_warning_active:
                    # Person returned - reset rate limit when person returns
                    self._last_warning_logged.pop('no_person', None)

                # Motion-based detection (low concentration only)
                try:
                    gray_small = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_small = cv2.resize(gray_small, (160, 120))
                    if last_gray is not None:
                        diff = cv2.absdiff(gray_small, last_gray)
                        motion_score = float(diff.mean())
                        now_ts = _t.time()
                        
                        # Low concentration detection
                        prev_low_concentration = self._last_warning_state.get("low_concentration", False)
                        if motion_score > MOTION_LOW_THRESH:
                            last_motion_ts = now_ts
                            # Clear low concentration when motion detected
                            if prev_low_concentration:
                                self._last_warning_state["low_concentration"] = False
                        # Low concentration if no meaningful motion for a window and a person present
                        low_concentration = has_person and (now_ts - last_motion_ts) >= LOW_MOTION_WINDOW_SEC
                        self._last_warning_state["low_concentration"] = low_concentration
                        
                        # Log low concentration warning with snapshot when first detected
                        if low_concentration and not prev_low_concentration:
                            self._log_warning_with_snapshot_async('low_concentration', True, False, frame)
                        # Clear when person moves
                        elif not low_concentration and prev_low_concentration:
                            self._last_warning_state["low_concentration"] = False
                    last_gray = gray_small
                except Exception as e:
                    # Motion detection errors are not critical - continue silently
                    pass
                # Sleep to cap CPU usage - increased to reduce load
                time.sleep(0.2)  # Increased from 0.1s to 0.2s to cut CPU usage in half
            except Exception as e:
                print(f"⚠️ Detection loop error: {e}")
                time.sleep(0.1)

    def get_frame(self) -> bytes:
        """Get a frame as JPEG bytes."""
        try:
            # Check if video is opened - if not, return fallback (don't try to reopen here as detection loop handles it)
            if not self.video.isOpened():
                return self._create_fallback_frame()
            
            frame = None
            # Try to get frame from detection loop buffer first (most efficient)
            # Use non-blocking lock to prevent get_frame() from blocking
            if self._frame_lock is not None:
                try:
                    if self._frame_lock.acquire(blocking=False):
                        try:
                            if self._latest_frame is not None:
                                frame = self._latest_frame.copy()
                        finally:
                            self._frame_lock.release()
                except Exception as e:
                    pass
            
            # If no buffered frame, read directly from camera (fallback)
            if frame is None:
                try:
                    ret, tmp = self.video.read()
                    if ret and tmp is not None and tmp.size > 0:
                        frame = tmp
                        # Also update the buffer if we got a frame (non-blocking)
                        if self._frame_lock is not None:
                            try:
                                if self._frame_lock.acquire(blocking=False):
                                    try:
                                        self._latest_frame = frame.copy()
                                    finally:
                                        self._frame_lock.release()
                            except:
                                pass
                except Exception:
                    pass
            
            # If we have a valid frame, add overlays and encode
            if frame is not None and frame.size > 0:
                try:
                    # Add overlays
                    import time as time_module
                    timestamp = time_module.strftime("%H:%M:%S")
                    cv2.putText(frame, timestamp, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    session_text = f"Session: {str(self.session_id)[:8]}"
                    cv2.putText(frame, session_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Add "LIVE" indicator to show feed is active
                    cv2.rectangle(frame, (frame.shape[1] - 120, 10), (frame.shape[1] - 10, 40), (0, 255, 0), -1)
                    cv2.putText(frame, "LIVE", (frame.shape[1] - 110, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    
                    # Encode as JPEG with lower quality for faster transmission
                    ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok and buf is not None and len(buf) > 0:
                        return buf.tobytes()
                except Exception as e:
                    print(f"[Camera] Error encoding frame: {e}")
                    pass
            
            # Fallback if all else fails
            return self._create_fallback_frame()
        except Exception as e:
            print(f"[Camera] Error in get_frame: {e}")
            return self._create_fallback_frame()

    def _create_fallback_frame(self) -> bytes:
        """Create a fallback frame when camera is not available."""
        # Only print warning every 10 seconds to reduce log spam
        now = time.time()
        if now - self._fallback_print_time > 10:
            print(f"⚠️ Camera not available for session {str(self.session_id)[:8]}... using fallback frames")
            self._fallback_print_time = now
        
        # Cache fallback frame for 1 second to avoid regenerating
        if self._cached_fallback_frame is None or (now - self._fallback_frame_cache_time) > 1.0:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add session info with green background box
            text = f"Session: {str(self.session_id)[:8]}"
            cv2.rectangle(frame, (10, 20), (630, 60), (0, 128, 0), -1)
            cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add status message with background
            if not self.video.isOpened():
                status_text = "Camera Not Available"
                text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                cv2.rectangle(frame, (10, 70), (20 + text_size[0], 110), (0, 0, 128), -1)
                cv2.putText(frame, status_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                status_text = "Camera Loading..."
                text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                cv2.rectangle(frame, (10, 70), (20 + text_size[0], 110), (128, 128, 128), -1)
                cv2.putText(frame, status_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add timestamp with background
            timestamp = time.strftime("%H:%M:%S")
            text_size = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (10, frame.shape[0] - 35), (20 + text_size[0], frame.shape[0] - 5), (64, 64, 64), -1)
            cv2.putText(frame, timestamp, (20, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add a pulsing indicator in the corner to show stream is alive
            import time as time_module
            pulse = int(time_module.time() * 2) % 2
            cv2.circle(frame, (620, 30), 10, (0, 255, 0) if pulse else (0, 128, 0), -1)
            
            ok, buf = cv2.imencode('.jpg', frame)
            if ok and buf is not None and len(buf) > 0:
                self._cached_fallback_frame = buf.tobytes()
                self._fallback_frame_cache_time = now
                return self._cached_fallback_frame
            else:
                # Last resort: return a minimal valid JPEG
                return b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x11\x08\x01\xe0\x02\x80\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xaa\xff\xd9'
        else:
            # Return cached fallback frame
            return self._cached_fallback_frame

    def _log_warning_with_snapshot_async(self, warning_type, current_state, prev_state, frame, rate_limit_seconds=2):
        """Log warning to database with snapshot capture (non-blocking via background thread).
        
        Args:
            warning_type: Type of warning (e.g., 'multiple_people', 'phone_detected')
            current_state: Current warning state (True/False)
            prev_state: Previous warning state (True/False)
            frame: Current camera frame (numpy array)
            rate_limit_seconds: Minimum seconds between logs for same warning (default 2)
        """
        # Only log warnings if proctoring is active (technical interview has started)
        # Skip warnings during identity verification
        if not self._proctoring_active:
            return
        
        # Only log when warning first becomes active (transitions from False to True)
        if not current_state or prev_state:
            return
        
        # Rate limiting to avoid spam - increased for better performance
        now = time.time()
        last_logged = self._last_warning_logged.get(warning_type, 0)
        # Increased default rate limit to 10 seconds for most warnings, 15 seconds for no_person
        effective_rate_limit = 15 if warning_type == 'no_person' else (10 if rate_limit_seconds < 10 else rate_limit_seconds)
        if now - last_logged < effective_rate_limit:
            return
        
        # Update warning count and timestamp immediately (non-blocking)
        self._warning_counts[warning_type] = self._warning_counts.get(warning_type, 0) + 1
        self._last_warning_logged[warning_type] = now
        
        # Queue warning for background thread (non-blocking I/O)
        try:
            if hasattr(self, '_warning_logger_queue'):
                snapshot_filename = None
                if frame is not None:
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                    session_id_str = str(self.session_id)
                    snapshot_filename = f"{session_id_str}_{warning_type}_{ts}.jpg"
                
                # Copy frame to avoid reference issues
                frame_copy = frame.copy() if frame is not None else None
                
                # Try to put in queue (non-blocking if queue is full)
                try:
                    self._warning_logger_queue.put_nowait((warning_type, frame_copy, snapshot_filename))
                except:
                    # Queue full - skip this warning to avoid blocking
                    pass
        except Exception as e:
            print(f"[Proctoring] Failed to queue warning {warning_type}: {e}")

    def get_latest_warnings(self) -> dict:
        """Get current warning states and include warning counts."""
        warnings = self._last_warning_state.copy()
        # Ensure all expected keys exist with default False
        expected_keys = [
            'no_person_warning_active', 'multiple_people', 'phone_detected',
            'no_person', 'low_concentration', 'tab_switched',
            'excessive_noise', 'multiple_speakers'
        ]
        for key in expected_keys:
            if key not in warnings:
                warnings[key] = False
        # Add warning counts as metadata (can be used for display)
        warnings['_counts'] = self._warning_counts.copy()
        return warnings
    
    def get_warning_counts(self) -> dict:
        """Get warning counts for the session."""
        return self._warning_counts.copy()

    def set_tab_switch_status(self, hidden: bool) -> None:
        """Set tab switch status and log warning with snapshot if tab was switched."""
        prev_tab_switched = self._last_warning_state.get("tab_switched", False)
        self._last_warning_state["tab_switched"] = bool(hidden)
        
        # Log warning when tab is switched (becomes True)
        if hidden and not prev_tab_switched:
            # Get latest frame for snapshot (non-blocking)
            frame = None
            if self._frame_lock is not None:
                try:
                    if self._frame_lock.acquire(blocking=False):
                        try:
                            if self._latest_frame is not None:
                                frame = self._latest_frame.copy()
                        finally:
                            self._frame_lock.release()
                except:
                    pass
            
            if frame is not None:
                self._log_warning_with_snapshot_async('tab_switched', True, False, frame, rate_limit_seconds=3)

    def cleanup(self) -> None:
        """Clean up camera resources."""
        print(f"🧹 Cleaning up simple camera for session {self.session_id}")
        try:
            self._running = False
            # Join thread if exists
            t = getattr(self, '_detector_thread', None)
            if t and t.is_alive():
                t.join(timeout=1.5)
        except Exception:
            pass
        if self.video:
            self.video.release()
        print(f"✅ Simple camera cleanup completed for session {self.session_id}")
