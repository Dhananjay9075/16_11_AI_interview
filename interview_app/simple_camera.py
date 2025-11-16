import time
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class _VideoStub:
    def isOpened(self) -> bool:
        return True


class SimpleVideoCamera:
    """Minimal camera stub that generates a placeholder JPEG stream.

    Provides the attributes/methods expected by the views:
    - .video.isOpened()
    - get_frame() -> bytes (JPEG)
    - get_latest_warnings() -> dict
    - set_tab_switch_status(hidden: bool)
    - cleanup()
    """

    def __init__(self, session_id):
        self.session_id = session_id
        self.video = _VideoStub()
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

    def get_frame(self) -> bytes:
        # Create a simple 480x640 black frame with overlay text
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        text = f"Session: {str(self.session_id)[:8]}"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, time.strftime("%H:%M:%S"), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ok, buf = cv2.imencode('.jpg', frame)
        return buf.tobytes() if ok else b''

    def get_latest_warnings(self) -> dict:
        return self._last_warning_state

    def set_tab_switch_status(self, hidden: bool) -> None:
        self._last_warning_state["tab_switched"] = bool(hidden)

    def cleanup(self) -> None:
        # Nothing to release in the stub
        pass


