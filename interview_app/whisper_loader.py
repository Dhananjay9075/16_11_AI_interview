import whisper
import threading

# Global Whisper model instance
_whisper_model = None
_whisper_lock = threading.Lock()

def get_whisper_model():
    """
    Get or create the Whisper model instance (singleton pattern).
    This prevents loading the model multiple times.
    """
    global _whisper_model
    
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:  # Double-check locking
                try:
                    _whisper_model = whisper.load_model("small")
                    print("Whisper model 'small' loaded (singleton).")
                except Exception as e:
                    print(f"Error loading Whisper model: {e}")
                    _whisper_model = None
    
    return _whisper_model

def is_whisper_available():
    """
    Check if Whisper model is available.
    """
    return get_whisper_model() is not None

