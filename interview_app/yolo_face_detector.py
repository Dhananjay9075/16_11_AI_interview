
# interview_app/yolo_face_detector.py
from ultralytics import YOLO

# Load the YOLOv8 model specialized for face detection.
# You can also use the generic 'yolov8n.pt' if a specialized one isn't available,
# but a face-specific model is generally better.
# Let's use the generic one for simplicity as it's already part of the project.
model = YOLO('yolov8n.pt')

def detect_face_with_yolo(image_input):
    """
    Takes a file path or numpy array and returns YOLO detection results.
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    else:
        img = image_input

    if img is None:
        raise ValueError("Image not found or invalid format.")

    results = model(img)
    return results