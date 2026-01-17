import cv2
import numpy as np
from PIL import Image
import tempfile

def draw_bboxes(image_np, predictions):
    """Draw bounding boxes on image for given predictions"""
    image_cv = image_np.copy()
    for det in predictions:
        cls = det.get("class", "")
        conf = det.get("confidence", 0)
        x, y, w, h = det["x"], det["y"], det["width"], det["height"]
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        color = (46, 134, 193)  # blue
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_cv, f"{cls} ({conf:.2f})", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return image_cv

def process_video(video_path, model, threshold=0.5, st_progress=None, stop_flag=lambda: True):
    """Process video frame by frame and return processed video + timeline data"""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(tmp_out.name, fourcc, fps, (width, height))
    
    frame_idx = 0
    timeline = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or not stop_flag():
            break

        # Convert frame to PIL for Roboflow
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        frame_pil.save(frame_tmp.name)

        # Prediction
        try:
            results = model.predict(frame_tmp.name, confidence=threshold).json()
            predictions = results.get("predictions", [])
        except Exception as e:
            predictions = []

        frame = draw_bboxes(frame, predictions)

        # Timeline: 1 if any player has ball
        has_ball = 1 if any(p['class'] == 'has_ball' for p in predictions) else 0
        timeline.append({"frame": frame_idx, "has_ball": has_ball})

        out.write(frame)
        frame_idx += 1

        if st_progress:
            st_progress.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    out.release()
    return tmp_out.name, timeline
