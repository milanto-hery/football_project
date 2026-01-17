import cv2
import numpy as np
import streamlit as st

def draw_bboxes(image_np, predictions):
    """Draw bounding boxes filtered by confidence."""
    for pred in predictions:
        x1, y1 = int(pred['x']), int(pred['y'])
        x2, y2 = int(pred['x'] + pred['width']), int(pred['y'] + pred['height'])
        label = pred['class']
        conf = pred['confidence']
        color = (0,255,0) if label=="has_ball" else (255,0,0)
        cv2.rectangle(image_np, (x1,y1), (x2,y2), color, 2)
        cv2.putText(image_np, f"{label} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

def process_video(video_path, model, threshold=0.5, st_progress=None, stop_flag=None):
    """Process video frame by frame with stop/start functionality."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter("processed_video.mp4",
                          cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
    timeline_data = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if stop_flag and not stop_flag():
            break

        results = model.predict(frame)
        predictions = [p for p in results.get("predictions", []) if p['confidence'] >= threshold]

        frame_out = draw_bboxes(frame, predictions)
        out.write(cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR))

        has_ball_flag = 1 if any(p['class']=="has_ball" for p in predictions) else 0
        timeline_data.append({"frame": frame_idx, "has_ball": has_ball_flag})

        if st_progress:
            st_progress.progress(min(frame_idx/total_frames, 1.0))

        frame_idx += 1

    cap.release()
    out.release()
    return "processed_video.mp4", timeline_data
