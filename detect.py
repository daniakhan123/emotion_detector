"""
detect.py — Real-time Emotion Detection via Webcam
---------------------------------------------------
Usage:
    python detect.py                        # webcam (default)
    python detect.py --source video.mp4     # video file
    python detect.py --source image.jpg     # single image

Controls:
    Q  →  quit
    S  →  save screenshot
    P  →  pause / resume
"""

import cv2
import numpy as np
import argparse
import time
import os

# local imports
import sys
sys.path.insert(0, os.path.dirname(__file__))
from models.emotion_cnn import load_trained_model, EMOTIONS, IMG_SIZE


# ── Colour palette (BGR) per emotion ──────────────────────────────────────────
EMOTION_COLOURS = {
    'Angry':    (0,   0,   220),
    'Disgust':  (0,   140, 0  ),
    'Fear':     (128, 0,   128),
    'Happy':    (0,   200, 80 ),
    'Sad':      (200, 80,  0  ),
    'Surprise': (0,   200, 255),
    'Neutral':  (180, 180, 180),
}

HAAR_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'


def preprocess_face(face_roi):
    """Resize, normalise, and reshape a face crop for model input."""
    face = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
    face = face.astype('float32') / 255.0
    return face.reshape(1, IMG_SIZE, IMG_SIZE, 1)


def draw_overlay(frame, x, y, w, h, emotion, confidence, colour):
    """Draw bounding box, emotion label, and confidence bar."""

    # ── Bounding box ──────────────────────────────────────────
    cv2.rectangle(frame, (x, y), (x+w, y+h), colour, 2)

    # ── Corner accents ────────────────────────────────────────
    corner = 18
    thickness = 3
    for (cx, cy, dx, dy) in [
        (x,   y,   1,  1),
        (x+w, y,  -1,  1),
        (x,   y+h, 1, -1),
        (x+w, y+h,-1, -1),
    ]:
        cv2.line(frame, (cx, cy), (cx + dx*corner, cy), colour, thickness)
        cv2.line(frame, (cx, cy), (cx, cy + dy*corner), colour, thickness)

    # ── Label background ──────────────────────────────────────
    label = f"{emotion}  {confidence*100:.1f}%"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    label_y = max(y - 10, lh + 10)
    cv2.rectangle(frame,
                  (x, label_y - lh - 8),
                  (x + lw + 12, label_y + 4),
                  colour, cv2.FILLED)
    cv2.putText(frame, label,
                (x + 6, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)

    # ── Confidence bar ────────────────────────────────────────
    bar_x, bar_y = x, y + h + 8
    bar_max_w = w
    bar_h = 6
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + bar_h),
                  (50, 50, 50), cv2.FILLED)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + int(bar_max_w * confidence), bar_y + bar_h),
                  colour, cv2.FILLED)


def draw_hud(frame, fps, face_count, paused):
    """Top-left HUD with FPS, face count, and status."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (220, 80), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    status = "PAUSED" if paused else "LIVE"
    colour = (0, 150, 255) if paused else (0, 220, 80)
    cv2.putText(frame, f"Status : {status}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS    : {fps:.1f}", (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Faces  : {face_count}", (10, 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)


def run_detection(source=0, model_path='emotion_model.h5', save_dir='screenshots'):
    print(f"[*] Loading model from '{model_path}' ...")
    model = load_trained_model(model_path)

    face_cascade = cv2.CascadeClassifier(HAAR_PATH)
    if face_cascade.empty():
        raise RuntimeError("Haar cascade XML not found. Reinstall OpenCV.")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    os.makedirs(save_dir, exist_ok=True)

    print("[*] Starting detection  |  Q=quit  S=screenshot  P=pause")
    paused       = False
    prev_time    = time.time()
    screenshot_n = 0
    frame_buffer = None  # holds last frame while paused

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[!] Stream ended.")
                break
            frame_buffer = frame.copy()
        else:
            frame = frame_buffer.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            inp = preprocess_face(roi)

            preds      = model.predict(inp, verbose=0)[0]
            emotion    = EMOTIONS[np.argmax(preds)]
            confidence = float(np.max(preds))
            colour     = EMOTION_COLOURS[emotion]

            draw_overlay(frame, x, y, w, h, emotion, confidence, colour)

        # FPS
        now  = time.time()
        fps  = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        draw_hud(frame, fps, len(faces), paused)

        cv2.imshow('Emotion Detector', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            path = os.path.join(save_dir, f"screenshot_{screenshot_n:04d}.png")
            cv2.imwrite(path, frame)
            print(f"[✓] Screenshot saved → {path}")
            screenshot_n += 1
        elif key == ord('p'):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


# ── Standalone image mode ─────────────────────────────────────────────────────
def detect_image(image_path, model_path='emotion_model.h5'):
    model = load_trained_model(model_path)
    face_cascade = cv2.CascadeClassifier(HAAR_PATH)

    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))

    results = []
    for (x, y, w, h) in faces:
        roi   = gray[y:y+h, x:x+w]
        inp   = preprocess_face(roi)
        preds = model.predict(inp, verbose=0)[0]
        emotion    = EMOTIONS[np.argmax(preds)]
        confidence = float(np.max(preds))
        colour     = EMOTION_COLOURS[emotion]
        draw_overlay(frame, x, y, w, h, emotion, confidence, colour)
        results.append({'emotion': emotion, 'confidence': confidence,
                        'all_scores': dict(zip(EMOTIONS, preds.tolist()))})

    out_path = image_path.replace('.', '_detected.')
    cv2.imwrite(out_path, frame)
    print(f"[✓] Result saved → {out_path}")
    for i, r in enumerate(results):
        print(f"  Face {i+1}: {r['emotion']}  ({r['confidence']*100:.1f}%)")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time Emotion Detector')
    parser.add_argument('--source',      default='0',
                        help="0 = webcam, path = video/image file")
    parser.add_argument('--model',       default='emotion_model.h5')
    parser.add_argument('--screenshots', default='screenshots')
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source

    # Auto-detect image mode
    if isinstance(src, str) and src.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
        detect_image(src, args.model)
    else:
        run_detection(src, args.model, args.screenshots)
