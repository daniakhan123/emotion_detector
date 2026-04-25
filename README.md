# 😎 Real-Time Emotion Detection — CNN + OpenCV

Detects **7 emotions** from faces in real-time using a Convolutional Neural Network trained on FER-2013.

| Emotion   | Label    |
|-----------|----------|
| 😡 Angry   | `Angry`   |
| 🤢 Disgust | `Disgust` |
| 😨 Fear    | `Fear`    |
| 😄 Happy   | `Happy`   |
| 😢 Sad     | `Sad`     |
| 😲 Surprise| `Surprise`|
| 😐 Neutral | `Neutral` |

---

## Tech Stack

| Component      | Tool                     |
|----------------|--------------------------|
| Model          | CNN (Keras / TensorFlow) |
| Face Detection | OpenCV Haar Cascade      |
| Dataset        | FER-2013 (35,887 images) |
| Language       | Python 3.10+             |

---

## Project Structure

```
emotion_detection/
├── models/
│   └── emotion_cnn.py      # CNN architecture
├── utils/
│   └── data_loader.py      # FER-2013 data pipeline + augmentation
├── train.py                # Training script
├── detect.py               # Real-time webcam / video / image inference
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Clone / download this project
cd emotion_detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download FER-2013 dataset from Kaggle
#    https://www.kaggle.com/datasets/msambare/fer2013
#    Extract to:  data/train/  and  data/test/
```

---

## Training

```bash
python train.py
```

This will:
- Load FER-2013 with augmentation
- Apply class weights (handles dataset imbalance)
- Use EarlyStopping + ReduceLROnPlateau callbacks
- Save best model → `emotion_model.h5`
- Plot accuracy/loss curves → `training_history.png`
- Print per-class classification report

Expected accuracy: **~63–67%** (FER-2013 human accuracy is ~65%)

---

## Inference

### Webcam (live)
```bash
python detect.py
```

### Video file
```bash
python detect.py --source path/to/video.mp4
```

### Single image
```bash
python detect.py --source path/to/photo.jpg
```

### Controls
| Key | Action         |
|-----|----------------|
| `Q` | Quit           |
| `S` | Save screenshot|
| `P` | Pause / Resume |

---

## Model Architecture

```
Input (48×48×1)
    │
    ├─ Conv2D(32) × 2 → BN → ReLU → MaxPool → Dropout(0.25)
    ├─ Conv2D(64) × 2 → BN → ReLU → MaxPool → Dropout(0.25)
    ├─ Conv2D(128)× 2 → BN → ReLU → MaxPool → Dropout(0.35)
    ├─ Conv2D(256)× 2 → BN → ReLU → GlobalAvgPool → Dropout(0.5)
    │
    ├─ Dense(512) → BN → Dropout(0.5)
    └─ Dense(7, softmax)
```

Total parameters: ~3.2M

---

## Tips to Improve Accuracy

- Use **MobileNetV2** or **EfficientNet-B0** as backbone (transfer learning)
- Add **face alignment** before cropping (dlib landmarks)
- Train on **AffectNet** or **RAF-DB** for more diverse data
- Use **label smoothing** (0.1) to reduce overconfidence

---

## License
MIT — free to use, modify, and showcase on your portfolio.
