"""
train.py — Train the Emotion Detection CNN
------------------------------------------
Usage:
    python train.py

Output:
    emotion_model.h5        ← saved model weights
    training_history.png    ← accuracy / loss curves
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping,
    ReduceLROnPlateau, TensorBoard
)

# local imports
import sys
sys.path.insert(0, os.path.dirname(__file__))
from models.emotion_cnn import build_model, compile_model, EMOTIONS
from utils.data_loader  import get_generators, get_class_weights


# ── Hyper-parameters ──────────────────────────────────────────────────────────
EPOCHS        = 100
BATCH_SIZE    = 64
LEARNING_RATE = 1e-3
DATA_DIR      = 'data'
MODEL_SAVE    = 'emotion_model.h5'


def get_callbacks(model_path=MODEL_SAVE):
    return [
        # Save best val_accuracy checkpoint
        ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),

        # Stop if val_loss doesn't improve for 15 epochs
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),

        # Halve LR when val_loss plateaus for 5 epochs
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),

        # TensorBoard (run: tensorboard --logdir logs/)
        TensorBoard(log_dir='logs/', histogram_freq=1),
    ]


def plot_history(history, save_path='training_history.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training History — Emotion CNN', fontsize=14, fontweight='bold')

    epochs = range(1, len(history.history['accuracy']) + 1)

    # Accuracy
    ax1.plot(epochs, history.history['accuracy'],     label='Train',      color='#6366f1', linewidth=2)
    ax1.plot(epochs, history.history['val_accuracy'], label='Validation', color='#10b981', linewidth=2)
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Loss
    ax2.plot(epochs, history.history['loss'],     label='Train',      color='#6366f1', linewidth=2)
    ax2.plot(epochs, history.history['val_loss'], label='Validation', color='#10b981', linewidth=2)
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Categorical Crossentropy')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[✓] Training history saved → {save_path}")
    plt.show()


def evaluate_model(model, val_gen):
    """Print per-class accuracy on the validation set."""
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns

    val_gen.reset()
    preds = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes

    print("\n── Classification Report ──────────────────────────────")
    print(classification_report(y_true, y_pred, target_names=EMOTIONS))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    print("[✓] Confusion matrix saved → confusion_matrix.png")
    plt.show()


def main():
    print("── GPU Check ──────────────────────────────────────────")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs available: {len(gpus)}" if gpus else "No GPU — using CPU (training will be slow)")

    print("\n── Loading Data ───────────────────────────────────────")
    train_gen, val_gen = get_generators(DATA_DIR, BATCH_SIZE)
    class_weights = get_class_weights(train_gen)
    print(f"Class weights: { {EMOTIONS[k]: round(v,2) for k,v in class_weights.items()} }")

    print("\n── Building Model ─────────────────────────────────────")
    model = build_model()
    model = compile_model(model, LEARNING_RATE)
    model.summary()

    print("\n── Training ───────────────────────────────────────────")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=get_callbacks(),
        verbose=1,
    )

    plot_history(history)
    evaluate_model(model, val_gen)

    print(f"\n[✓] Best model saved → {MODEL_SAVE}")


if __name__ == '__main__':
    main()
