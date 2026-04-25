"""
Emotion Detection CNN Model
Architecture: 4 Conv blocks + Dense layers
Trained on FER-2013 dataset (48x48 grayscale)
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Dropout, Dense, Flatten, GlobalAveragePooling2D
)
from tensorflow.keras.regularizers import l2
import os


EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
IMG_SIZE  = 48
N_CLASSES = len(EMOTIONS)


def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), n_classes=N_CLASSES):
    """
    CNN architecture:
      Block 1-2 : 32/64 filters, 3x3 conv → BN → ReLU → MaxPool → Dropout
      Block 3-4 : 128/256 filters, same pattern
      Head      : GlobalAvgPool → Dense(512) → Dense(n_classes, softmax)
    """
    model = Sequential([
        # ── Block 1 ──────────────────────────────────────────
        Conv2D(32, (3,3), padding='same', kernel_regularizer=l2(1e-4),
               input_shape=input_shape),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Conv2D(32, (3,3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # ── Block 2 ──────────────────────────────────────────
        Conv2D(64, (3,3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Conv2D(64, (3,3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # ── Block 3 ──────────────────────────────────────────
        Conv2D(128, (3,3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Conv2D(128, (3,3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        MaxPooling2D(2, 2),
        Dropout(0.35),

        # ── Block 4 ──────────────────────────────────────────
        Conv2D(256, (3,3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Conv2D(256, (3,3), padding='same', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        GlobalAveragePooling2D(),
        Dropout(0.5),

        # ── Classification head ───────────────────────────────
        Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(n_classes, activation='softmax'),
    ], name='EmotionCNN')

    return model


def compile_model(model, learning_rate=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def load_trained_model(path='emotion_model.h5'):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'.\n"
            "Run train.py first to train and save the model."
        )
    return load_model(path)


if __name__ == '__main__':
    model = build_model()
    compile_model(model)
    model.summary()
