"""
Data loading & augmentation for FER-2013
Dataset: https://www.kaggle.com/datasets/msambare/fer2013
Expected folder structure after download:
    data/
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── sad/
    │   ├── surprise/
    │   └── neutral/
    └── test/
        └── ... (same structure)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


IMG_SIZE   = 48
BATCH_SIZE = 64


def get_generators(data_dir='data', batch_size=BATCH_SIZE):
    """
    Returns (train_gen, val_gen) using ImageDataGenerator.
    Training uses augmentation; validation uses only rescaling.
    """

    # ── Training augmentation ─────────────────────────────────
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        f'{data_dir}/train',
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        f'{data_dir}/test',
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
    )

    return train_gen, val_gen


def get_class_weights(train_gen):
    """
    Computes inverse-frequency class weights to handle FER-2013's
    imbalance (Disgust has ~5x fewer samples than Happy).
    """
    from sklearn.utils.class_weight import compute_class_weight

    labels = train_gen.classes
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return dict(enumerate(weights))


def preview_batch(train_gen, n=16):
    """Visualise a batch — useful sanity check before training."""
    label_map = {v: k for k, v in train_gen.class_indices.items()}
    imgs, labels = next(train_gen)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle('Sample Training Batch', fontsize=14)
    for i, ax in enumerate(axes.flat):
        if i >= n:
            break
        ax.imshow(imgs[i].squeeze(), cmap='gray')
        ax.set_title(label_map[labels[i].argmax()], fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('batch_preview.png', dpi=100)
    plt.show()


if __name__ == '__main__':
    train_gen, val_gen = get_generators()
    print(f"Training samples  : {train_gen.n}")
    print(f"Validation samples: {val_gen.n}")
    print(f"Classes           : {train_gen.class_indices}")
    preview_batch(train_gen)
