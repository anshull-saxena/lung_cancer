"""
Dataset loading, splitting, and augmentation for LC25000 lung dataset.
"""
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, StratifiedKFold

from config import CLASS_NAMES, IMG_SIZE, SEED, BATCH_SIZE


def load_dataset(data_dir, img_size=IMG_SIZE):
    """
    Load all images from disk into numpy arrays.

    Args:
        data_dir: path containing subfolders lung_aca, lung_n, lung_scc
        img_size: tuple (H, W) for resizing

    Returns:
        X: np.array of shape (N, H, W, 3), dtype float32, pixel values [0, 255]
        y: np.array of integer labels (N,)
    """
    images, labels = [], []
    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        filenames = sorted(os.listdir(class_dir))
        print(f"  Loading {class_name}: {len(filenames)} images ...")
        for fname in filenames:
            fpath = os.path.join(class_dir, fname)
            try:
                img = load_img(fpath, target_size=img_size)
                images.append(img_to_array(img))
                labels.append(label_idx)
            except Exception as e:
                print(f"  Warning: skipping {fpath}: {e}")
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    print(f"  Dataset loaded: {X.shape}, classes={np.bincount(y)}")
    return X, y


def load_dataset_paths(data_dir):
    """Load dataset as file paths (no image pixels loaded).

    This is a low-memory alternative to `load_dataset()`.

    Args:
        data_dir: path containing subfolders lung_aca, lung_n, lung_scc

    Returns:
        paths: np.array of shape (N,), dtype=str
        y: np.array of integer labels (N,)
    """
    paths, labels = [], []
    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        filenames = sorted(os.listdir(class_dir))
        print(f"  Indexing {class_name}: {len(filenames)} images ...")
        for fname in filenames:
            paths.append(os.path.join(class_dir, fname))
            labels.append(label_idx)

    paths = np.array(paths, dtype=object)
    y = np.array(labels, dtype=np.int32)
    print(f"  Dataset indexed: ({len(paths)},), classes={np.bincount(y)}")
    return paths, y


def get_splits(X, y, train_ratio=0.70, val_ratio=0.10, test_ratio=0.20, seed=SEED):
    """
    Stratified train/val/test split.

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    # First split: separate test set
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=seed, stratify=y
    )
    # Second split: separate val from train
    val_frac = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_frac, random_state=seed, stratify=y_rest
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def get_kfold_splits(X, y, n_folds=10, seed=SEED):
    """
    Generator yielding (train_indices, test_indices) for stratified k-fold.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(X, y):
        yield train_idx, test_idx


def get_augmented_generator(X, y, batch_size=BATCH_SIZE):
    """
    Returns a Keras ImageDataGenerator flow with standard augmentation.
    Input X is expected to have pixel values in [0, 255].
    """
    datagen = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        rescale=1.0 / 255.0,
    )
    return datagen.flow(X, y, batch_size=batch_size, seed=SEED, shuffle=True)
