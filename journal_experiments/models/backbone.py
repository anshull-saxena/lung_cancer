"""
CNN backbone feature extractors with pluggable attention mechanisms.
"""
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet121, ResNet50, VGG16, EfficientNetB0
from tensorflow.keras.applications.densenet import preprocess_input as pre_densenet
from tensorflow.keras.applications.resnet import preprocess_input as pre_resnet
from tensorflow.keras.applications.vgg16 import preprocess_input as pre_vgg
from tensorflow.keras.applications.efficientnet import preprocess_input as pre_effnet

from .attention import get_attention_layer

# Add parent dir to path so config is importable when run as a package
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in __import__('sys').path:
    __import__('sys').path.insert(0, _parent)
from config import CACHE_DIR, IMG_SIZE

# Backbone registry: name → (keras_app, preprocess_fn, feature_dim)
BACKBONE_REGISTRY = {
    "densenet121": (DenseNet121, pre_densenet, 1024),
    "resnet50":    (ResNet50,    pre_resnet,   2048),
    "vgg16":       (VGG16,       pre_vgg,       512),
    "efficientnetb0": (EfficientNetB0, pre_effnet, 1280),
}


def build_feature_extractor(backbone_name="densenet121", attention_type="se",
                             img_size=IMG_SIZE):
    """
    Build a feature extractor: backbone + attention + GAP.

    Args:
        backbone_name: one of 'densenet121', 'resnet50', 'vgg16', 'efficientnetb0'
        attention_type: one of 'se', 'eca', 'cbam', 'split', 'dual', 'vit', 'swin'
        img_size: (H, W) input size

    Returns:
        model: Keras Model outputting 1-D feature vector
        feature_dim: int, dimensionality of output
    """
    if backbone_name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone '{backbone_name}'. "
                         f"Choose from {list(BACKBONE_REGISTRY.keys())}")

    backbone_cls, preprocess_fn, _nominal_dim = BACKBONE_REGISTRY[backbone_name]

    inp = layers.Input(shape=(*img_size, 3), name="input_image")
    x = layers.Lambda(preprocess_fn, name=f"preprocess_{backbone_name}")(inp)
    base_model = backbone_cls(include_top=False, weights="imagenet",
                              input_tensor=x)
    # Freeze backbone weights
    base_model.trainable = False
    feat_map = base_model.output  # (B, H', W', C)

    # Attention
    attn_layer = get_attention_layer(attention_type)
    feat_map = attn_layer(feat_map)

    # Global Average Pooling → 1-D vector
    out = layers.GlobalAveragePooling2D(name="gap_final")(feat_map)

    model = Model(inp, out, name=f"{backbone_name}_{attention_type}")
    # Derive actual feature dim from the built model
    feature_dim = int(model.output_shape[-1])
    return model, feature_dim


def extract_features(model, X, batch_size=16):
    """
    Extract features from a numpy array of images.

    Args:
        model: Keras feature extractor model
        X: numpy array of shape (N, H, W, 3), pixel values [0, 255]
        batch_size: prediction batch size

    Returns:
        features: numpy array (N, feature_dim)
    """
    n = len(X)
    feats = []
    pbar = tqdm(range(0, n, batch_size), desc="  Extracting", unit="batch", leave=False)
    for start in pbar:
        end = min(start + batch_size, n)
        batch = X[start:end]
        if batch.dtype != np.float32:
            batch = batch.astype(np.float32, copy=False)
        pred = model.predict(batch, verbose=0)
        feats.append(pred)
    return np.vstack(feats)


def extract_features_from_paths(model, paths, img_size=IMG_SIZE, batch_size=16):
    """Extract features by streaming images from disk.

    Args:
        model: Keras feature extractor model
        paths: array-like of image file paths
        img_size: (H, W)
        batch_size: prediction batch size

    Returns:
        features: numpy array (N, feature_dim)
    """
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    paths = list(paths)
    n = len(paths)
    feats = []
    pbar = tqdm(range(0, n, batch_size), desc="  Extracting", unit="batch", leave=False)
    for start in pbar:
        end = min(start + batch_size, n)
        batch_paths = paths[start:end]
        batch_imgs = []
        for p in batch_paths:
            try:
                img = load_img(p, target_size=img_size)
                batch_imgs.append(img_to_array(img))
            except Exception as e:
                raise RuntimeError(f"Failed to load image: {p} ({e})") from e

        batch = np.asarray(batch_imgs, dtype=np.float32)
        pred = model.predict(batch, verbose=0)
        feats.append(pred)
    return np.vstack(feats)


def extract_features_cached(model, X, cache_name, force=False, batch_size=16):
    """
    Extract features with disk caching (.npy files).

    Args:
        model: Keras feature extractor
        X: image array
        cache_name: string identifier for cache file
        force: if True, re-extract even if cache exists

    Returns:
        features: numpy array
    """
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.npy")
    if os.path.exists(cache_path) and not force:
        print(f"  Loading cached features: {cache_path}")
        return np.load(cache_path)
    print(f"  Extracting features ({cache_name}) ...")
    feats = extract_features(model, X, batch_size=batch_size)
    np.save(cache_path, feats)
    print(f"  Cached to {cache_path}")
    return feats


def extract_features_from_paths_cached(
    model,
    paths,
    cache_name,
    *,
    img_size=IMG_SIZE,
    force=False,
    batch_size=16,
):
    """Extract features from disk paths with caching (.npy)."""
    cache_path = os.path.join(CACHE_DIR, f"{cache_name}.npy")
    if os.path.exists(cache_path) and not force:
        print(f"  Loading cached features: {cache_path}")
        return np.load(cache_path)
    print(f"  Extracting features ({cache_name}) ...")
    feats = extract_features_from_paths(model, paths, img_size=img_size, batch_size=batch_size)
    np.save(cache_path, feats)
    print(f"  Cached to {cache_path}")
    return feats
