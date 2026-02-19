"""
Seven attention mechanisms implemented as Keras layers/functions.
Each takes a 4-D tensor (batch, H, W, C) and returns an attention-weighted tensor
of the same shape.
"""
import tensorflow as tf
from tensorflow.keras import layers
import math


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SE Block (Squeeze-and-Excitation)
# ═══════════════════════════════════════════════════════════════════════════════
class SEBlock(layers.Layer):
    """Squeeze-and-Excitation: GAP+GMP → shared Dense MLP → sigmoid → scale."""

    def __init__(self, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        ch = input_shape[-1]
        mid = max(ch // self.reduction, 1)
        self.gap = layers.GlobalAveragePooling2D()
        self.gmp = layers.GlobalMaxPooling2D()
        self.fc1 = layers.Dense(mid, activation="relu")
        self.fc2 = layers.Dense(ch)
        self.reshape = layers.Reshape((1, 1, ch))

    def call(self, x):
        g = self.fc2(self.fc1(self.gap(x)))
        m = self.fc2(self.fc1(self.gmp(x)))
        scale = tf.nn.sigmoid(g + m)
        scale = self.reshape(scale)
        return x * scale

    def get_config(self):
        return {**super().get_config(), "reduction": self.reduction}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ECA (Efficient Channel Attention)
# ═══════════════════════════════════════════════════════════════════════════════
class ECABlock(layers.Layer):
    """GAP → adaptive 1-D conv → sigmoid → channel multiply."""

    def __init__(self, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        ch = input_shape[-1]
        self.gap = layers.GlobalAveragePooling2D()
        self.conv1d = layers.Conv1D(
            1, kernel_size=self.kernel_size,
            padding="same", use_bias=False
        )
        self.reshape = layers.Reshape((1, 1, ch))

    def call(self, x):
        g = self.gap(x)                        # (B, C)
        g = tf.expand_dims(g, axis=-1)          # (B, C, 1)
        g = self.conv1d(g)                      # (B, C, 1)
        g = tf.squeeze(g, axis=-1)              # (B, C)
        scale = tf.nn.sigmoid(g)
        scale = self.reshape(scale)
        return x * scale

    def get_config(self):
        return {**super().get_config(), "kernel_size": self.kernel_size}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CBAM (Convolutional Block Attention Module)
# ═══════════════════════════════════════════════════════════════════════════════
class CBAMBlock(layers.Layer):
    """Channel attention (SE-style) + spatial attention (7×7 conv)."""

    def __init__(self, reduction=16, spatial_kernel=7, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        self.spatial_kernel = spatial_kernel

    def build(self, input_shape):
        ch = input_shape[-1]
        mid = max(ch // self.reduction, 1)
        # Channel attention
        self.gap = layers.GlobalAveragePooling2D()
        self.gmp = layers.GlobalMaxPooling2D()
        self.fc1 = layers.Dense(mid, activation="relu")
        self.fc2 = layers.Dense(ch)
        self.ch_reshape = layers.Reshape((1, 1, ch))
        # Spatial attention
        self.spatial_conv = layers.Conv2D(
            1, kernel_size=self.spatial_kernel, padding="same", activation="sigmoid"
        )

    def call(self, x):
        # Channel attention
        g = self.fc2(self.fc1(self.gap(x)))
        m = self.fc2(self.fc1(self.gmp(x)))
        ch_scale = tf.nn.sigmoid(g + m)
        ch_scale = self.ch_reshape(ch_scale)
        x = x * ch_scale

        # Spatial attention
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        sp = tf.concat([avg_pool, max_pool], axis=-1)
        sp_scale = self.spatial_conv(sp)
        return x * sp_scale

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"reduction": self.reduction, "spatial_kernel": self.spatial_kernel})
        return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Split-Attention (ResNeSt-style)
# ═══════════════════════════════════════════════════════════════════════════════
class SplitAttention(layers.Layer):
    """Split input channels into groups → per-group attention → weighted sum."""

    def __init__(self, num_splits=4, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.num_splits = num_splits
        self.reduction = reduction

    def build(self, input_shape):
        ch = input_shape[-1]
        self.group_ch = ch // self.num_splits
        mid = max(ch // self.reduction, 1)
        self.gap = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(mid, activation="relu")
        self.fcs = [layers.Dense(self.group_ch) for _ in range(self.num_splits)]

    def call(self, x):
        splits = tf.split(x, self.num_splits, axis=-1)
        combined = tf.add_n(splits)
        g = self.gap(combined)
        g = self.fc1(g)
        attns = [tf.nn.softmax(fc(g), axis=-1) for fc in self.fcs]
        out = []
        for s, a in zip(splits, attns):
            a = tf.reshape(a, (-1, 1, 1, self.group_ch))
            out.append(s * a)
        return tf.concat(out, axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_splits": self.num_splits, "reduction": self.reduction})
        return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Dual Attention (Channel + Position)
# ═══════════════════════════════════════════════════════════════════════════════
class DualAttention(layers.Layer):
    """Channel attention branch + position/spatial attention branch → sum."""

    def __init__(self, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction

    def build(self, input_shape):
        ch = input_shape[-1]
        mid = max(ch // self.reduction, 1)
        # Channel attention branch
        self.gap_c = layers.GlobalAveragePooling2D()
        self.fc1_c = layers.Dense(mid, activation="relu")
        self.fc2_c = layers.Dense(ch)
        self.reshape_c = layers.Reshape((1, 1, ch))
        # Position (spatial) attention branch
        self.conv_pos = layers.Conv2D(ch, 1, padding="same")
        self.bn_pos = layers.BatchNormalization()

    def call(self, x):
        # Channel branch
        g = self.fc2_c(self.fc1_c(self.gap_c(x)))
        ch_w = tf.nn.sigmoid(g)
        ch_w = self.reshape_c(ch_w)
        ch_out = x * ch_w

        # Position branch — spatial softmax
        sp = self.bn_pos(self.conv_pos(x))
        b, h, w, c = tf.unstack(tf.shape(sp))
        sp_flat = tf.reshape(sp, (b, h * w, c))
        sp_attn = tf.nn.softmax(sp_flat, axis=1)
        sp_attn = tf.reshape(sp_attn, (b, h, w, c))
        sp_out = x * sp_attn

        return ch_out + sp_out

    def get_config(self):
        return {**super().get_config(), "reduction": self.reduction}


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ViT Self-Attention
# ═══════════════════════════════════════════════════════════════════════════════
class ViTSelfAttention(layers.Layer):
    """Reshape spatial dims to sequence → multi-head self-attention → reshape back."""

    def __init__(self, num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        ch = input_shape[-1]
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=ch // self.num_heads
        )
        self.ln = layers.LayerNormalization()

    def call(self, x):
        b = tf.shape(x)[0]
        h, w, c = x.shape[1], x.shape[2], x.shape[3]
        seq = tf.reshape(x, (b, h * w, c))
        attn_out = self.mha(seq, seq)
        attn_out = self.ln(attn_out + seq)
        return tf.reshape(attn_out, (b, h, w, c))

    def get_config(self):
        return {**super().get_config(), "num_heads": self.num_heads}


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Swin-style Window Attention
# ═══════════════════════════════════════════════════════════════════════════════
class SwinWindowAttention(layers.Layer):
    """Window-based local self-attention with shifted windows."""

    def __init__(self, window_size=7, num_heads=8, shift=False, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.num_heads = num_heads
        self.shift = shift

    def build(self, input_shape):
        ch = input_shape[-1]
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=ch // self.num_heads
        )
        self.ln = layers.LayerNormalization()

    def _window_partition(self, x, window_size):
        b = tf.shape(x)[0]
        h, w, c = x.shape[1], x.shape[2], x.shape[3]
        nh, nw = h // window_size, w // window_size
        x = tf.reshape(x, (b, nh, window_size, nw, window_size, c))
        x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
        x = tf.reshape(x, (b * nh * nw, window_size * window_size, c))
        return x, nh, nw

    def _window_reverse(self, x, nh, nw, window_size):
        b_windows = tf.shape(x)[0]
        b = b_windows // (nh * nw)
        c = x.shape[-1]
        x = tf.reshape(x, (b, nh, nw, window_size, window_size, c))
        x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
        x = tf.reshape(x, (b, nh * window_size, nw * window_size, c))
        return x

    def call(self, x):
        h, w = x.shape[1], x.shape[2]
        ws = self.window_size
        # Pad if needed so H,W are divisible by window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = tf.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
        h_p, w_p = h + pad_h, w + pad_w

        shift_size = ws // 2 if self.shift else 0
        if shift_size > 0:
            x = tf.roll(x, shift=(-shift_size, -shift_size), axis=(1, 2))

        windows, nh, nw = self._window_partition(x, ws)
        attn_out = self.mha(windows, windows)
        attn_out = self.ln(attn_out + windows)
        x = self._window_reverse(attn_out, nh, nw, ws)

        if shift_size > 0:
            x = tf.roll(x, shift=(shift_size, shift_size), axis=(1, 2))
        if pad_h > 0 or pad_w > 0:
            x = x[:, :h, :w, :]
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "window_size": self.window_size,
            "num_heads": self.num_heads,
            "shift": self.shift,
        })
        return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════
ATTENTION_REGISTRY = {
    "se": SEBlock,
    "eca": ECABlock,
    "cbam": CBAMBlock,
    "split": SplitAttention,
    "dual": DualAttention,
    "vit": ViTSelfAttention,
    "swin": SwinWindowAttention,
}


def get_attention_layer(name, **kwargs):
    """Return an instantiated attention layer by name."""
    name = name.lower()
    if name not in ATTENTION_REGISTRY:
        raise ValueError(
            f"Unknown attention '{name}'. Choose from {list(ATTENTION_REGISTRY.keys())}"
        )
    return ATTENTION_REGISTRY[name](**kwargs)
