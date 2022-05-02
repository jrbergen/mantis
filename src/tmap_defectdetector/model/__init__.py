from __future__ import annotations
import tensorflow as tf

try:
    GPU_AVAILABLE: bool = tf.test.is_built_with_cuda() and tf.config.list_physical_devices("GPU")
except Exception:
    GPU_AVAILABLE: bool = False

DEFAULT_GPU: str = "gpu:0"
