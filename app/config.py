"""Configuración y estado compartido de la aplicación."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Extensiones de imagen válidas
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Defaults
DEFAULT_IMG_SIZE = 128
DEFAULT_BATCH_SIZE = 32
DEFAULT_SPLIT_RATIO = 0.8

# Estado compartido entre tabs (single-user, local)
state = {
    "model": None,
    "class_names": [],
    "img_size": DEFAULT_IMG_SIZE,
    "history": None,
    "dataset_dir": "",
    "split_dir": "",
    "preprocessing": "rescale",  # 'rescale' (CNN custom) | 'mobilenet' (MobileNetV2)
    "model_type": "",
}
