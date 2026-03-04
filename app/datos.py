"""Utilidades de dataset: carga, split, augmentation, preprocesamiento."""

import os
import random
import shutil
import time

import numpy as np
from PIL import Image

from app.config import IMG_EXTENSIONS, DEFAULT_SPLIT_RATIO


def scan_dataset(path):
    """Escanea directorio con subcarpetas por clase.

    Returns:
        dict con 'classes', 'counts', 'total', 'path' — o None si no es válido.
    """
    if not os.path.isdir(path):
        return None

    clases = sorted([
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d)) and not d.startswith(".")
    ])
    if len(clases) < 2:
        return None

    counts = {}
    for c in clases:
        cdir = os.path.join(path, c)
        imgs = [f for f in os.listdir(cdir) if f.lower().endswith(IMG_EXTENSIONS)]
        counts[c] = len(imgs)

    return {
        "classes": clases,
        "counts": counts,
        "total": sum(counts.values()),
        "path": path,
    }


def split_dataset(src_dir, dst_dir, class_names, ratio=DEFAULT_SPLIT_RATIO):
    """Divide dataset en train/validation con seed fijo."""
    if os.path.exists(os.path.join(dst_dir, "train")):
        return dst_dir

    rng = random.Random(42)
    for c in class_names:
        imgs = sorted([
            f for f in os.listdir(os.path.join(src_dir, c))
            if f.lower().endswith(IMG_EXTENSIONS)
        ])
        rng.shuffle(imgs)
        n = int(len(imgs) * ratio)
        for sub, lst in [("train", imgs[:n]), ("validation", imgs[n:])]:
            d = os.path.join(dst_dir, sub, c)
            os.makedirs(d, exist_ok=True)
            for f in lst:
                dst = os.path.join(d, f)
                if not os.path.exists(dst):
                    shutil.copy(os.path.join(src_dir, c, f), dst)
    return dst_dir


def create_generators(split_dir, img_size, batch_size, preprocessing="rescale",
                      aug_params=None):
    """Crea generadores de train y validation.

    Args:
        preprocessing: 'rescale' (divide entre 255) o 'mobilenet' (escala a [-1,1]).
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    aug = aug_params or {}

    if preprocessing == "mobilenet":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=aug.get("rotation", 20),
            width_shift_range=aug.get("shift", 0.1),
            height_shift_range=aug.get("shift", 0.1),
            brightness_range=aug.get("brightness", [0.8, 1.2]),
            zoom_range=aug.get("zoom", 0.1),
            fill_mode="nearest",
        )
        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    else:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=aug.get("rotation", 20),
            width_shift_range=aug.get("shift", 0.1),
            height_shift_range=aug.get("shift", 0.1),
            brightness_range=aug.get("brightness", [0.8, 1.2]),
            zoom_range=aug.get("zoom", 0.1),
            fill_mode="nearest",
        )
        val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train = train_datagen.flow_from_directory(
        os.path.join(split_dir, "train"),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )
    val = val_datagen.flow_from_directory(
        os.path.join(split_dir, "validation"),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return train, val


def save_webcam_image(image_array, class_name, output_dir):
    """Guarda imagen de webcam en subcarpeta de clase.

    Returns:
        (filepath, count) — ruta guardada y total de imágenes en la clase.
    """
    cdir = os.path.join(output_dir, class_name)
    os.makedirs(cdir, exist_ok=True)
    # Usar timestamp para evitar colisiones al borrar y re-capturar
    ts = int(time.time() * 1000) % 10_000_000
    existing = len([
        f for f in os.listdir(cdir) if f.lower().endswith(IMG_EXTENSIONS)
    ])
    fname = f"{class_name}_{ts}_{existing:04d}.jpg"
    path = os.path.join(cdir, fname)

    if isinstance(image_array, np.ndarray):
        Image.fromarray(image_array).save(path)
    else:
        image_array.save(path)
    return path, existing + 1


def list_class_images(output_dir, class_name):
    """Lista rutas de imágenes de una clase (para galería)."""
    cdir = os.path.join(output_dir, class_name)
    if not os.path.isdir(cdir):
        return []
    return sorted([
        os.path.join(cdir, f)
        for f in os.listdir(cdir)
        if f.lower().endswith(IMG_EXTENSIONS)
    ])


def delete_image(filepath):
    """Elimina una imagen del disco."""
    if os.path.isfile(filepath):
        os.remove(filepath)
        return True
    return False


def delete_class_images(output_dir, class_name):
    """Elimina todas las imágenes de una clase."""
    cdir = os.path.join(output_dir, class_name)
    if os.path.isdir(cdir):
        shutil.rmtree(cdir)
        os.makedirs(cdir, exist_ok=True)


def preprocess_image(image, img_size, preprocessing="rescale"):
    """Preprocesa imagen para predicción.

    Args:
        image: numpy array (RGB, uint8) o PIL Image.
        img_size: tamaño target.
        preprocessing: 'rescale' o 'mobilenet'.

    Returns:
        numpy array con shape (1, img_size, img_size, 3), listo para predict().
    """
    if isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        img = image
    img = img.convert("RGB").resize((img_size, img_size))
    arr = np.array(img).astype("float32")

    if preprocessing == "mobilenet":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        arr = preprocess_input(arr)
    else:
        arr = arr / 255.0

    return np.expand_dims(arr, axis=0)
