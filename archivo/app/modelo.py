"""Constructores de modelos: MobileNetV2 (transfer learning) y CNN custom."""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def build_mobilenetv2(num_classes, img_size=128):
    """MobileNetV2 con transfer learning.

    Returns:
        (model, base_model) — modelo completo y referencia al base para fine-tuning.
    """
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False

    # Wrappear base como sub-modelo (training=False mantiene BN en modo inferencia)
    inp = layers.Input(shape=(img_size, img_size, 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inp, outputs=out)
    return model, base


def build_custom_cnn(num_classes, img_size=128):
    """CNN de 3 bloques (32/64/128 filtros) + BatchNorm + GAP + Dropout."""
    inp = layers.Input(shape=(img_size, img_size, 3))
    x = inp
    for f in [32, 64, 128]:
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs=inp, outputs=out)


def detect_preprocessing(model):
    """Detecta si un modelo usa preprocesamiento MobileNetV2."""
    for layer in model.layers:
        name = layer.name.lower()
        if "mobilenet" in name:
            return "mobilenet"
    return "rescale"


def train_model(model_type, train_gen, val_gen, num_classes, img_size, epochs, lr):
    """Entrena modelo según tipo seleccionado.

    Args:
        model_type: 'mobilenetv2' o 'custom_cnn'.

    Returns:
        (model, history_dict) — modelo entrenado y dict con métricas por época.
    """
    if model_type == "mobilenetv2":
        model, base = build_mobilenetv2(num_classes, img_size)

        # Fase 1: base congelada, entrenar solo el clasificador
        phase1_epochs = min(5, max(1, epochs // 4))
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        h1 = model.fit(
            train_gen, epochs=phase1_epochs, validation_data=val_gen, verbose=0,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        )

        # Fase 2: descongelar últimas capas del base para fine-tuning
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False

        phase2_epochs = epochs - phase1_epochs
        model.compile(
            optimizer=Adam(learning_rate=lr / 100),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        h2 = model.fit(
            train_gen, epochs=phase2_epochs, validation_data=val_gen, verbose=0,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5),
            ],
        )

        # Combinar historias
        history = {}
        for key in h1.history:
            history[key] = h1.history[key] + h2.history[key]
        return model, history

    else:
        model = build_custom_cnn(num_classes, img_size)
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        h = model.fit(
            train_gen, epochs=epochs, validation_data=val_gen, verbose=0,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5),
            ],
        )
        return model, h.history
