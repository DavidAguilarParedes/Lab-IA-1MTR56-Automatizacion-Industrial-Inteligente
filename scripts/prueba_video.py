"""
PRUEBA DE MODELO CNN EN TIEMPO REAL

Usa la cámara para clasificar en tiempo real con suavizado temporal.

USO:
  python scripts/prueba_video.py                          # usa el primer .h5 en modelos/
  python scripts/prueba_video.py modelos/mi_modelo.h5     # modelo específico

Si el modelo tiene un .json asociado (generado por entrenar.ipynb),
se auto-detectan IMG_SIZE, CLASS_NAMES y preprocessing.

CONTROLES:
  'q' → Salir
  's' → Guardar captura actual (en capturas/<modelo>/)
  'd' → Mostrar/ocultar barras de confianza
"""

import os
import sys
import glob
import json
import cv2
import time
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

# ============================================================
# CONFIGURACIÓN
# ============================================================

CONFIDENCE_THRESHOLD = 0.6
CAMERA_INDEX = 0
SMOOTHING_WINDOW = 5      # promedio últimas N predicciones (anti-parpadeo)

# ============================================================
# RESOLVER MODELO (argumento CLI o auto-detectar)
# ============================================================

def find_model_path():
    """Determina la ruta del modelo desde CLI arg o buscando en modelos/."""
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path):
            return path
        print(f"✘ No se encontró: {path}")
        sys.exit(1)

    # Buscar en modelos/
    candidates = sorted(
        glob.glob('modelos/*.h5') + glob.glob('modelos/*.keras'),
        key=os.path.getmtime, reverse=True
    )
    # También buscar en proyecto/modelos/ por si se ejecuta desde raíz
    if not candidates:
        candidates = sorted(
            glob.glob('proyecto/modelos/*.h5') + glob.glob('proyecto/modelos/*.keras'),
            key=os.path.getmtime, reverse=True
        )

    if not candidates:
        print("✘ No se encontraron modelos .h5/.keras")
        print("  Pase la ruta como argumento: python scripts/prueba_video.py modelos/mi_modelo.h5")
        sys.exit(1)

    print(f"Modelos disponibles:")
    for i, c in enumerate(candidates):
        mb = os.path.getsize(c) / 1024 / 1024
        marker = " ← (más reciente)" if i == 0 else ""
        print(f"  {c} ({mb:.1f} MB){marker}")

    return candidates[0]


MODEL_PATH = find_model_path()

# ============================================================
# CARGAR MODELO + METADATA JSON
# ============================================================

print(f"\nCargando modelo: {MODEL_PATH}")
model = load_model(MODEL_PATH, compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

IMG_SIZE = model.input_shape[1]
n_clases = model.output_shape[-1]

# Detectar tipo de preprocesamiento
PREPROCESSING = "rescale"
for layer in model.layers:
    if "mobilenet" in layer.name.lower():
        PREPROCESSING = "mobilenet"
        break

# Intentar leer .json del modelo
json_path = MODEL_PATH.rsplit('.', 1)[0] + '.json'
CLASS_NAMES = []
metadata = {}

if os.path.exists(json_path):
    with open(json_path) as f:
        metadata = json.load(f)
    print(f"✔ Metadata cargada: {os.path.basename(json_path)}")

    if metadata.get('class_names'):
        CLASS_NAMES = metadata['class_names']
    if metadata.get('img_size'):
        json_size = metadata['img_size']
        if json_size != IMG_SIZE:
            print(f"  ⚠ JSON dice {json_size}px pero modelo espera {IMG_SIZE}px. Usando {IMG_SIZE}.")
    if metadata.get('preprocessing'):
        PREPROCESSING = metadata['preprocessing']
else:
    print(f"  ⚠ No se encontró {os.path.basename(json_path)}")

# Fallback para CLASS_NAMES
if not CLASS_NAMES:
    CLASS_NAMES = [f'Clase {i}' for i in range(n_clases)]
    print(f"  ⚠ Sin nombres de clase. Usando: {CLASS_NAMES}")
    print(f"    Para arreglar: entrene con entrenar.ipynb (genera .json automático)")

print(f"\n  Tipo: {'MobileNetV2' if PREPROCESSING == 'mobilenet' else 'CNN custom'}")
print(f"  Input: {IMG_SIZE}×{IMG_SIZE} | Clases: {n_clases}")
print(f"  Clases: {CLASS_NAMES}")
if metadata.get('val_accuracy'):
    print(f"  Val accuracy: {metadata['val_accuracy']:.1%}")

# Colores para cada clase (BGR para OpenCV)
COLORES = [
    (0, 200, 0),     # Verde
    (200, 100, 0),   # Azul
    (0, 0, 200),     # Rojo
    (0, 200, 200),   # Amarillo
    (200, 0, 200),   # Magenta
    (200, 200, 0),   # Cyan
]

# ============================================================
# CARPETA DE CAPTURAS
# ============================================================

model_name = os.path.splitext(os.path.basename(MODEL_PATH))[0]
capture_dir = os.path.join('capturas', model_name)
os.makedirs(capture_dir, exist_ok=True)

# ============================================================
# CLASIFICACIÓN EN TIEMPO REAL
# ============================================================

print()
print("Iniciando cámara...")
print(f"  Suavizado: últimas {SMOOTHING_WINDOW} predicciones")
print(f"  Capturas en: {capture_dir}/")
print("  'q' → Salir | 's' → Captura | 'd' → Panel")

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"No se pudo abrir la cámara (índice {CAMERA_INDEX})")
    print("Intente cambiar CAMERA_INDEX a 1")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = 0
show_diag = True
capture_count = 0
pred_buffer = deque(maxlen=SMOOTHING_WINDOW)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer frame")
            break

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        # Preprocesar
        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32")
        if PREPROCESSING == "mobilenet":
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            img_normalized = preprocess_input(img)
        else:
            img_normalized = img / 255.0
        img_input = np.expand_dims(img_normalized, axis=0)

        # Predecir + suavizado temporal
        preds = model.predict(img_input, verbose=0)[0]
        pred_buffer.append(preds)
        smooth_preds = np.mean(pred_buffer, axis=0)

        pred_class = np.argmax(smooth_preds)
        confidence = smooth_preds[pred_class]
        class_name = CLASS_NAMES[pred_class]

        # --- Visualización con UI ---
        h_frame, w_frame = frame.shape[:2]
        color = COLORES[pred_class % len(COLORES)]

        # Barra superior semitransparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w_frame, 60), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        if confidence >= CONFIDENCE_THRESHOLD:
            label = f"{class_name}  {confidence*100:.0f}%"
            cv2.rectangle(frame, (0, 0), (6, 60), color, -1)
        else:
            label = f"?  {confidence*100:.0f}%"
            color = (128, 128, 128)
            cv2.rectangle(frame, (0, 0), (6, 60), color, -1)

        cv2.putText(frame, label, (16, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{fps:.0f} FPS", (w_frame - 90, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        # Panel lateral de confianza
        if show_diag:
            panel_w = 220
            panel_h = 30 + len(CLASS_NAMES) * 36
            panel_x = w_frame - panel_w - 10
            panel_y = 70

            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (panel_x, panel_y),
                          (panel_x + panel_w, panel_y + panel_h),
                          (20, 20, 20), -1)
            cv2.addWeighted(overlay2, 0.8, frame, 0.2, 0, frame)

            cv2.putText(frame, "Confianza", (panel_x + 10, panel_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

            bar_w = panel_w - 20
            for i, (name, prob) in enumerate(zip(CLASS_NAMES, smooth_preds)):
                y = panel_y + 32 + i * 36
                cv2.putText(frame, name, (panel_x + 10, y + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (220, 220, 220) if i == pred_class else (140, 140, 140),
                            1, cv2.LINE_AA)
                bar_y = y + 8
                cv2.rectangle(frame, (panel_x + 10, bar_y),
                              (panel_x + 10 + bar_w, bar_y + 14), (60, 60, 60), -1)
                fill_w = max(2, int(bar_w * prob))
                bc = COLORES[i % len(COLORES)] if i == pred_class else (100, 100, 100)
                cv2.rectangle(frame, (panel_x + 10, bar_y),
                              (panel_x + 10 + fill_w, bar_y + 14), bc, -1)
                cv2.putText(frame, f"{prob*100:.0f}%",
                            (panel_x + 10 + bar_w + 2, bar_y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

        # Barra inferior de controles
        overlay3 = frame.copy()
        cv2.rectangle(overlay3, (0, h_frame - 30), (w_frame, h_frame), (30, 30, 30), -1)
        cv2.addWeighted(overlay3, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, "Q salir  |  S captura  |  D panel",
                    (10, h_frame - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (140, 140, 140), 1, cv2.LINE_AA)

        cv2.imshow('PUCP - Clasificacion en Tiempo Real', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            path = os.path.join(capture_dir, f"captura_{capture_count:03d}.jpg")
            cv2.imwrite(path, frame)
            print(f"  Captura guardada: {path}")
            capture_count += 1
        elif key == ord('d'):
            show_diag = not show_diag

except KeyboardInterrupt:
    print("\nInterrumpido por usuario")
except Exception as e:
    print(f"Error: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Cámara liberada")
