"""
PRUEBA DE MODELO CNN EN TIEMPO REAL

Usa la cámara para clasificar tapitas en la zona de inspección.

ANTES DE EJECUTAR:
  Complete las variables de configuración con los MISMOS valores
  que usó en el notebook de entrenamiento.

CONTROLES:
  'q' → Salir
  's' → Guardar captura actual
  'd' → Mostrar/ocultar barras de confianza
"""

import cv2
import time
import numpy as np
from tensorflow.keras.models import load_model

# ============================================================
# CONFIGURACIÓN - USAR LOS MISMOS VALORES DEL NOTEBOOK
# ============================================================

MODEL_PATH = "COMPLETAR.h5"       # COMPLETAR: ruta al modelo exportado
IMG_SIZE = None                    # COMPLETAR: mismo valor que en el notebook
CLASS_NAMES = []                   # COMPLETAR: misma lista que en el notebook
CONFIDENCE_THRESHOLD = 0.6         # COMPLETAR: confianza mínima para clasificar
CAMERA_INDEX = 0                   # Cambiar a 1 si no detecta la cámara

# ============================================================
# VALIDACIÓN DE CONFIGURACIÓN
# ============================================================

errores = []
if MODEL_PATH == "COMPLETAR.h5":
    errores.append("MODEL_PATH no configurado")
if IMG_SIZE is None:
    errores.append("IMG_SIZE no configurado")
if len(CLASS_NAMES) == 0:
    errores.append("CLASS_NAMES vacío")

if errores:
    print("=" * 55)
    print("  ERROR: Complete la configuración primero")
    print("=" * 55)
    for e in errores:
        print(f"  - {e}")
    print()
    print("Use los mismos valores que en su notebook:")
    print(f"  MODEL_PATH = \"{MODEL_PATH}\"")
    print(f"  IMG_SIZE = {IMG_SIZE}")
    print(f"  CLASS_NAMES = {CLASS_NAMES}")
    exit(1)

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
# CARGA DEL MODELO
# ============================================================

print(f"Cargando modelo: {MODEL_PATH}")
try:
    model = load_model(MODEL_PATH)
    print("Modelo cargado correctamente")
    expected_size = model.input_shape[1]
    if expected_size != IMG_SIZE:
        print(f"  ⚠ El modelo espera {expected_size}x{expected_size} pero IMG_SIZE={IMG_SIZE}")
        print(f"    Ajuste IMG_SIZE = {expected_size}")

    # Detectar tipo de preprocesamiento
    PREPROCESSING = "rescale"
    for layer in model.layers:
        if "mobilenet" in layer.name.lower():
            PREPROCESSING = "mobilenet"
            break
    print(f"  Preprocesamiento: {PREPROCESSING}")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit(1)

# ============================================================
# CLASIFICACIÓN EN TIEMPO REAL
# ============================================================

print()
print("Iniciando cámara...")
print("  'q' → Salir | 's' → Captura | 'd' → Diagnóstico")

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"No se pudo abrir la cámara (índice {CAMERA_INDEX})")
    print("Intente cambiar CAMERA_INDEX a 1")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = 0
show_diag = True
capture_count = 0

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

        # Preprocesar (misma lógica que en el notebook)
        img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV usa BGR, el modelo espera RGB
        img = img.astype("float32")
        if PREPROCESSING == "mobilenet":
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            img_normalized = preprocess_input(img)
        else:
            img_normalized = img / 255.0
        img_input = np.expand_dims(img_normalized, axis=0)

        # Predecir
        preds = model.predict(img_input, verbose=0)[0]
        pred_class = np.argmax(preds)
        confidence = preds[pred_class]
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
            for i, (name, prob) in enumerate(zip(CLASS_NAMES, preds)):
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
            path = f"captura_{capture_count:03d}.jpg"
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


"""
PREGUNTAS ACTIVIDAD 5:

1. ¿El modelo clasificó correctamente las tapitas en tiempo real?
   Si no, ¿cuál cree que es la causa principal?

2. ¿La cámara estaba en la MISMA posición y con la MISMA iluminación
   que cuando grabó los datos de entrenamiento?

3. ¿Qué mejoras concretas propone para mejorar los resultados?
"""
