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
        img_normalized = img / 255.0
        img_input = np.expand_dims(img_normalized, axis=0)

        # Predecir
        preds = model.predict(img_input, verbose=0)[0]
        pred_class = np.argmax(preds)
        confidence = preds[pred_class]
        class_name = CLASS_NAMES[pred_class]

        # --- Visualización ---
        h_frame, w_frame = frame.shape[:2]

        # Resultado principal
        if confidence >= CONFIDENCE_THRESHOLD:
            color = COLORES[pred_class % len(COLORES)]
            text = f"{class_name}: {confidence*100:.1f}%"
        else:
            color = (128, 128, 128)
            text = f"Incierto ({confidence*100:.1f}%)"

        cv2.rectangle(frame, (5, 5), (380, 45), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # FPS
        cv2.putText(frame, f"FPS: {fps:.0f}", (w_frame - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Barras de confianza (diagnóstico)
        if show_diag:
            bar_x = 10
            bar_y = 55
            bar_w = 200
            bar_h = 22

            for i, (name, prob) in enumerate(zip(CLASS_NAMES, preds)):
                y = bar_y + i * (bar_h + 4)
                # Fondo
                cv2.rectangle(frame, (bar_x, y),
                              (bar_x + bar_w, y + bar_h), (40, 40, 40), -1)
                # Barra
                w = int(bar_w * prob)
                c = COLORES[i % len(COLORES)] if i == pred_class else (80, 80, 80)
                cv2.rectangle(frame, (bar_x, y),
                              (bar_x + w, y + bar_h), c, -1)
                # Texto
                cv2.putText(frame, f"{name}: {prob*100:.0f}%",
                            (bar_x + 5, y + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Mostrar
        cv2.imshow('Clasificacion en Tiempo Real', frame)

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
