"""
PLANTILLA: INFERENCIA CNN + COMUNICACIÓN PLC

Este script captura frames de la cámara, clasifica con el modelo CNN,
y envía el resultado al PLC via OPC UA.

INSTRUCCIONES:
  1. Complete la sección CONFIGURACIÓN con los valores de su modelo
  2. Complete los TODOs con la lógica de comunicación PLC
  3. Ejecute: python scripts/inferencia_plc.py

CONTROLES:
  'q' → Salir
"""

import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ============================================================
# CONFIGURACIÓN — completar con los valores de su modelo
# ============================================================

MODEL_PATH = "COMPLETAR.h5"          # ruta al modelo exportado
IMG_SIZE = 128                        # mismo valor que en el notebook
CLASS_NAMES = []                      # misma lista que en el notebook
CONFIDENCE_THRESHOLD = 0.6            # confianza mínima para enviar al PLC
CAMERA_INDEX = 0                      # cambiar a 1 si no detecta la cámara

# OPC UA
OPC_URL = "opc.tcp://localhost:4840/pucp/"
NODE_CLASIFICACION = "ns=2;s=Clasificacion"
NODE_CONFIANZA = "ns=2;s=Confianza"

# ============================================================
# FUNCIONES DE INFERENCIA (ya implementadas)
# ============================================================

def cargar_modelo(path):
    """Carga el modelo .h5 y valida la configuración."""
    print(f"Cargando modelo: {path}")
    model = load_model(path)

    expected_size = model.input_shape[1]
    if expected_size != IMG_SIZE:
        print(f"  ⚠ El modelo espera {expected_size}×{expected_size} pero IMG_SIZE={IMG_SIZE}")

    n_out = model.output_shape[-1]
    if len(CLASS_NAMES) != n_out:
        print(f"  ⚠ El modelo tiene {n_out} salidas pero CLASS_NAMES tiene {len(CLASS_NAMES)}")

    print(f"  ✓ Modelo listo ({n_out} clases, {expected_size}px)")
    return model


def _detectar_preprocesamiento(model):
    """Detecta si el modelo usa MobileNetV2 (requiere preprocesamiento diferente)."""
    for layer in model.layers:
        if 'mobilenet' in layer.name.lower():
            return 'mobilenet'
    return 'rescale'


def predecir(model, frame, preprocesamiento='rescale'):
    """Preprocesa un frame y retorna (clase, confianza, todas_las_probabilidades)."""
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32")

    if preprocesamiento == 'mobilenet':
        # MobileNetV2: normaliza a [-1, 1]
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        img = preprocess_input(img)
    else:
        # CNN custom: normaliza a [0, 1]
        img = img / 255.0

    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)[0]
    idx = np.argmax(preds)
    return CLASS_NAMES[idx], float(preds[idx]), preds


# ============================================================
# TODO: CONEXIÓN PLC
#
# Ejemplo usando opcua (python-opcua):
#
#   from opcua import Client, ua
#
#   client = Client(OPC_URL)
#   client.connect()
#
#   node_clas = client.get_node(NODE_CLASIFICACION)
#   node_conf = client.get_node(NODE_CONFIANZA)
#
#   # Escribir valor entero (1=clase1, 2=clase2, 0=incierto):
#   node_clas.set_value(ua.Variant(valor, ua.VariantType.Int16))
#   node_conf.set_value(ua.Variant(confianza, ua.VariantType.Float))
#
#   # Para leer un pulso de entrada del PLC:
#   # pulso = client.get_node("ns=2;s=Pulso").get_value()
#
#   client.disconnect()
# ============================================================

def conectar_plc():
    """TODO: Conectar al PLC via OPC UA."""
    # Descomentar y completar:
    # from opcua import Client
    # client = Client(OPC_URL)
    # client.connect()
    # print(f"✓ Conectado a {OPC_URL}")
    # return client
    print("⚠ PLC no configurado (complete conectar_plc)")
    return None


def enviar_al_plc(client, clase, confianza):
    """TODO: Enviar clasificación al PLC."""
    # Descomentar y completar:
    # from opcua import ua
    # valor = CLASS_NAMES.index(clase) + 1 if confianza >= CONFIDENCE_THRESHOLD else 0
    # node = client.get_node(NODE_CLASIFICACION)
    # node.set_value(ua.Variant(valor, ua.VariantType.Int16))
    # node_c = client.get_node(NODE_CONFIANZA)
    # node_c.set_value(ua.Variant(confianza, ua.VariantType.Float))
    pass


def leer_pulso_plc(client):
    """TODO: Leer señal de entrada del PLC (ej: pulso de sensor)."""
    # Descomentar y completar:
    # pulso = client.get_node("ns=2;s=Pulso").get_value()
    # return pulso
    return True  # siempre clasificar (sin PLC)


# ============================================================
# LOOP PRINCIPAL
# ============================================================

def main():
    # Validar configuración
    if MODEL_PATH == "COMPLETAR.h5" or not CLASS_NAMES:
        print("=" * 55)
        print("  Complete la sección CONFIGURACIÓN primero")
        print("=" * 55)
        print(f"  MODEL_PATH = \"{MODEL_PATH}\"")
        print(f"  CLASS_NAMES = {CLASS_NAMES}")
        return

    model = cargar_modelo(MODEL_PATH)
    preproc = _detectar_preprocesamiento(model)
    print(f"  Preprocesamiento: {preproc}")
    plc_client = conectar_plc()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara (índice {CAMERA_INDEX})")
        return

    print("\nClasificando... ('q' para salir)\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # TODO: leer pulso de entrada del PLC
            if not leer_pulso_plc(plc_client):
                continue

            # Predecir
            clase, confianza, _ = predecir(model, frame, preproc)

            # Mostrar resultado
            if confianza >= CONFIDENCE_THRESHOLD:
                print(f"  {clase}: {confianza:.1%}")
            else:
                print(f"  Incierto ({confianza:.1%})")

            # TODO: enviar clasificación al PLC
            enviar_al_plc(plc_client, clase, confianza)

            # Mostrar frame (opcional)
            cv2.putText(frame, f"{clase} ({confianza:.0%})", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
            cv2.imshow("Clasificacion", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nInterrumpido")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # TODO: desconectar PLC
        # if plc_client: plc_client.disconnect()
        print("Finalizado")


if __name__ == "__main__":
    main()
