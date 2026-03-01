"""
CAPTURA DE IMÁGENES POR CLASE — Teachable Machine Style

Abre la cámara y captura imágenes para cada clase con un temporizador.
Cada clase tiene el MISMO tiempo de captura para mantener el dataset balanceado.

USO:
  python capturar_clases.py                         # Modo interactivo
  python capturar_clases.py rojo azul verde         # Clases directas
  python capturar_clases.py rojo azul --tiempo 15   # 15 segundos por clase

CONTROLES:
  ESPACIO → Iniciar captura de la clase actual
  q       → Salir
"""

import cv2
import os
import sys
import time
import argparse
import numpy as np


def capturar_clases(class_names, output_dir, seconds_per_class=10,
                    interval=0.3, camera_index=0, min_sharpness=30.0):
    """
    Captura imágenes de la cámara para cada clase.

    Args:
        class_names: Lista de nombres de clase
        output_dir: Directorio base donde crear subcarpetas
        seconds_per_class: Segundos de captura por clase (igual para todas)
        interval: Segundos entre capturas (0.3 = ~3 fotos/segundo)
        camera_index: Índice de la cámara (0 = default)
        min_sharpness: Umbral de nitidez para descartar frames borrosos
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara (índice {camera_index})")
        print("Intente cambiar camera_index a 1")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    print(f"\nClases a capturar: {class_names}")
    print(f"Tiempo por clase: {seconds_per_class}s")
    print(f"Directorio: {os.path.abspath(output_dir)}")
    print(f"\nCONTROLES: ESPACIO = iniciar captura | q = salir\n")

    for i, clase in enumerate(class_names):
        clase_dir = os.path.join(output_dir, clase)
        os.makedirs(clase_dir, exist_ok=True)

        saved = 0
        capturing = False
        start_time = None
        last_capture = 0

        print(f"--- Clase {i+1}/{len(class_names)}: {clase} ---")
        print(f"    Coloque un objeto de clase '{clase}' y presione ESPACIO")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            h, w = display.shape[:2]

            if capturing:
                elapsed = time.time() - start_time
                remaining = max(0, seconds_per_class - elapsed)

                if remaining <= 0:
                    # Tiempo terminado
                    capturing = False
                    results[clase] = saved
                    print(f"\n    {clase}: {saved} imágenes capturadas")
                    break

                # Barra de progreso
                progress = elapsed / seconds_per_class
                bar_w = int(w * 0.8)
                bar_x = int(w * 0.1)
                bar_y = h - 40
                cv2.rectangle(display, (bar_x, bar_y),
                              (bar_x + bar_w, bar_y + 20), (50, 50, 50), -1)
                cv2.rectangle(display, (bar_x, bar_y),
                              (bar_x + int(bar_w * progress), bar_y + 20),
                              (0, 200, 0), -1)

                # Capturar frame si pasó el intervalo
                now = time.time()
                if now - last_capture >= interval:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                    if sharpness >= min_sharpness:
                        path = os.path.join(clase_dir, f"{clase}_{saved:04d}.jpg")
                        cv2.imwrite(path, frame)
                        saved += 1
                        last_capture = now

                # Info
                cv2.rectangle(display, (0, 0), (w, 60), (0, 0, 0), -1)
                cv2.putText(display,
                            f"CAPTURANDO: {clase} | {remaining:.0f}s | {saved} fotos",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)
            else:
                # Esperando
                cv2.rectangle(display, (0, 0), (w, 60), (0, 0, 0), -1)
                cv2.putText(display,
                            f"Clase: {clase} ({i+1}/{len(class_names)}) - ESPACIO para capturar",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)

            cv2.imshow('Captura de Clases', display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                if saved > 0:
                    results[clase] = saved
                cap.release()
                cv2.destroyAllWindows()
                return results
            elif key == ord(' ') and not capturing:
                # Countdown 3..2..1
                for countdown in [3, 2, 1]:
                    ret, frame = cap.read()
                    if ret:
                        display = frame.copy()
                        cv2.rectangle(display, (0, 0), (w, 80), (0, 0, 0), -1)
                        cv2.putText(display, str(countdown),
                                    (w // 2 - 20, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2,
                                    (0, 255, 255), 3)
                        cv2.imshow('Captura de Clases', display)
                        cv2.waitKey(1000)

                capturing = True
                start_time = time.time()
                last_capture = 0
                print(f"    Capturando {seconds_per_class}s...", end='', flush=True)

    cap.release()
    cv2.destroyAllWindows()

    # Resumen
    print(f"\n{'='*50}")
    print("RESUMEN DE CAPTURA")
    print(f"{'='*50}")
    total = 0
    for clase, n in results.items():
        print(f"  {clase}: {n} imágenes")
        total += n
    print(f"  Total: {total} imágenes")
    print(f"  Directorio: {os.path.abspath(output_dir)}")
    if results:
        counts = list(results.values())
        if max(counts) - min(counts) > min(counts) * 0.3:
            print(f"\n  ⚠ Dataset algo desbalanceado. Considere re-capturar.")
        else:
            print(f"\n  ✓ Dataset balanceado")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Captura de imágenes por clase')
    parser.add_argument('clases', nargs='*', help='Nombres de las clases')
    parser.add_argument('--tiempo', type=int, default=10,
                        help='Segundos por clase (default: 10)')
    parser.add_argument('--salida', default=None,
                        help='Directorio de salida (default: ../data/mi_dataset)')
    parser.add_argument('--camara', type=int, default=0,
                        help='Índice de cámara (default: 0)')
    args = parser.parse_args()

    # Modo interactivo si no se dan clases
    clases = args.clases
    if not clases:
        print("Ingrese los nombres de las clases separados por coma:")
        raw = input("  > ").strip()
        clases = [c.strip() for c in raw.split(',') if c.strip()]
        if len(clases) < 2:
            print("Se necesitan al menos 2 clases.")
            sys.exit(1)

    output_dir = args.salida or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'mi_dataset')

    capturar_clases(clases, output_dir, seconds_per_class=args.tiempo,
                    camera_index=args.camara)
