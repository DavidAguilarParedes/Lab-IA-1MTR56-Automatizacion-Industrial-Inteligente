#===================================#
#===== DIVISOR DE VIDEO FRAMES =====#
#===================================#
"""
Extractor de Frames para Zona de Inspección
Convierte un video .mp4 en imágenes para entrenar la CNN.

INSTRUCCIONES PARA ZONA DE INSPECCIÓN:
  1. Monte el celular en un soporte FIJO apuntando a la zona de inspección
  2. Grabe un video donde el robot coloca diferentes tapitas
  3. Ajuste interval_seconds (0.5s recomendado)
  4. Active detect_stability=True para descartar frames durante el movimiento del robot
  5. Use el MISMO setup de cámara para entrenar Y para probar

NOTA: Es mejor tener 50 imágenes variadas que 500 casi idénticas.
      Grabe videos CORTOS (30-60s) por cada clase.
"""

import cv2
import os
import numpy as np


def video_to_frames(video_path, output_folder, clase, interval_seconds=0.5,
                    min_sharpness=50.0, detect_stability=True):
    """
    Extrae frames de un video para crear dataset de entrenamiento.

    Args:
        video_path: Ruta al video .mp4
        output_folder: Carpeta donde guardar las imágenes
        clase: Nombre de la clase (para nombrar archivos)
        interval_seconds: Segundos entre cada frame (mínimo 0.5 recomendado)
        min_sharpness: Umbral de nitidez. Frames borrosos (movimiento) se descartan.
                       Valor más alto = más estricto. 50 es buen punto de partida.
        detect_stability: Si True, solo extrae cuando la escena está estable
                          (detecta cuándo el robot ya colocó la pieza)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"No se pudo abrir el video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video: {video_path}")
    print(f"  FPS: {fps:.0f} | Duración: {duration:.1f}s | Frames totales: {total_frames}")
    print(f"  Intervalo: {interval_seconds}s | Nitidez mín: {min_sharpness}")
    print(f"  Detección de estabilidad: {'Sí' if detect_stability else 'No'}")
    print()

    if fps == 0:
        print("No se pudo obtener FPS del video.")
        return

    frame_interval = max(1, int(fps * interval_seconds))
    prev_gray = None
    count = 0
    saved = 0
    skipped_blur = 0
    skipped_motion = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Filtro 1: Descartar frames borrosos (cámara moviéndose, robot en tránsito)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            if sharpness < min_sharpness:
                skipped_blur += 1
                count += 1
                continue

            # Filtro 2: Detectar estabilidad (solo capturar cuando la escena está quieta)
            if detect_stability and prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion = np.mean(diff)
                if motion > 15:  # Mucho cambio entre frames = robot en movimiento
                    skipped_motion += 1
                    prev_gray = gray
                    count += 1
                    continue

            prev_gray = gray

            frame_path = os.path.join(output_folder, f'frame_{clase}_{saved:05d}.jpg')
            cv2.imwrite(frame_path, frame)
            saved += 1

            # Progreso cada 10 frames
            if saved % 10 == 0:
                print(f"  Guardados: {saved}...", end='\r')

        count += 1

    cap.release()
    print(f"\nResultados:")
    print(f"  Frames guardados: {saved}")
    print(f"  Descartados por borrosidad: {skipped_blur}")
    if detect_stability:
        print(f"  Descartados por movimiento: {skipped_motion}")
    print(f"  Carpeta: {os.path.abspath(output_folder)}")

    if saved < 20:
        print(f"\n⚠ Solo se guardaron {saved} frames. Considere:")
        print(f"  - Reducir min_sharpness (actual: {min_sharpness})")
        print(f"  - Reducir interval_seconds (actual: {interval_seconds})")
        print(f"  - Grabar un video más largo")


# ==========================
# CLI
# ==========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extrae frames de un video para crear dataset de entrenamiento."
    )
    parser.add_argument(
        "clase", help="Nombre de la clase (ej: amarillo, rojo)"
    )
    parser.add_argument(
        "--video", default=None,
        help="Ruta al video .mp4 (default: <clase>.mp4 en el directorio del script)"
    )
    parser.add_argument(
        "--salida", default=None,
        help="Directorio de salida (default: ../data/tapitas/<clase>)"
    )
    parser.add_argument(
        "--intervalo", type=float, default=0.5,
        help="Segundos entre frames (default: 0.5)"
    )
    parser.add_argument(
        "--nitidez", type=float, default=50.0,
        help="Umbral de nitidez mínima (default: 50.0)"
    )
    parser.add_argument(
        "--sin-estabilidad", action="store_true",
        help="Desactivar detección de estabilidad"
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = args.video or os.path.join(base_dir, f"{args.clase}.mp4")
    output_folder = args.salida or os.path.join(
        base_dir, "..", "data", "tapitas", args.clase
    )

    video_to_frames(
        video_path,
        output_folder,
        args.clase,
        interval_seconds=args.intervalo,
        min_sharpness=args.nitidez,
        detect_stability=not args.sin_estabilidad,
    )
