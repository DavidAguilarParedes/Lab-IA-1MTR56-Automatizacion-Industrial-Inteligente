"""Tab 3: HMI de producción — Control de calidad con PLC Beckhoff."""

import os
import glob
import json
import time
import threading
import datetime

import cv2
import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model

from app.config import state, BASE_DIR
from app.datos import preprocess_image
from app.plc import plc_bridge


# ── Estado del HMI ──

hmi_state = {
    'model': None,
    'class_names': [],
    'img_size': 128,
    'preprocessing': 'rescale',
    'model_path': '',
    'polling': False,
    'poll_thread': None,
    'history': [],
    'captures_per_inspection': 5,
    'confidence_threshold': 0.6,
    'var_inicio': 'GVL.bInicioControlDeCalidad',
    'var_clase': 'GVL.nResultadoClase',
    'var_confianza': 'GVL.rConfianza',
    'session_dir': '',
    'last_status': 'ESPERANDO',
}


def _discover_models():
    """Descubre modelos .h5/.keras con metadata JSON."""
    modelos = {}
    search_dirs = [
        os.path.join(BASE_DIR, 'modelos'),
        os.path.join(BASE_DIR, 'proyecto', 'modelos'),
    ]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for ext in ('*.h5', '*.keras'):
            for h5 in glob.glob(os.path.join(d, ext)):
                json_path = h5.rsplit('.', 1)[0] + '.json'
                meta = {}
                if os.path.exists(json_path):
                    with open(json_path) as f:
                        meta = json.load(f)
                mb = os.path.getsize(h5) / 1024 / 1024
                va = meta.get('val_accuracy')
                label = os.path.basename(h5)
                if va is not None:
                    label += f" (val {va:.0%})"
                label += f" [{mb:.0f}MB]"
                modelos[label] = {'path': h5, 'meta': meta}
    return modelos


def _load_model_hmi(model_path):
    """Carga modelo para el HMI."""
    model = load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    img_size = model.input_shape[1]
    preprocessing = 'rescale'
    for layer in model.layers:
        if 'mobilenet' in layer.name.lower():
            preprocessing = 'mobilenet'
            break

    json_path = model_path.rsplit('.', 1)[0] + '.json'
    class_names = []
    if os.path.exists(json_path):
        with open(json_path) as f:
            meta = json.load(f)
        class_names = meta.get('class_names', [])
        if meta.get('preprocessing'):
            preprocessing = meta['preprocessing']

    if not class_names:
        n = model.output_shape[-1]
        class_names = [f'Clase {i}' for i in range(n)]

    hmi_state['model'] = model
    hmi_state['class_names'] = class_names
    hmi_state['img_size'] = img_size
    hmi_state['preprocessing'] = preprocessing
    hmi_state['model_path'] = model_path

    return class_names, img_size, preprocessing


def _classify_frame(frame):
    """Clasifica un frame BGR de OpenCV. Retorna (clase_idx, clase_nombre, confianza, preds)."""
    model = hmi_state['model']
    if model is None:
        return -1, '—', 0.0, []

    sz = hmi_state['img_size']
    img = cv2.resize(frame, (sz, sz))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    arr = preprocess_image(img, sz, hmi_state['preprocessing'])
    preds = model.predict(arr, verbose=0)[0]

    idx = int(np.argmax(preds))
    nombre = hmi_state['class_names'][idx]
    conf = float(preds[idx])
    return idx, nombre, conf, preds.tolist()


def _run_inspection(cap, n_captures):
    """Ejecuta inspección: captura N fotos, clasifica, voto mayoritario.

    Returns:
        (clase_idx, clase_nombre, confianza, capturas_paths, detalles)
    """
    if hmi_state['model'] is None:
        return -1, '—', 0.0, [], []

    # Crear carpeta de sesión
    if not hmi_state['session_dir']:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        hmi_state['session_dir'] = os.path.join(BASE_DIR, 'capturas', f'sesion_{ts}')
    os.makedirs(hmi_state['session_dir'], exist_ok=True)

    capturas = []
    detalles = []
    all_preds = []

    for i in range(n_captures):
        ret, frame = cap.read()
        if not ret:
            continue

        idx, nombre, conf, preds = _classify_frame(frame)

        # Guardar captura
        ts = datetime.datetime.now().strftime('%H%M%S_%f')[:-3]
        path = os.path.join(hmi_state['session_dir'], f'insp_{ts}.jpg')
        cv2.imwrite(path, frame)

        capturas.append(path)
        detalles.append({'clase': nombre, 'idx': idx, 'confianza': conf})
        all_preds.append(preds)

        time.sleep(0.05)  # breve pausa entre capturas

    if not all_preds:
        return -1, '—', 0.0, [], []

    # Voto por promedio de confianzas
    mean_preds = np.mean(all_preds, axis=0)
    final_idx = int(np.argmax(mean_preds))
    final_nombre = hmi_state['class_names'][final_idx]
    final_conf = float(mean_preds[final_idx])

    return final_idx, final_nombre, final_conf, capturas, detalles


def _history_to_table():
    """Convierte historial a lista de listas para Dataframe."""
    rows = []
    for h in reversed(hmi_state['history'][-50:]):
        ok = "OK" if h['confianza'] >= hmi_state['confidence_threshold'] else "BAJA"
        rows.append([
            h['timestamp'],
            h['clase'],
            f"{h['confianza']:.0%}",
            ok,
            h.get('modo', '—'),
        ])
    return rows


def create():
    modelos = _discover_models()
    model_choices = list(modelos.keys()) if modelos else ["(no hay modelos)"]

    # ══════════════════════════════════════
    #  LAYOUT HMI
    # ══════════════════════════════════════

    gr.Markdown("### CONTROL DE CALIDAD — HMI")

    with gr.Row():
        # ── Columna izquierda: cámara ──
        with gr.Column(scale=1):
            with gr.Group(elem_classes="card"):
                cam_feed = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    type="numpy",
                    label="Feed de camara",
                    height=320,
                )

        # ── Columna derecha: estado + controles ──
        with gr.Column(scale=1):
            with gr.Group(elem_classes="card"):
                with gr.Row():
                    status_plc = gr.Textbox(
                        value="🔴 PLC Desconectado",
                        label="Estado PLC",
                        interactive=False,
                        scale=2,
                    )
                    status_hmi = gr.Textbox(
                        value="ESPERANDO",
                        label="Estado HMI",
                        interactive=False,
                        scale=1,
                    )

                result_clase = gr.Textbox(
                    value="—", label="Clase detectada",
                    interactive=False,
                )
                result_conf = gr.Textbox(
                    value="—%", label="Confianza",
                    interactive=False,
                )
                result_detail = gr.Textbox(
                    value="", label="Detalle capturas",
                    interactive=False, lines=2,
                )

            with gr.Group(elem_classes="card"):
                btn_manual = gr.Button(
                    "FORZAR INSPECCION",
                    variant="primary",
                    size="lg",
                )
                gr.Markdown(
                    "*Ejecuta inspeccion sin esperar señal del PLC.*"
                )

    # ── Historial ──
    with gr.Group(elem_classes="card"):
        gr.Markdown("#### Historial de inspecciones")
        history_table = gr.Dataframe(
            headers=["Hora", "Clase", "Confianza", "Resultado", "Modo"],
            datatype=["str", "str", "str", "str", "str"],
            row_count=5,
            column_count=(5, "fixed"),
            interactive=False,
        )

    # ── Configuración (acordeón) ──
    with gr.Accordion("Configuracion", open=False):
        with gr.Row():
            dd_modelo = gr.Dropdown(
                choices=model_choices,
                value=model_choices[0] if model_choices else None,
                label="Modelo (.h5)",
                scale=2,
            )
            btn_cargar = gr.Button("Cargar modelo", scale=1)

        model_info = gr.Textbox(
            value="No hay modelo cargado", label="Info del modelo",
            interactive=False,
        )

        gr.Markdown("**Conexion PLC (pyads)**")
        with gr.Row():
            inp_ams = gr.Textbox(
                value="5.80.201.232.1.1",
                label="AMS Net ID",
                placeholder="ej: 5.80.201.232.1.1",
                scale=2,
            )
            inp_port = gr.Number(
                value=851, label="Puerto", precision=0,
                scale=1,
            )
        with gr.Row():
            btn_connect = gr.Button("Conectar PLC", variant="secondary")
            btn_disconnect = gr.Button("Desconectar")

        gr.Markdown("**Variables PLC**")
        with gr.Row():
            inp_var_inicio = gr.Textbox(
                value="GVL.bInicioControlDeCalidad",
                label="Variable inicio (BOOL)",
            )
        with gr.Row():
            inp_var_clase = gr.Textbox(
                value="GVL.nResultadoClase",
                label="Variable resultado clase (INT)",
            )
            inp_var_conf = gr.Textbox(
                value="GVL.rConfianza",
                label="Variable confianza (REAL)",
            )

        gr.Markdown("**Parametros de inspeccion**")
        with gr.Row():
            inp_n_capturas = gr.Slider(
                minimum=1, maximum=10, value=5, step=1,
                label="Capturas por inspeccion",
            )
            inp_threshold = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.6, step=0.05,
                label="Threshold de confianza",
            )

        plc_log = gr.Textbox(
            value="", label="Log PLC", lines=4, interactive=False,
        )

    # ══════════════════════════════════════
    #  HANDLERS
    # ══════════════════════════════════════

    def cargar_modelo(modelo_label):
        if modelo_label not in modelos:
            return "No se pudo cargar el modelo."
        info = modelos[modelo_label]
        try:
            names, sz, prep = _load_model_hmi(info['path'])
            tipo = 'MobileNetV2' if prep == 'mobilenet' else 'CNN custom'
            return (
                f"Modelo: {os.path.basename(info['path'])}\n"
                f"Tipo: {tipo} | Input: {sz}x{sz}\n"
                f"Clases: {names}"
            )
        except Exception as e:
            return f"Error: {e}"

    def conectar_plc(ams, port):
        ok, msg = plc_bridge.connect(ams, int(port))
        status = plc_bridge.status_emoji
        return status, plc_bridge.get_log()

    def desconectar_plc():
        plc_bridge.disconnect()
        return plc_bridge.status_emoji, plc_bridge.get_log()

    def actualizar_vars(var_inicio, var_clase, var_conf, n_cap, threshold):
        hmi_state['var_inicio'] = var_inicio
        hmi_state['var_clase'] = var_clase
        hmi_state['var_confianza'] = var_conf
        hmi_state['captures_per_inspection'] = int(n_cap)
        hmi_state['confidence_threshold'] = threshold

    def forzar_inspeccion():
        """Ejecuta inspección manual sin PLC."""
        if hmi_state['model'] is None:
            return (
                "FALTA MODELO", "—", "—%", "",
                _history_to_table(), plc_bridge.get_log()
            )

        # Abrir cámara para captura rápida
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return (
                "ERROR CAMARA", "—", "—%", "No se pudo abrir camara",
                _history_to_table(), plc_bridge.get_log()
            )

        try:
            n = hmi_state['captures_per_inspection']
            # Descartar algunos frames para que la cámara se estabilice
            for _ in range(5):
                cap.read()

            idx, nombre, conf, paths, detalles = _run_inspection(cap, n)
        finally:
            cap.release()

        if idx < 0:
            return (
                "ERROR", "—", "—%", "Inspección fallida",
                _history_to_table(), plc_bridge.get_log()
            )

        # Registrar en historial
        threshold = hmi_state['confidence_threshold']
        hmi_state['history'].append({
            'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
            'clase': nombre,
            'clase_idx': idx,
            'confianza': conf,
            'modo': 'Manual',
        })

        # Intentar enviar al PLC si conectado
        if plc_bridge.connected:
            plc_bridge.enviar_resultado(
                idx, conf,
                hmi_state['var_clase'],
                hmi_state['var_confianza'],
            )

        # Detalle de cada captura
        detail_lines = []
        for i, d in enumerate(detalles):
            detail_lines.append(f"  #{i+1}: {d['clase']} ({d['confianza']:.0%})")
        detail_str = f"{n} capturas → voto: {nombre}\n" + "\n".join(detail_lines)

        ok_str = "OK" if conf >= threshold else "BAJA CONFIANZA"
        status = f"RESULTADO: {ok_str}"

        return (
            status,
            nombre,
            f"{conf:.0%}",
            detail_str,
            _history_to_table(),
            plc_bridge.get_log(),
        )

    # ── Wiring ──

    btn_cargar.click(
        cargar_modelo,
        inputs=[dd_modelo],
        outputs=[model_info],
    )

    btn_connect.click(
        conectar_plc,
        inputs=[inp_ams, inp_port],
        outputs=[status_plc, plc_log],
    )
    btn_disconnect.click(
        desconectar_plc,
        outputs=[status_plc, plc_log],
    )

    # Guardar variables cuando cambian
    for inp in [inp_var_inicio, inp_var_clase, inp_var_conf]:
        inp.change(
            actualizar_vars,
            inputs=[inp_var_inicio, inp_var_clase, inp_var_conf,
                    inp_n_capturas, inp_threshold],
        )
    for inp in [inp_n_capturas, inp_threshold]:
        inp.change(
            actualizar_vars,
            inputs=[inp_var_inicio, inp_var_clase, inp_var_conf,
                    inp_n_capturas, inp_threshold],
        )

    btn_manual.click(
        forzar_inspeccion,
        outputs=[
            status_hmi, result_clase, result_conf, result_detail,
            history_table, plc_log,
        ],
    )
