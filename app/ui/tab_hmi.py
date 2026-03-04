"""Tab 3: HMI de producción — Control de calidad con PLC Beckhoff.

Layout industrial simplificado: cámara grande + resultado prominente.
Inspección multi-frame con buffer circular.
"""

import os
import glob
import json
import datetime
from collections import deque

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
    'history': [],
    'confidence_threshold': 0.6,
    'var_inicio': 'GVL.bInicioControlDeCalidad',
    'var_clase': 'GVL.nResultadoClase',
    'var_confianza': 'GVL.rConfianza',
    'session_dir': '',
}

# Buffer circular para inspección multi-frame
frame_buffer = deque(maxlen=10)


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


def _classify_frame(frame_rgb):
    """Clasifica un frame RGB. Retorna vector de probabilidades."""
    model = hmi_state['model']
    if model is None:
        return None

    sz = hmi_state['img_size']
    arr = preprocess_image(frame_rgb, sz, hmi_state['preprocessing'])
    preds = model.predict(arr, verbose=0)[0]
    return preds


def _render_clase_html(nombre, confianza, threshold):
    """Genera HTML para el panel lateral de resultado."""
    if nombre == '—':
        return (
            '<div class="hmi-panel">'
            '<div class="hmi-clase" style="color:#888;">—</div>'
            '<div class="hmi-conf-track"><div class="hmi-conf-fill" style="width:0%"></div></div>'
            '<div class="hmi-conf-text">—</div>'
            '</div>'
        )

    pct = confianza * 100
    color = '#4caf50' if confianza >= threshold else '#f44336'

    return (
        '<div class="hmi-panel">'
        f'<div class="hmi-clase">{nombre}</div>'
        f'<div class="hmi-conf-track">'
        f'<div class="hmi-conf-fill" style="width:{pct:.0f}%;background:{color}"></div>'
        f'</div>'
        f'<div class="hmi-conf-text" style="color:{color}">{pct:.0f}%</div>'
        '</div>'
    )


def _render_plc_status():
    """Genera HTML para indicador PLC."""
    if plc_bridge.connected:
        return '<div class="hmi-status"><span class="hmi-dot hmi-dot-ok"></span> PLC: OK</div>'
    return '<div class="hmi-status"><span class="hmi-dot hmi-dot-off"></span> PLC: Desconectado</div>'


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
            str(h.get('n_frames', 1)),
        ])
    return rows


def create():
    modelos = _discover_models()
    model_choices = list(modelos.keys()) if modelos else ["(no hay modelos)"]

    # ══════════════════════════════════════
    #  LAYOUT HMI INDUSTRIAL
    # ══════════════════════════════════════

    gr.Markdown("### CONTROL DE CALIDAD — HMI")

    with gr.Row():
        # ── Cámara (70% ancho) ──
        with gr.Column(scale=3):
            cam_feed = gr.Image(
                sources=["webcam"],
                streaming=True,
                type="numpy",
                label="Camara en vivo",
                height=400,
            )

        # ── Panel lateral resultado (30% ancho) ──
        with gr.Column(scale=1, elem_classes="hmi-sidebar"):
            result_html = gr.HTML(
                value=_render_clase_html('—', 0, 0.6),
                label="Resultado",
            )
            plc_html = gr.HTML(
                value=_render_plc_status(),
                label="PLC",
            )

    # ── Botón FORZAR ──
    btn_manual = gr.Button(
        "FORZAR INSPECCION",
        variant="primary",
        size="lg",
        elem_classes="hmi-btn-forzar",
    )

    # ── Historial (acordeón cerrado) ──
    with gr.Accordion("Historial de inspecciones", open=False):
        history_table = gr.Dataframe(
            headers=["Hora", "Clase", "Confianza", "Resultado", "Modo", "Frames"],
            datatype=["str", "str", "str", "str", "str", "str"],
            row_count=5,
            column_count=(6, "fixed"),
            interactive=False,
        )

    # ── Configuración (acordeón cerrado) ──
    with gr.Accordion("Configuracion", open=False):
        gr.Markdown("**Modelo**")
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
            inp_threshold = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.6, step=0.05,
                label="Threshold de confianza",
            )

        plc_log = gr.Textbox(
            value="", label="Log PLC", lines=3, interactive=False,
        )

    # ══════════════════════════════════════
    #  HANDLERS
    # ══════════════════════════════════════

    def on_stream(frame):
        """Callback del stream: guarda frame en buffer + predicción live."""
        if frame is None:
            return _render_clase_html('—', 0, hmi_state['confidence_threshold'])

        # Guardar en buffer circular
        frame_buffer.append(frame.copy())

        # Predicción live (preview en sidebar)
        preds = _classify_frame(frame)
        if preds is None:
            return _render_clase_html('—', 0, hmi_state['confidence_threshold'])

        idx = int(np.argmax(preds))
        nombre = hmi_state['class_names'][idx]
        conf = float(preds[idx])
        return _render_clase_html(nombre, conf, hmi_state['confidence_threshold'])

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
        return _render_plc_status(), plc_bridge.get_log()

    def desconectar_plc():
        plc_bridge.disconnect()
        return _render_plc_status(), plc_bridge.get_log()

    def actualizar_vars(var_inicio, var_clase, var_conf, threshold):
        hmi_state['var_inicio'] = var_inicio
        hmi_state['var_clase'] = var_clase
        hmi_state['var_confianza'] = var_conf
        hmi_state['confidence_threshold'] = threshold

    def forzar_inspeccion():
        """Inspección multi-frame: promedia predicciones del buffer."""
        if hmi_state['model'] is None:
            return (
                _render_clase_html('—', 0, 0.6),
                _render_plc_status(),
                _history_to_table(),
                plc_bridge.get_log(),
            )

        frames = list(frame_buffer)
        if not frames:
            return (
                _render_clase_html('—', 0, hmi_state['confidence_threshold']),
                _render_plc_status(),
                _history_to_table(),
                plc_bridge.get_log(),
            )

        # Clasificar cada frame del buffer
        all_preds = []
        for f in frames:
            p = _classify_frame(f)
            if p is not None:
                all_preds.append(p)

        if not all_preds:
            return (
                _render_clase_html('—', 0, hmi_state['confidence_threshold']),
                _render_plc_status(),
                _history_to_table(),
                plc_bridge.get_log(),
            )

        # Promedio de predicciones
        mean_preds = np.mean(all_preds, axis=0)
        idx = int(np.argmax(mean_preds))
        nombre = hmi_state['class_names'][idx]
        conf = float(mean_preds[idx])
        threshold = hmi_state['confidence_threshold']

        # Guardar captura (último frame)
        if not hmi_state['session_dir']:
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M')
            hmi_state['session_dir'] = os.path.join(BASE_DIR, 'capturas', f'sesion_{ts}')
        os.makedirs(hmi_state['session_dir'], exist_ok=True)

        ts = datetime.datetime.now().strftime('%H%M%S_%f')[:-3]
        path = os.path.join(hmi_state['session_dir'], f'insp_{ts}.jpg')
        frame_bgr = cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, frame_bgr)

        # Registrar en historial
        hmi_state['history'].append({
            'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
            'clase': nombre,
            'clase_idx': idx,
            'confianza': conf,
            'modo': 'Manual',
            'n_frames': len(all_preds),
        })

        # Enviar al PLC si conectado
        if plc_bridge.connected:
            plc_bridge.enviar_resultado(
                idx, conf,
                hmi_state['var_clase'],
                hmi_state['var_confianza'],
            )

        return (
            _render_clase_html(nombre, conf, threshold),
            _render_plc_status(),
            _history_to_table(),
            plc_bridge.get_log(),
        )

    # ── Wiring ──

    cam_feed.stream(
        on_stream,
        inputs=[cam_feed],
        outputs=[result_html],
    )

    btn_cargar.click(
        cargar_modelo,
        inputs=[dd_modelo],
        outputs=[model_info],
    )

    btn_connect.click(
        conectar_plc,
        inputs=[inp_ams, inp_port],
        outputs=[plc_html, plc_log],
    )
    btn_disconnect.click(
        desconectar_plc,
        outputs=[plc_html, plc_log],
    )

    for inp in [inp_var_inicio, inp_var_clase, inp_var_conf]:
        inp.change(
            actualizar_vars,
            inputs=[inp_var_inicio, inp_var_clase, inp_var_conf, inp_threshold],
        )
    inp_threshold.change(
        actualizar_vars,
        inputs=[inp_var_inicio, inp_var_clase, inp_var_conf, inp_threshold],
    )

    btn_manual.click(
        forzar_inspeccion,
        outputs=[result_html, plc_html, history_table, plc_log],
    )
