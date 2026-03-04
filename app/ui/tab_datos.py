"""Tab 1: Crear dataset.

Flujo: definir clases → capturar fotos → eliminar las malas → preparar.
"""

import os
import shutil
import time

import gradio as gr

from app.config import state, BASE_DIR, DEFAULT_IMG_SIZE
from app.datos import (
    scan_dataset,
    split_dataset,
    save_webcam_image,
    list_class_images,
    delete_image,
    delete_class_images,
)


def _resolve(p):
    p = p.strip()
    return os.path.join(BASE_DIR, p) if not os.path.isabs(p) else p


def _summary(dir_abs):
    info = scan_dataset(dir_abs)
    if not info:
        # Verificar si hay al menos una carpeta con fotos
        if os.path.isdir(dir_abs):
            subdirs = [
                d for d in os.listdir(dir_abs)
                if os.path.isdir(os.path.join(dir_abs, d))
                and not d.startswith(".")
            ]
            if len(subdirs) == 1:
                return "Se necesitan al menos **2 clases** para formar un dataset."
        return ""
    parts = [f"**{c}**: {info['counts'][c]}" for c in info["classes"]]
    txt = " · ".join(parts) + f" · Total: **{info['total']}**"
    vals = list(info["counts"].values())
    if vals and min(vals) == 0:
        txt += " · Algunas clases no tienen fotos."
    return txt


def create():

    # ── Clases ──
    with gr.Group(elem_classes="card"):
        clases_input = gr.Textbox(
            label="Clases (separar con coma)",
            value="clase_1, clase_2",
        )
        clase_activa = gr.Dropdown(
            label="Clase activa",
            choices=["clase_1", "clase_2"],
            value="clase_1",
            interactive=True,
        )

    # ── Captura ──
    with gr.Group(elem_classes="card"):
        with gr.Row():
            with gr.Column(scale=1, min_width=240):
                webcam = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    type="numpy",
                    label="Webcam",
                    height=220,
                )
                btn_grabar = gr.Button(
                    "Iniciar captura",
                    variant="primary",
                    elem_classes="btn-record",
                )

            with gr.Column(scale=2, min_width=280):
                gallery_label = gr.Markdown("Sin fotos aún")
                galeria = gr.Gallery(
                    columns=5,
                    rows=3,
                    height=220,
                    object_fit="cover",
                    elem_classes="tm-gallery",
                    show_label=False,
                )
                with gr.Row():
                    btn_borrar_foto = gr.Button(
                        "Eliminar foto seleccionada", size="sm",
                        visible=False, elem_classes="btn-sm",
                    )
                    btn_borrar_todas = gr.Button(
                        "Eliminar todas", size="sm",
                        visible=False, elem_classes="btn-sm",
                    )

        rec_status = gr.Markdown("")
        resumen = gr.Markdown("")

    # ── Preparar ──
    with gr.Group(elem_classes="card"):
        btn_preparar = gr.Button("Preparar dataset", variant="primary")
        status_final = gr.Markdown("")

    # ── Avanzado ──
    with gr.Accordion("Opciones avanzadas", open=False):
        output_dir = gr.Textbox(label="Carpeta de salida", value="data/mi_dataset")
        img_size_radio = gr.Radio(
            [64, 128, 224], value=DEFAULT_IMG_SIZE, label="Resolución (px)",
        )
        split_slider = gr.Slider(
            0.6, 0.9, value=0.8, step=0.05, label="Proporción train/val",
        )

    with gr.Accordion("Cargar desde carpeta existente", open=False):
        with gr.Row():
            ruta_carpeta = gr.Textbox(
                label="Ruta", value="data/botellas", scale=3,
            )
            btn_cargar = gr.Button("Cargar", scale=1)

    # ── Estado interno ──
    rec_state = gr.State({
        "active": False, "class_name": "", "count": 0,
        "start": 0.0, "last_status": "",
    })
    selected_idx = gr.State(-1)

    # ── Helpers ──

    def _refresh(dir_abs, clase):
        """Returns: gallery, label, btn_foto_vis, btn_todas_vis."""
        if not clase:
            return (gr.update(value=[]), "Sin fotos aún",
                    gr.update(visible=False), gr.update(visible=False))
        imgs = list_class_images(dir_abs, clase)
        n = len(imgs)
        label = f"**{clase}** — {n} fotos" if n else f"**{clase}** — sin fotos"
        has = n > 0
        return (gr.update(value=imgs), label,
                gr.update(visible=has), gr.update(visible=has))

    # ── Handlers ──

    def actualizar_clases(clases_str):
        clases = [c.strip() for c in clases_str.split(",") if c.strip()]
        if not clases:
            return gr.update(choices=[], value=None)
        return gr.update(choices=clases, value=clases[0])

    def cambiar_clase(clase, out_dir):
        dir_abs = _resolve(out_dir)
        gal, label, bf, bt = _refresh(dir_abs, clase)
        return gal, label, bf, bt, _summary(dir_abs), -1

    def toggle_captura(clase, rec):
        """Click para iniciar/detener captura."""
        if not clase:
            return rec, "Selecciona una clase primero.", gr.update()
        if rec["active"]:
            # Detener
            rec["active"] = False
            rec["last_status"] = (
                f"{rec['count']} fotos de **{rec['class_name']}** capturadas"
            )
            return (rec, rec["last_status"],
                    gr.update(value="Iniciar captura", variant="primary"))
        else:
            # Iniciar
            rec.update(
                active=True, class_name=clase, count=0, start=time.time(),
            )
            rec["last_status"] = f"Capturando **{clase}**..."
            return (rec, rec["last_status"],
                    gr.update(value="Detener captura", variant="stop"))

    def on_frame(frame, rec, out_dir_str, clase):
        """Stream: captura frames y actualiza galería en vivo."""
        if frame is None or not rec.get("active"):
            return (rec, rec.get("last_status", ""),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update())

        dir_abs = _resolve(out_dir_str)
        save_webcam_image(frame, rec["class_name"], dir_abs)
        rec["count"] += 1
        rec["last_status"] = (
            f"Capturando **{rec['class_name']}** — {rec['count']} fotos"
        )

        # Actualizar galería en vivo (cada 3 frames para no saturar)
        if rec["count"] % 3 == 1:
            gal, label, bf, bt = _refresh(dir_abs, clase)
            info = scan_dataset(dir_abs)
            if info:
                state["dataset_dir"] = dir_abs
                state["class_names"] = info["classes"]
            return (rec, rec["last_status"],
                    gal, label, bf, bt, _summary(dir_abs))

        return (rec, rec["last_status"],
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update())

    def on_select(evt: gr.SelectData):
        return evt.index

    def borrar_foto(idx, clase, out_dir):
        if idx < 0 or not clase:
            return (gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), -1)
        dir_abs = _resolve(out_dir)
        imgs = list_class_images(dir_abs, clase)
        if 0 <= idx < len(imgs):
            delete_image(imgs[idx])
        gal, label, bf, bt = _refresh(dir_abs, clase)
        return gal, label, bf, bt, _summary(dir_abs), -1

    def borrar_todas(clase, out_dir):
        if not clase:
            return (gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update())
        dir_abs = _resolve(out_dir)
        delete_class_images(dir_abs, clase)
        gal, label, bf, bt = _refresh(dir_abs, clase)
        return gal, label, bf, bt, _summary(dir_abs)

    def cargar_carpeta(ruta_str, img_size):
        ruta_abs = _resolve(ruta_str)
        info = scan_dataset(ruta_abs)
        if info is None:
            return ("No se encontró dataset válido.",
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update())
        state["dataset_dir"] = ruta_abs
        state["class_names"] = info["classes"]
        state["img_size"] = img_size
        clase0 = info["classes"][0]
        gal, label, bf, bt = _refresh(ruta_abs, clase0)
        return (
            _summary(ruta_abs), gal, label, bf, bt,
            ", ".join(info["classes"]),
            gr.update(choices=info["classes"], value=clase0),
        )

    def preparar(img_size, split_ratio, out_dir):
        dir_abs = _resolve(out_dir)
        ds_dir = state["dataset_dir"] or dir_abs
        info = scan_dataset(ds_dir)
        if not info:
            # Dar error específico
            if os.path.isdir(ds_dir):
                subdirs = [
                    d for d in os.listdir(ds_dir)
                    if os.path.isdir(os.path.join(ds_dir, d))
                    and not d.startswith(".")
                ]
                if len(subdirs) == 1:
                    return (
                        "Se necesitan al menos **2 clases** con fotos. "
                        f"Solo se encontró: {subdirs[0]}"
                    )
                if len(subdirs) == 0:
                    return "No hay fotos capturadas. Captura fotos primero."
            return "No se encontró dataset. Captura fotos o carga una carpeta."
        state["dataset_dir"] = ds_dir
        state["class_names"] = info["classes"]
        state["img_size"] = img_size

        split_dir = ds_dir.rstrip("/") + "_split"
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        split_dataset(ds_dir, split_dir, info["classes"], split_ratio)
        state["split_dir"] = split_dir
        n_train = sum(
            len(os.listdir(os.path.join(split_dir, "train", c)))
            for c in info["classes"]
        )
        n_val = sum(
            len(os.listdir(os.path.join(split_dir, "validation", c)))
            for c in info["classes"]
        )
        return (
            f"Dataset listo: **{n_train}** train + **{n_val}** val "
            f"({len(info['classes'])} clases, {img_size}px). "
            f"Continúa en **Tab 2**."
        )

    # ── Wiring ──

    clases_input.change(actualizar_clases, [clases_input], [clase_activa])

    clase_activa.change(
        cambiar_clase, [clase_activa, output_dir],
        [galeria, gallery_label, btn_borrar_foto, btn_borrar_todas,
         resumen, selected_idx],
    )

    btn_grabar.click(
        toggle_captura, [clase_activa, rec_state],
        [rec_state, rec_status, btn_grabar],
    )

    webcam.stream(
        on_frame, [webcam, rec_state, output_dir, clase_activa],
        [rec_state, rec_status, galeria, gallery_label,
         btn_borrar_foto, btn_borrar_todas, resumen],
        stream_every=0.3,
    )

    galeria.select(on_select, outputs=[selected_idx])

    btn_borrar_foto.click(
        borrar_foto, [selected_idx, clase_activa, output_dir],
        [galeria, gallery_label, btn_borrar_foto, btn_borrar_todas,
         resumen, selected_idx],
    )

    btn_borrar_todas.click(
        borrar_todas, [clase_activa, output_dir],
        [galeria, gallery_label, btn_borrar_foto, btn_borrar_todas, resumen],
    )

    btn_cargar.click(
        cargar_carpeta, [ruta_carpeta, img_size_radio],
        [resumen, galeria, gallery_label, btn_borrar_foto, btn_borrar_todas,
         clases_input, clase_activa],
    )

    btn_preparar.click(
        preparar, [img_size_radio, split_slider, output_dir], [status_final],
    )
