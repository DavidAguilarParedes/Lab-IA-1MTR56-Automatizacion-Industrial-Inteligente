"""Tab 3: Probar predicciones — con barras animadas."""

import os

import gradio as gr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from app.config import state, DEFAULT_BATCH_SIZE
from app.datos import preprocess_image
from app.ui.pred_html import render_prediction, render_error


def create():

    # ── Predicción individual ──
    with gr.Group(elem_classes="card"):
        gr.Markdown("#### Probar una imagen")
        with gr.Row():
            img_input = gr.Image(
                sources=["upload", "webcam"],
                type="numpy",
                label="Sube una imagen o toma una foto",
                height=300,
            )
            pred_output = gr.HTML(
                value=render_prediction(None, animate=False),
            )
        btn_predecir = gr.Button("Predecir", variant="primary")

    # ── Predicción en tiempo real ──
    with gr.Group(elem_classes="card"):
        gr.Markdown("#### Predicción en tiempo real")
        gr.Markdown(
            "Activa la cámara y las predicciones se actualizan automáticamente.",
            elem_classes="hint",
        )
        with gr.Row():
            webcam_stream = gr.Image(
                sources=["webcam"],
                streaming=True,
                type="numpy",
                label="Cámara",
                height=300,
            )
            stream_output = gr.HTML(
                value=render_prediction(None, live=True),
            )

    # ── Evaluar dataset completo ──
    with gr.Accordion("Evaluar dataset completo", open=False):
        btn_evaluar = gr.Button("Evaluar en datos de validación")
        eval_status = gr.Markdown("")
        plot_cm = gr.Plot(visible=False)
        eval_metrics = gr.Markdown("")

    # ── Handlers ──

    def predecir(image):
        if state["model"] is None:
            return render_error(
                "No hay modelo cargado. Entrena uno en la Tab 2."
            )
        if image is None:
            return render_error(
                "Sube una imagen o toma una foto con el botón de cámara."
            )

        try:
            arr = preprocess_image(
                image, state["img_size"], state["preprocessing"],
            )
            preds = state["model"].predict(arr, verbose=0)[0]
            names = state["class_names"] or [
                f"Clase {i}" for i in range(len(preds))
            ]
            result = {name: float(prob) for name, prob in zip(names, preds)}
            return render_prediction(result, animate=True)
        except Exception as e:
            return render_error(f"Error al predecir: {e}")

    def predecir_stream(image):
        if state["model"] is None or image is None:
            return render_prediction(None, live=True)

        try:
            arr = preprocess_image(
                image, state["img_size"], state["preprocessing"],
            )
            preds = state["model"].predict(arr, verbose=0)[0]
            names = state["class_names"] or [
                f"Clase {i}" for i in range(len(preds))
            ]
            result = {name: float(prob) for name, prob in zip(names, preds)}
            return render_prediction(result, animate=False, live=True)
        except Exception:
            return render_prediction(None, live=True)

    def evaluar_dataset():
        if state["model"] is None:
            return "No hay modelo cargado.", gr.update(), ""
        if not state.get("split_dir"):
            if not state["dataset_dir"]:
                return "No hay dataset cargado.", gr.update(), ""
            return (
                "No hay split de validación. "
                "Haz clic en **Preparar dataset** en la Tab 1.",
                gr.update(), "",
            )

        # Usar el split de validación, no el dataset completo
        val_dir = os.path.join(state["split_dir"], "validation")
        if not os.path.isdir(val_dir):
            return "No se encontró carpeta de validación.", gr.update(), ""

        names = state["class_names"]
        sz = state["img_size"]
        preprocessing = state["preprocessing"]

        if preprocessing == "mobilenet":
            from tensorflow.keras.applications.mobilenet_v2 import (
                preprocess_input,
            )
            gen = ImageDataGenerator(
                preprocessing_function=preprocess_input,
            )
        else:
            gen = ImageDataGenerator(rescale=1.0 / 255)

        flow = gen.flow_from_directory(
            val_dir,
            target_size=(sz, sz),
            batch_size=DEFAULT_BATCH_SIZE,
            class_mode="categorical",
            shuffle=False,
            classes=names,
        )

        preds = state["model"].predict(flow, verbose=0)
        y_pred = np.argmax(preds, axis=1)
        y_true = flow.classes
        labels = list(flow.class_indices.keys())

        acc = np.mean(y_pred == y_true)
        report = classification_report(
            y_true, y_pred, target_names=labels, zero_division=0,
        )

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(
            figsize=(max(5, len(labels) * 1.5), max(4, len(labels) * 1.2)),
        )
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax,
        )
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        plt.tight_layout()

        result = (
            f"Accuracy en validación: **{acc:.1%}** ({len(y_true)} imágenes)",
            gr.update(value=fig, visible=True),
            f"```\n{report}\n```",
        )
        plt.close(fig)
        return result

    # ── Wiring ──

    btn_predecir.click(
        predecir, inputs=[img_input], outputs=[pred_output],
    )
    webcam_stream.stream(
        predecir_stream,
        inputs=[webcam_stream],
        outputs=[stream_output],
        stream_every=0.5,
    )
    btn_evaluar.click(
        evaluar_dataset, outputs=[eval_status, plot_cm, eval_metrics],
    )
