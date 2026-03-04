"""
HMI de Producción — Control de Calidad con PLC Beckhoff.

App standalone fullscreen con CustomTkinter.
Entry point: python -m app.hmi

Layout industrial: cámara en vivo + resultado prominente + log PLC.
Panel de configuración colapsable con snippet de código pyads dinámico.
"""

import os
import sys
import time
import threading
from collections import deque

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk

# Agregar raíz del proyecto al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import BASE_DIR, PLC_DEFAULTS
from app.datos import preprocess_image
from app.plc import plc_bridge
from app.ui.tab_hmi import _discover_models, _load_model_hmi

# ── Configuración ──

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

CAMERA_INDEX = 0
SMOOTHING_WINDOW = 5
BUFFER_SIZE = 10

# Colores semáforo
COLOR_GREEN = "#4caf50"
COLOR_YELLOW = "#ff9800"
COLOR_RED = "#f44336"
COLOR_ACCENT = "#1a73e8"
COLOR_BG_DARK = "#1a1a2e"
COLOR_PANEL = "#16213e"


class HMIApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Estado
        self.model = None
        self.class_names = []
        self.img_size = 128
        self.preprocessing = "rescale"
        self.threshold = PLC_DEFAULTS["confidence_threshold"]

        # Cámara
        self.cap = None
        self.camera_running = False
        self.frame_buffer = deque(maxlen=BUFFER_SIZE)
        self.pred_buffer = deque(maxlen=SMOOTHING_WINDOW)
        self.current_frame = None
        self.fps = 0.0

        # Modelos descubiertos
        self.modelos = _discover_models()

        # Ventana
        self.title("PUCP — Control de Calidad")
        self.geometry("1200x750")
        self.minsize(900, 600)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Config panel visible
        self.config_visible = False

        # Construir UI
        self._build_topbar()
        self._build_main()
        self._build_config()

        # Ocultar config por defecto
        self.config_frame.grid_remove()

        # Iniciar cámara
        self._start_camera()

    # ══════════════════════════════════════
    #  LAYOUT
    # ══════════════════════════════════════

    def _build_topbar(self):
        self.topbar = ctk.CTkFrame(self, height=50, corner_radius=0,
                                   fg_color=COLOR_BG_DARK)
        self.topbar.grid(row=0, column=0, sticky="ew")
        self.topbar.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            self.topbar, text="  PUCP — Control de Calidad",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="white",
        ).grid(row=0, column=0, padx=10, pady=8, sticky="w")

        # PLC status
        self.plc_status_label = ctk.CTkLabel(
            self.topbar, text="● PLC: Desconectado",
            font=ctk.CTkFont(size=13), text_color="#888",
        )
        self.plc_status_label.grid(row=0, column=1, padx=10, sticky="e")

        # Botón config
        self.btn_config_toggle = ctk.CTkButton(
            self.topbar, text="⚙ Config", width=90, height=32,
            font=ctk.CTkFont(size=13),
            fg_color="transparent", border_width=1, border_color="#555",
            command=self._toggle_config,
        )
        self.btn_config_toggle.grid(row=0, column=2, padx=10, pady=8)

    def _build_main(self):
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 5))
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=7)
        self.main_frame.grid_columnconfigure(1, weight=3)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # ── Panel cámara (izquierda) ──
        cam_frame = ctk.CTkFrame(self.main_frame, fg_color=COLOR_BG_DARK,
                                 corner_radius=10)
        cam_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        cam_frame.grid_rowconfigure(0, weight=1)
        cam_frame.grid_columnconfigure(0, weight=1)

        self.cam_label = ctk.CTkLabel(cam_frame, text="Iniciando cámara...",
                                      font=ctk.CTkFont(size=14),
                                      text_color="#888")
        self.cam_label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.fps_label = ctk.CTkLabel(cam_frame, text="",
                                      font=ctk.CTkFont(size=11),
                                      text_color="#666")
        self.fps_label.grid(row=0, column=0, sticky="ne", padx=12, pady=8)

        # ── Panel resultado (derecha) ──
        right_frame = ctk.CTkFrame(self.main_frame, fg_color=COLOR_PANEL,
                                   corner_radius=10)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right_frame.grid_columnconfigure(0, weight=1)

        # Clase
        self.lbl_clase = ctk.CTkLabel(
            right_frame, text="—",
            font=ctk.CTkFont(size=36, weight="bold"),
            text_color="white",
        )
        self.lbl_clase.grid(row=0, column=0, padx=15, pady=(25, 5), sticky="ew")

        # Barra de confianza
        self.progress_bar = ctk.CTkProgressBar(
            right_frame, height=22, corner_radius=8,
            progress_color=COLOR_GREEN,
        )
        self.progress_bar.grid(row=1, column=0, padx=20, pady=(5, 2), sticky="ew")
        self.progress_bar.set(0)

        # Porcentaje
        self.lbl_conf = ctk.CTkLabel(
            right_frame, text="—",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#888",
        )
        self.lbl_conf.grid(row=2, column=0, padx=15, pady=(2, 10), sticky="ew")

        # Frames counter
        self.lbl_frames = ctk.CTkLabel(
            right_frame, text="Frames: 0/10",
            font=ctk.CTkFont(size=13), text_color="#aaa",
        )
        self.lbl_frames.grid(row=3, column=0, padx=15, pady=(0, 10), sticky="ew")

        # Separador
        ctk.CTkFrame(right_frame, height=2, fg_color="#333").grid(
            row=4, column=0, sticky="ew", padx=15, pady=5)

        # Botón FORZAR
        self.btn_forzar = ctk.CTkButton(
            right_frame, text="⚡ FORZAR INSPECCIÓN",
            font=ctk.CTkFont(size=15, weight="bold"),
            height=50, corner_radius=8,
            fg_color=COLOR_ACCENT, hover_color="#1557b0",
            command=self._forzar,
        )
        self.btn_forzar.grid(row=5, column=0, padx=15, pady=10, sticky="ew")

        # Separador
        ctk.CTkFrame(right_frame, height=2, fg_color="#333").grid(
            row=6, column=0, sticky="ew", padx=15, pady=5)

        # Log PLC
        ctk.CTkLabel(
            right_frame, text="Log PLC:",
            font=ctk.CTkFont(size=12), text_color="#888",
            anchor="w",
        ).grid(row=7, column=0, padx=15, pady=(5, 0), sticky="w")

        self.log_textbox = ctk.CTkTextbox(
            right_frame, height=120, font=ctk.CTkFont(size=11),
            fg_color="#0d1117", text_color="#8b949e",
            corner_radius=6, state="disabled",
        )
        self.log_textbox.grid(row=8, column=0, padx=15, pady=(2, 15), sticky="ew")

    def _build_config(self):
        self.config_frame = ctk.CTkFrame(self, fg_color=COLOR_PANEL,
                                         corner_radius=10)
        self.config_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.config_frame.grid_columnconfigure((0, 1), weight=1)

        # ── Fila 1: Modelo + PLC ──
        # Modelo
        modelo_frame = ctk.CTkFrame(self.config_frame, fg_color="transparent")
        modelo_frame.grid(row=0, column=0, padx=15, pady=(10, 5), sticky="ew")
        modelo_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(modelo_frame, text="Modelo:", font=ctk.CTkFont(size=13),
                     text_color="#ccc").grid(row=0, column=0, padx=(0, 8))

        model_choices = list(self.modelos.keys()) if self.modelos else ["(no hay modelos)"]
        self.dd_modelo = ctk.CTkOptionMenu(
            modelo_frame, values=model_choices,
            font=ctk.CTkFont(size=12), width=250,
        )
        self.dd_modelo.grid(row=0, column=1, sticky="ew", padx=(0, 8))

        self.btn_cargar = ctk.CTkButton(
            modelo_frame, text="Cargar", width=80,
            font=ctk.CTkFont(size=12),
            command=self._cargar_modelo,
        )
        self.btn_cargar.grid(row=0, column=2)

        self.lbl_model_info = ctk.CTkLabel(
            modelo_frame, text="Sin modelo cargado",
            font=ctk.CTkFont(size=11), text_color="#888", anchor="w",
        )
        self.lbl_model_info.grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))

        # PLC
        plc_frame = ctk.CTkFrame(self.config_frame, fg_color="transparent")
        plc_frame.grid(row=0, column=1, padx=15, pady=(10, 5), sticky="ew")
        plc_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(plc_frame, text="AMS Net ID:", font=ctk.CTkFont(size=13),
                     text_color="#ccc").grid(row=0, column=0, padx=(0, 8))
        self.inp_ams = ctk.CTkEntry(plc_frame, font=ctk.CTkFont(size=12),
                                    width=180)
        self.inp_ams.insert(0, PLC_DEFAULTS["ams_net_id"])
        self.inp_ams.grid(row=0, column=1, sticky="ew", padx=(0, 8))

        ctk.CTkLabel(plc_frame, text="Puerto:", font=ctk.CTkFont(size=13),
                     text_color="#ccc").grid(row=0, column=2, padx=(0, 5))
        self.inp_port = ctk.CTkEntry(plc_frame, font=ctk.CTkFont(size=12),
                                     width=60)
        self.inp_port.insert(0, str(PLC_DEFAULTS["port"]))
        self.inp_port.grid(row=0, column=3, padx=(0, 8))

        self.btn_conectar = ctk.CTkButton(
            plc_frame, text="Conectar", width=80,
            font=ctk.CTkFont(size=12), fg_color=COLOR_GREEN,
            hover_color="#388e3c", command=self._conectar_plc,
        )
        self.btn_conectar.grid(row=0, column=4, padx=(0, 5))

        self.btn_desconectar = ctk.CTkButton(
            plc_frame, text="Desconectar", width=90,
            font=ctk.CTkFont(size=12), fg_color="#555",
            hover_color="#777", command=self._desconectar_plc,
        )
        self.btn_desconectar.grid(row=0, column=5)

        # ── Fila 2: Variables PLC + Threshold ──
        vars_frame = ctk.CTkFrame(self.config_frame, fg_color="transparent")
        vars_frame.grid(row=1, column=0, padx=15, pady=5, sticky="ew")
        vars_frame.grid_columnconfigure((1, 3), weight=1)

        ctk.CTkLabel(vars_frame, text="Var clase:", font=ctk.CTkFont(size=12),
                     text_color="#aaa").grid(row=0, column=0, padx=(0, 5))
        self.inp_var_clase = ctk.CTkEntry(vars_frame, font=ctk.CTkFont(size=11))
        self.inp_var_clase.insert(0, PLC_DEFAULTS["var_clase"])
        self.inp_var_clase.grid(row=0, column=1, sticky="ew", padx=(0, 15))

        ctk.CTkLabel(vars_frame, text="Var confianza:", font=ctk.CTkFont(size=12),
                     text_color="#aaa").grid(row=0, column=2, padx=(0, 5))
        self.inp_var_conf = ctk.CTkEntry(vars_frame, font=ctk.CTkFont(size=11))
        self.inp_var_conf.insert(0, PLC_DEFAULTS["var_confianza"])
        self.inp_var_conf.grid(row=0, column=3, sticky="ew")

        thresh_frame = ctk.CTkFrame(self.config_frame, fg_color="transparent")
        thresh_frame.grid(row=1, column=1, padx=15, pady=5, sticky="ew")
        thresh_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(thresh_frame, text="Threshold:", font=ctk.CTkFont(size=12),
                     text_color="#aaa").grid(row=0, column=0, padx=(0, 8))
        self.slider_threshold = ctk.CTkSlider(
            thresh_frame, from_=0.1, to=1.0, number_of_steps=18,
            command=self._on_threshold_change,
        )
        self.slider_threshold.set(self.threshold)
        self.slider_threshold.grid(row=0, column=1, sticky="ew", padx=(0, 8))
        self.lbl_threshold = ctk.CTkLabel(
            thresh_frame, text=f"{self.threshold:.0%}",
            font=ctk.CTkFont(size=12), text_color="#ccc", width=40,
        )
        self.lbl_threshold.grid(row=0, column=2)

        # ── Fila 3: Snippet de código PLC ──
        ctk.CTkLabel(
            self.config_frame, text="Código PLC (pyads):",
            font=ctk.CTkFont(size=12), text_color="#888", anchor="w",
        ).grid(row=2, column=0, padx=15, pady=(8, 0), sticky="w")

        self.snippet_textbox = ctk.CTkTextbox(
            self.config_frame, height=100, font=ctk.CTkFont(family="Courier", size=12),
            fg_color="#0d1117", text_color="#58a6ff",
            corner_radius=6, state="disabled",
        )
        self.snippet_textbox.grid(row=3, column=0, columnspan=2,
                                  padx=15, pady=(4, 12), sticky="ew")
        self._update_snippet()

        # Bind para actualizar snippet al cambiar variables
        for entry in (self.inp_ams, self.inp_port, self.inp_var_clase, self.inp_var_conf):
            entry.bind("<KeyRelease>", lambda e: self._update_snippet())

    # ══════════════════════════════════════
    #  CÁMARA
    # ══════════════════════════════════════

    def _start_camera(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            self.cam_label.configure(text="No se pudo abrir la cámara",
                                     text_color=COLOR_RED)
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera_running = True
        self.cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.cam_thread.start()
        self._update_camera()

    def _camera_loop(self):
        """Thread: captura frames + predicción live."""
        prev_time = 0
        while self.camera_running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            # FPS
            now = time.time()
            if prev_time > 0:
                self.fps = 1.0 / (now - prev_time)
            prev_time = now

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame = frame_rgb
            self.frame_buffer.append(frame_rgb.copy())

            # Predicción live con suavizado
            if self.model is not None:
                try:
                    arr = preprocess_image(frame_rgb, self.img_size,
                                           self.preprocessing)
                    preds = self.model.predict(arr, verbose=0)[0]
                    self.pred_buffer.append(preds)
                except Exception:
                    pass

            time.sleep(0.01)  # ~60 fps cap

    def _update_camera(self):
        """Actualiza el canvas de cámara desde el thread principal."""
        if not self.camera_running:
            return

        frame = self.current_frame
        if frame is not None:
            # Ajustar al tamaño del label
            lbl_w = self.cam_label.winfo_width()
            lbl_h = self.cam_label.winfo_height()
            if lbl_w > 10 and lbl_h > 10:
                h, w = frame.shape[:2]
                scale = min(lbl_w / w, lbl_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(frame, (new_w, new_h))
                img = Image.fromarray(resized)
                photo = ImageTk.PhotoImage(img)
                self.cam_label.configure(image=photo, text="")
                self.cam_label._image = photo  # prevent GC

            # FPS
            self.fps_label.configure(text=f"{self.fps:.0f} FPS")

        # Actualizar predicción live
        self._update_live()

        self.after(33, self._update_camera)  # ~30 fps display

    def _update_live(self):
        """Actualiza labels de clase/confianza con suavizado temporal."""
        if len(self.pred_buffer) == 0:
            return

        smooth = np.mean(list(self.pred_buffer), axis=0)
        idx = int(np.argmax(smooth))
        conf = float(smooth[idx])

        if idx < len(self.class_names):
            nombre = self.class_names[idx]
        else:
            nombre = f"Clase {idx}"

        self.lbl_clase.configure(text=nombre)

        # Color semáforo
        if conf >= self.threshold:
            color = COLOR_GREEN
        elif conf >= self.threshold * 0.7:
            color = COLOR_YELLOW
        else:
            color = COLOR_RED

        self.progress_bar.configure(progress_color=color)
        self.progress_bar.set(conf)
        self.lbl_conf.configure(text=f"{conf:.0%}", text_color=color)
        self.lbl_frames.configure(text=f"Frames: {len(self.frame_buffer)}/{BUFFER_SIZE}")

    # ══════════════════════════════════════
    #  ACCIONES
    # ══════════════════════════════════════

    def _forzar(self):
        """Inspección multi-frame: promedia buffer → resultado → envía PLC."""
        if self.model is None:
            self._log("Sin modelo cargado", "⚠")
            return

        frames = list(self.frame_buffer)
        if not frames:
            self._log("Buffer vacío — esperando frames", "⚠")
            return

        # Clasificar cada frame
        all_preds = []
        for f in frames:
            try:
                arr = preprocess_image(f, self.img_size, self.preprocessing)
                preds = self.model.predict(arr, verbose=0)[0]
                all_preds.append(preds)
            except Exception:
                pass

        if not all_preds:
            self._log("No se pudo clasificar ningún frame", "✗")
            return

        # Promedio
        mean_preds = np.mean(all_preds, axis=0)
        idx = int(np.argmax(mean_preds))
        conf = float(mean_preds[idx])
        nombre = self.class_names[idx] if idx < len(self.class_names) else f"Clase {idx}"

        self._log(f"Inspección: {nombre} ({conf:.0%}) [{len(all_preds)} frames]", "✓")

        # Actualizar UI
        color = COLOR_GREEN if conf >= self.threshold else COLOR_RED
        self.lbl_clase.configure(text=nombre)
        self.progress_bar.configure(progress_color=color)
        self.progress_bar.set(conf)
        self.lbl_conf.configure(text=f"{conf:.0%}", text_color=color)

        # Enviar al PLC
        var_clase = self.inp_var_clase.get().strip()
        var_conf = self.inp_var_conf.get().strip()

        if plc_bridge.connected:
            ok, msg = plc_bridge.enviar_resultado(idx, conf, var_clase, var_conf)
            self._log(msg, "✓" if ok else "⚠")
        else:
            self._log(f"PLC no conectado — clase={idx}, conf={conf:.2f}", "—")
            self._log(f"  write_by_name('{var_clase}', {idx}, PLCTYPE_INT)", "→")
            self._log(f"  write_by_name('{var_conf}', {conf:.4f}, PLCTYPE_REAL)", "→")

    def _cargar_modelo(self):
        label = self.dd_modelo.get()
        if label not in self.modelos:
            self._log("Modelo no encontrado", "✗")
            return

        info = self.modelos[label]
        self.btn_cargar.configure(state="disabled", text="Cargando...")
        self.update()

        try:
            names, sz, prep = _load_model_hmi(info["path"])
            # Copiar al estado local (no depender del hmi_state de tab_hmi)
            from app.ui.tab_hmi import hmi_state
            self.model = hmi_state["model"]
            self.class_names = names
            self.img_size = sz
            self.preprocessing = prep
            self.pred_buffer.clear()

            tipo = "MobileNetV2" if prep == "mobilenet" else "CNN custom"
            self.lbl_model_info.configure(
                text=f"{os.path.basename(info['path'])}  |  {tipo}  |  {sz}x{sz}  |  {len(names)} clases",
                text_color="#8b8"
            )
            self._log(f"Modelo cargado: {os.path.basename(info['path'])} ({tipo}, {sz}px)", "✓")
        except Exception as e:
            self._log(f"Error cargando modelo: {e}", "✗")
            self.lbl_model_info.configure(text=f"Error: {e}", text_color=COLOR_RED)
        finally:
            self.btn_cargar.configure(state="normal", text="Cargar")

    def _conectar_plc(self):
        ams = self.inp_ams.get().strip()
        try:
            port = int(self.inp_port.get().strip())
        except ValueError:
            port = 851

        ok, msg = plc_bridge.connect(ams, port)
        self._update_plc_status()
        self._log(msg, "✓" if ok else "⚠")
        self._update_snippet()

    def _desconectar_plc(self):
        plc_bridge.disconnect()
        self._update_plc_status()
        self._log("PLC desconectado", "—")

    def _toggle_config(self):
        self.config_visible = not self.config_visible
        if self.config_visible:
            self.config_frame.grid()
            self.btn_config_toggle.configure(text="⚙ Ocultar")
        else:
            self.config_frame.grid_remove()
            self.btn_config_toggle.configure(text="⚙ Config")

    def _on_threshold_change(self, value):
        self.threshold = float(value)
        self.lbl_threshold.configure(text=f"{self.threshold:.0%}")

    # ══════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════

    def _update_plc_status(self):
        if plc_bridge.connected:
            self.plc_status_label.configure(
                text=f"● PLC: {plc_bridge.ams_net_id}:{plc_bridge.port}",
                text_color=COLOR_GREEN,
            )
        else:
            self.plc_status_label.configure(
                text="● PLC: Desconectado",
                text_color="#888",
            )

    def _update_snippet(self):
        """Actualiza el snippet de código pyads dinámico."""
        ams = self.inp_ams.get().strip() or "5.80.201.232.1.1"
        port = self.inp_port.get().strip() or "851"
        var_clase = self.inp_var_clase.get().strip() or "GVL.nResultadoClase"
        var_conf = self.inp_var_conf.get().strip() or "GVL.rConfianza"

        snippet = (
            f"import pyads\n"
            f"\n"
            f"plc = pyads.Connection('{ams}', {port})\n"
            f"plc.open()\n"
            f"\n"
            f"plc.write_by_name('{var_clase}', 1, pyads.PLCTYPE_INT)\n"
            f"plc.write_by_name('{var_conf}', 0.95, pyads.PLCTYPE_REAL)\n"
            f"\n"
            f"plc.close()"
        )

        self.snippet_textbox.configure(state="normal")
        self.snippet_textbox.delete("1.0", "end")
        self.snippet_textbox.insert("1.0", snippet)
        self.snippet_textbox.configure(state="disabled")

    def _log(self, msg, icon=""):
        """Agrega línea al log PLC visible."""
        ts = time.strftime("%H:%M:%S")
        line = f"{ts}  {icon} {msg}\n"
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", line)
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")

    def _on_close(self):
        """Libera cámara y cierra."""
        self.camera_running = False
        if self.cap is not None:
            self.cap.release()
        self.destroy()


def main():
    app = HMIApp()
    app.mainloop()


if __name__ == "__main__":
    main()
