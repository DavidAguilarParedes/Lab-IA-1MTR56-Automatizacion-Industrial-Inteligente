"""
HMI de Producción — Control de Calidad con PLC Beckhoff.

App standalone fullscreen con CustomTkinter.
Entry point: python -m app.hmi

Layout industrial: cámara en vivo + resultado prominente + log PLC.
Panel de configuración lateral colapsable con snippet de código pyads dinámico.
"""

import os
import sys
import json
import time
import threading
from collections import deque
from tkinter import filedialog

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk

# Agregar raíz del proyecto al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import BASE_DIR, PLC_DEFAULTS
from app.datos import preprocess_image
from app.plc import plc_bridge
from app.ui.tab_hmi import _discover_models

# ── Configuración ──

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

CAMERA_INDEX = 0
SMOOTHING_WINDOW = 5
BUFFER_SIZE = 10

# ── Paleta ──
BG = "#202027"             # fondo ventana (más claro que antes)
BG_SURFACE = "#2a2a32"     # paneles principales
BG_RAISED = "#34343e"      # cards, inputs hover
BG_INSET = "#181820"       # áreas hundidas (cámara, snippet, log)
BORDER = "#3e3e48"         # bordes normales
BORDER_HI = "#52525e"      # bordes activos

# Texto — más legible
TEXT_HI = "#f4f4f7"        # clase detectada, títulos
TEXT = "#d4d4db"           # texto principal (legible)
TEXT_LO = "#9e9eab"        # labels secundarios (aun legible)
TEXT_DIM = "#6b6b78"       # placeholders, hints

# Funcional
OK = "#34d399"
OK_SOFT = "#065f46"
WARN = "#fbbf24"
FAIL = "#f87171"
ACCENT = "#60a5fa"
ACCENT_HOVER = "#3b82f6"
LIVE_DOT = "#f87171"

# Config panel width
CONFIG_WIDTH = 310


class HMIApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.configure(fg_color=BG)

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

        # Ventana — título solo en la barra de Windows
        self.title("PUCP — Control de Calidad")
        self.geometry("1200x720")
        self.minsize(900, 550)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Config panel visible
        self.config_visible = False
        self._live_dot_on = True

        # Layout raíz: topbar (row 0) + main (row 1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._build_topbar()
        self._build_main()
        self._build_config_panel()

        self._start_camera()
        self._pulse_live_dot()

    # ══════════════════════════════════════
    #  TOPBAR — compacta, sin duplicar título
    # ══════════════════════════════════════

    def _build_topbar(self):
        self.topbar = ctk.CTkFrame(self, height=34, corner_radius=0,
                                   fg_color=BG_SURFACE)
        self.topbar.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.topbar.grid_columnconfigure(1, weight=1)
        self.topbar.grid_propagate(False)

        # Línea de acento superior
        accent = ctk.CTkFrame(self, height=2, corner_radius=0, fg_color=ACCENT)
        accent.grid(row=0, column=0, columnspan=2, sticky="new")
        accent.lift()

        # Live dot
        self.live_dot = ctk.CTkLabel(
            self.topbar, text="●", font=ctk.CTkFont(size=10),
            text_color=LIVE_DOT, width=20,
        )
        self.live_dot.grid(row=0, column=0, padx=(12, 0))

        # PLC status
        self.plc_status_label = ctk.CTkLabel(
            self.topbar, text="PLC: Desconectado",
            font=ctk.CTkFont(size=11), text_color=TEXT_DIM,
        )
        self.plc_status_label.grid(row=0, column=1, padx=10, sticky="e")

        # Botón config toggle
        self.btn_config_toggle = ctk.CTkButton(
            self.topbar, text="Configuracion", width=105, height=22,
            font=ctk.CTkFont(size=11),
            fg_color="transparent", border_width=1, border_color=BORDER,
            text_color=TEXT_LO, hover_color=BG_RAISED, corner_radius=3,
            command=self._toggle_config,
        )
        self.btn_config_toggle.grid(row=0, column=2, padx=(0, 10), pady=6)

    # ══════════════════════════════════════
    #  MAIN — cámara + resultado
    # ══════════════════════════════════════

    def _build_main(self):
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=1, column=0, sticky="nsew", padx=6, pady=(3, 6))
        self.main_frame.grid_columnconfigure(0, weight=7)
        self.main_frame.grid_columnconfigure(1, weight=3)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # ── Cámara (izquierda) ──
        cam_frame = ctk.CTkFrame(self.main_frame, fg_color=BG_INSET,
                                 corner_radius=4, border_width=1,
                                 border_color=BORDER)
        cam_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
        cam_frame.grid_rowconfigure(0, weight=1)
        cam_frame.grid_columnconfigure(0, weight=1)

        self.cam_label = ctk.CTkLabel(cam_frame, text="Iniciando camara...",
                                      font=ctk.CTkFont(size=12),
                                      text_color=TEXT_DIM)
        self.cam_label.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        self.fps_label = ctk.CTkLabel(cam_frame, text="",
                                      font=ctk.CTkFont(family="Courier", size=9),
                                      text_color=TEXT_DIM, fg_color=BG_INSET)
        self.fps_label.grid(row=0, column=0, sticky="se", padx=8, pady=6)

        # ── Resultado (derecha) ──
        self.right_frame = ctk.CTkFrame(self.main_frame, fg_color=BG_SURFACE,
                                        corner_radius=4, border_width=1,
                                        border_color=BORDER)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=(3, 0))
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(8, weight=1)  # log se expande

        ctk.CTkLabel(
            self.right_frame, text="RESULTADO",
            font=ctk.CTkFont(size=10), text_color=TEXT_DIM, anchor="w",
        ).grid(row=0, column=0, padx=14, pady=(10, 0), sticky="w")

        self.lbl_clase = ctk.CTkLabel(
            self.right_frame, text="—",
            font=ctk.CTkFont(size=30, weight="bold"),
            text_color=TEXT_HI,
        )
        self.lbl_clase.grid(row=1, column=0, padx=14, pady=(0, 4), sticky="ew")

        # Barra de confianza
        self.bar_track = ctk.CTkFrame(self.right_frame, height=14,
                                      fg_color=BG_INSET, corner_radius=3)
        self.bar_track.grid(row=2, column=0, padx=14, pady=(0, 2), sticky="ew")
        self.bar_track.grid_propagate(False)

        self.bar_fill = ctk.CTkFrame(self.bar_track, height=14,
                                     fg_color=TEXT_DIM, corner_radius=3)
        self.bar_fill.place(relx=0, rely=0, relwidth=0.0, relheight=1.0)

        # Porcentaje + frames
        stats = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        stats.grid(row=3, column=0, padx=14, pady=(2, 0), sticky="ew")
        stats.grid_columnconfigure(0, weight=1)

        self.lbl_conf = ctk.CTkLabel(
            stats, text="—",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=TEXT_DIM, anchor="w",
        )
        self.lbl_conf.grid(row=0, column=0, sticky="w")

        self.lbl_frames = ctk.CTkLabel(
            stats, text="0/10",
            font=ctk.CTkFont(family="Courier", size=10),
            text_color=TEXT_DIM, anchor="e",
        )
        self.lbl_frames.grid(row=0, column=1, sticky="e")

        # Botón EJECUTAR
        ctk.CTkFrame(self.right_frame, height=1, fg_color=BORDER).grid(
            row=4, column=0, sticky="ew", padx=10, pady=(10, 0))

        self.btn_forzar = ctk.CTkButton(
            self.right_frame, text="EJECUTAR INSPECCION",
            font=ctk.CTkFont(size=12, weight="bold"),
            height=40, corner_radius=4,
            fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color="white",
            command=self._forzar,
        )
        self.btn_forzar.grid(row=5, column=0, padx=14, pady=(10, 0), sticky="ew")

        # Info modelo
        self.lbl_model_status = ctk.CTkLabel(
            self.right_frame, text="Sin modelo",
            font=ctk.CTkFont(size=10), text_color=TEXT_DIM, anchor="w",
        )
        self.lbl_model_status.grid(row=6, column=0, padx=14, pady=(6, 0), sticky="w")

        # Log
        ctk.CTkFrame(self.right_frame, height=1, fg_color=BORDER).grid(
            row=7, column=0, sticky="ew", padx=10, pady=(8, 0))

        self.log_textbox = ctk.CTkTextbox(
            self.right_frame,
            font=ctk.CTkFont(family="Courier", size=10),
            fg_color=BG_INSET, text_color=TEXT_LO,
            corner_radius=3, border_width=0,
            state="disabled", wrap="none",
        )
        self.log_textbox.grid(row=8, column=0, padx=8, pady=(4, 8), sticky="nsew")

    # ══════════════════════════════════════
    #  CONFIG — panel lateral derecho
    # ══════════════════════════════════════

    def _build_config_panel(self):
        """Panel de configuración lateral (columna 1 de root, al lado de main)."""
        self.config_frame = ctk.CTkScrollableFrame(
            self, fg_color=BG_SURFACE, corner_radius=4,
            border_width=1, border_color=BORDER,
            width=CONFIG_WIDTH,
        )
        # Se coloca en row=1, column=1 (al lado de main_frame)
        self.config_frame.grid(row=1, column=1, sticky="nsew",
                               padx=(0, 6), pady=(3, 6))
        self.config_frame.grid_columnconfigure(0, weight=1)
        # Empieza oculto
        self.config_frame.grid_remove()

        # ── MODELO ──
        ctk.CTkLabel(self.config_frame, text="MODELO",
                     font=ctk.CTkFont(size=9, weight="bold"),
                     text_color=TEXT_DIM).grid(
            row=0, column=0, padx=10, pady=(10, 4), sticky="w")

        model_choices = list(self.modelos.keys()) if self.modelos else ["(sin modelos)"]
        self.dd_modelo = ctk.CTkOptionMenu(
            self.config_frame, values=model_choices,
            font=ctk.CTkFont(size=10), height=26,
            fg_color=BG_INSET, button_color=BORDER,
            button_hover_color=BORDER_HI,
            dropdown_fg_color=BG_RAISED, dropdown_hover_color=BG_SURFACE,
            text_color=TEXT,
        )
        self.dd_modelo.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 4))

        btn_modelo_row = ctk.CTkFrame(self.config_frame, fg_color="transparent")
        btn_modelo_row.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 8))
        btn_modelo_row.grid_columnconfigure(0, weight=1)

        self.btn_cargar = ctk.CTkButton(
            btn_modelo_row, text="Cargar", height=28,
            font=ctk.CTkFont(size=11),
            fg_color=BG_INSET, hover_color=BORDER,
            text_color=TEXT, corner_radius=3,
            command=self._cargar_modelo,
        )
        self.btn_cargar.grid(row=0, column=0, sticky="ew", padx=(0, 3))

        self.btn_buscar = ctk.CTkButton(
            btn_modelo_row, text="Buscar...", height=28, width=75,
            font=ctk.CTkFont(size=11),
            fg_color="transparent", hover_color=BG_INSET,
            text_color=TEXT_LO, corner_radius=3,
            border_width=1, border_color=BORDER,
            command=self._buscar_modelo,
        )
        self.btn_buscar.grid(row=0, column=1)

        # ── CONEXION PLC ──
        self._config_sep(3)
        ctk.CTkLabel(self.config_frame, text="CONEXION PLC",
                     font=ctk.CTkFont(size=9, weight="bold"),
                     text_color=TEXT_DIM).grid(
            row=4, column=0, padx=10, pady=(8, 4), sticky="w")

        # AMS
        ctk.CTkLabel(self.config_frame, text="AMS Net ID",
                     font=ctk.CTkFont(size=9), text_color=TEXT_LO).grid(
            row=5, column=0, padx=10, pady=(0, 1), sticky="w")
        self.inp_ams = ctk.CTkEntry(
            self.config_frame, font=ctk.CTkFont(size=11), height=26,
            fg_color=BG_INSET, border_color=BORDER, text_color=TEXT,
        )
        self.inp_ams.insert(0, PLC_DEFAULTS["ams_net_id"])
        self.inp_ams.grid(row=6, column=0, sticky="ew", padx=10, pady=(0, 4))

        # Puerto
        ctk.CTkLabel(self.config_frame, text="Puerto",
                     font=ctk.CTkFont(size=9), text_color=TEXT_LO).grid(
            row=7, column=0, padx=10, pady=(0, 1), sticky="w")
        self.inp_port = ctk.CTkEntry(
            self.config_frame, font=ctk.CTkFont(size=11), height=26,
            fg_color=BG_INSET, border_color=BORDER, text_color=TEXT,
        )
        self.inp_port.insert(0, str(PLC_DEFAULTS["port"]))
        self.inp_port.grid(row=8, column=0, sticky="ew", padx=10, pady=(0, 6))

        # Botones PLC
        btn_row = ctk.CTkFrame(self.config_frame, fg_color="transparent")
        btn_row.grid(row=9, column=0, padx=10, pady=(0, 8), sticky="ew")
        btn_row.grid_columnconfigure((0, 1), weight=1)

        self.btn_conectar = ctk.CTkButton(
            btn_row, text="Conectar", height=26,
            font=ctk.CTkFont(size=10),
            fg_color=BG_INSET, hover_color=OK_SOFT,
            text_color=OK, corner_radius=3,
            command=self._conectar_plc,
        )
        self.btn_conectar.grid(row=0, column=0, sticky="ew", padx=(0, 3))

        self.btn_desconectar = ctk.CTkButton(
            btn_row, text="Desconectar", height=26,
            font=ctk.CTkFont(size=10),
            fg_color="transparent", hover_color=BG_INSET,
            text_color=TEXT_DIM, corner_radius=3,
            command=self._desconectar_plc,
        )
        self.btn_desconectar.grid(row=0, column=1, sticky="ew", padx=(3, 0))

        # ── VARIABLES PLC ──
        self._config_sep(10)
        ctk.CTkLabel(self.config_frame, text="VARIABLES PLC",
                     font=ctk.CTkFont(size=9, weight="bold"),
                     text_color=TEXT_DIM).grid(
            row=11, column=0, padx=10, pady=(8, 4), sticky="w")

        ctk.CTkLabel(self.config_frame, text="Variable clase (INT)",
                     font=ctk.CTkFont(size=9), text_color=TEXT_LO).grid(
            row=12, column=0, padx=10, pady=(0, 1), sticky="w")
        self.inp_var_clase = ctk.CTkEntry(
            self.config_frame,
            font=ctk.CTkFont(family="Courier", size=10),
            height=24, fg_color=BG_INSET, border_color=BORDER, text_color=TEXT,
        )
        self.inp_var_clase.insert(0, PLC_DEFAULTS["var_clase"])
        self.inp_var_clase.grid(row=13, column=0, sticky="ew", padx=10, pady=(0, 4))

        ctk.CTkLabel(self.config_frame, text="Variable confianza (REAL)",
                     font=ctk.CTkFont(size=9), text_color=TEXT_LO).grid(
            row=14, column=0, padx=10, pady=(0, 1), sticky="w")
        self.inp_var_conf = ctk.CTkEntry(
            self.config_frame,
            font=ctk.CTkFont(family="Courier", size=10),
            height=24, fg_color=BG_INSET, border_color=BORDER, text_color=TEXT,
        )
        self.inp_var_conf.insert(0, PLC_DEFAULTS["var_confianza"])
        self.inp_var_conf.grid(row=15, column=0, sticky="ew", padx=10, pady=(0, 6))

        # Threshold
        ctk.CTkLabel(self.config_frame, text="Umbral de confianza",
                     font=ctk.CTkFont(size=9), text_color=TEXT_LO).grid(
            row=16, column=0, padx=10, pady=(0, 1), sticky="w")

        thresh_row = ctk.CTkFrame(self.config_frame, fg_color="transparent")
        thresh_row.grid(row=17, column=0, sticky="ew", padx=10, pady=(0, 8))
        thresh_row.grid_columnconfigure(0, weight=1)

        self.slider_threshold = ctk.CTkSlider(
            thresh_row, from_=0.1, to=1.0, number_of_steps=18,
            height=14, fg_color=BG_INSET, progress_color=BORDER_HI,
            button_color=TEXT_LO, button_hover_color=TEXT,
            command=self._on_threshold_change,
        )
        self.slider_threshold.set(self.threshold)
        self.slider_threshold.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.lbl_threshold = ctk.CTkLabel(
            thresh_row, text=f"{self.threshold:.0%}",
            font=ctk.CTkFont(family="Courier", size=11),
            text_color=TEXT, width=35,
        )
        self.lbl_threshold.grid(row=0, column=1)

        # ── SNIPPET CODIGO PLC ──
        self._config_sep(18)
        ctk.CTkLabel(self.config_frame, text="CODIGO PLC (pyads)",
                     font=ctk.CTkFont(size=9, weight="bold"),
                     text_color=TEXT_DIM).grid(
            row=19, column=0, padx=10, pady=(8, 2), sticky="w")

        self.snippet_textbox = ctk.CTkTextbox(
            self.config_frame, height=130,
            font=ctk.CTkFont(family="Courier", size=10),
            fg_color=BG_INSET, text_color="#93c5fd",
            corner_radius=3, border_width=1, border_color=BORDER,
            state="disabled",
        )
        self.snippet_textbox.grid(row=20, column=0, padx=10, pady=(0, 12),
                                  sticky="ew")
        self._update_snippet()

        # Binds
        for entry in (self.inp_ams, self.inp_port, self.inp_var_clase, self.inp_var_conf):
            entry.bind("<KeyRelease>", lambda e: self._update_snippet())

    def _config_sep(self, row):
        """Separador horizontal en el panel de config."""
        ctk.CTkFrame(self.config_frame, height=1, fg_color=BORDER).grid(
            row=row, column=0, sticky="ew", padx=8)

    # ══════════════════════════════════════
    #  CÁMARA
    # ══════════════════════════════════════

    def _start_camera(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            self.cam_label.configure(text="No se pudo abrir la camara",
                                     text_color=FAIL)
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera_running = True
        self.cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.cam_thread.start()
        self._update_camera()

    def _camera_loop(self):
        prev_time = 0
        while self.camera_running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.1)
                continue
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            now = time.time()
            if prev_time > 0:
                self.fps = 1.0 / (now - prev_time)
            prev_time = now

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame = frame_rgb
            self.frame_buffer.append(frame_rgb.copy())

            if self.model is not None:
                try:
                    arr = preprocess_image(frame_rgb, self.img_size,
                                           self.preprocessing)
                    preds = self.model.predict(arr, verbose=0)[0]
                    self.pred_buffer.append(preds)
                except Exception:
                    pass
            time.sleep(0.01)

    def _update_camera(self):
        if not self.camera_running:
            return
        frame = self.current_frame
        if frame is not None:
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
                self.cam_label._image = photo
            self.fps_label.configure(text=f"{self.fps:.0f} FPS")
        self._update_live()
        self.after(33, self._update_camera)

    def _update_live(self):
        if len(self.pred_buffer) == 0:
            return
        smooth = np.mean(list(self.pred_buffer), axis=0)
        idx = int(np.argmax(smooth))
        conf = float(smooth[idx])

        nombre = self.class_names[idx] if idx < len(self.class_names) else f"Clase {idx}"
        self.lbl_clase.configure(text=nombre)

        if conf >= self.threshold:
            c = OK
        elif conf >= self.threshold * 0.7:
            c = WARN
        else:
            c = FAIL

        self.bar_fill.configure(fg_color=c)
        self.bar_fill.place(relx=0, rely=0, relwidth=max(0.01, conf),
                            relheight=1.0)
        self.lbl_conf.configure(text=f"{conf:.0%}", text_color=c)
        self.lbl_frames.configure(text=f"{len(self.frame_buffer)}/{BUFFER_SIZE}")

    def _pulse_live_dot(self):
        if self.camera_running and self.current_frame is not None:
            self._live_dot_on = not self._live_dot_on
            self.live_dot.configure(
                text_color=LIVE_DOT if self._live_dot_on else BG_SURFACE)
        else:
            self.live_dot.configure(text_color=TEXT_DIM)
        self.after(600, self._pulse_live_dot)

    # ══════════════════════════════════════
    #  ACCIONES
    # ══════════════════════════════════════

    def _forzar(self):
        if self.model is None:
            self._log("Sin modelo cargado")
            return
        frames = list(self.frame_buffer)
        if not frames:
            self._log("Buffer vacio")
            return

        self.btn_forzar.configure(fg_color=ACCENT_HOVER, text="...")
        self.update()

        all_preds = []
        for f in frames:
            try:
                arr = preprocess_image(f, self.img_size, self.preprocessing)
                preds = self.model.predict(arr, verbose=0)[0]
                all_preds.append(preds)
            except Exception:
                pass

        self.btn_forzar.configure(fg_color=ACCENT, text="EJECUTAR INSPECCION")

        if not all_preds:
            self._log("ERR  Sin predicciones")
            return

        mean_preds = np.mean(all_preds, axis=0)
        idx = int(np.argmax(mean_preds))
        conf = float(mean_preds[idx])
        nombre = self.class_names[idx] if idx < len(self.class_names) else f"Clase {idx}"

        self._log(f"OK   {nombre}  {conf:.0%}  [{len(all_preds)}f]")

        bar_color = OK if conf >= self.threshold else FAIL
        self.lbl_clase.configure(text=nombre, text_color="white")
        self.bar_fill.configure(fg_color=bar_color)
        self.bar_fill.place(relx=0, rely=0, relwidth=max(0.01, conf),
                            relheight=1.0)
        self.lbl_conf.configure(text=f"{conf:.0%}", text_color=bar_color)

        # Flash borde del panel
        self.right_frame.configure(border_color=bar_color)
        self.after(800, lambda: self.right_frame.configure(border_color=BORDER))
        self.after(500, lambda: self.lbl_clase.configure(text_color=TEXT_HI))

        # PLC
        var_clase = self.inp_var_clase.get().strip()
        var_conf = self.inp_var_conf.get().strip()

        if plc_bridge.connected:
            ok, msg = plc_bridge.enviar_resultado(idx, conf, var_clase, var_conf)
            self._log(f"{'OK' if ok else 'WARN'}  {msg}")
        else:
            self._log(f"---  PLC offline  clase={idx}  conf={conf:.2f}")
            self._log(f"     write('{var_clase}', {idx})")
            self._log(f"     write('{var_conf}', {conf:.4f})")

    def _buscar_modelo(self):
        """Abre file dialog para buscar un .h5/.keras en cualquier ruta."""
        path = filedialog.askopenfilename(
            title="Seleccionar modelo",
            filetypes=[
                ("Modelos Keras", "*.h5 *.keras"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if path:
            self._load_model_from_path(path)

    def _cargar_modelo(self):
        """Carga el modelo seleccionado en el dropdown."""
        label = self.dd_modelo.get()
        if label not in self.modelos:
            self._log("ERR  Modelo no encontrado")
            return
        self._load_model_from_path(self.modelos[label]["path"])

    def _load_model_from_path(self, model_path):
        """Carga un modelo .h5/.keras desde ruta absoluta."""
        if not os.path.isfile(model_path):
            self._log(f"ERR  Archivo no existe: {model_path}")
            return

        self.btn_cargar.configure(state="disabled", text="...")
        self.update()

        try:
            from tensorflow.keras.models import load_model

            model = load_model(model_path, compile=False)
            model.compile(optimizer="adam", loss="categorical_crossentropy",
                          metrics=["accuracy"])

            img_size = model.input_shape[1]
            preprocessing = "rescale"
            for layer in model.layers:
                if "mobilenet" in layer.name.lower():
                    preprocessing = "mobilenet"
                    break

            # Leer metadata JSON si existe
            json_path = model_path.rsplit(".", 1)[0] + ".json"
            class_names = []
            if os.path.exists(json_path):
                with open(json_path) as f:
                    meta = json.load(f)
                class_names = meta.get("class_names", [])
                if meta.get("preprocessing"):
                    preprocessing = meta["preprocessing"]

            if not class_names:
                n = model.output_shape[-1]
                class_names = [f"Clase {i}" for i in range(n)]

            self.model = model
            self.class_names = class_names
            self.img_size = img_size
            self.preprocessing = preprocessing
            self.pred_buffer.clear()

            tipo = "MobileNet" if preprocessing == "mobilenet" else "CNN"
            fname = os.path.basename(model_path)
            self.lbl_model_status.configure(
                text=f"{fname}  |  {tipo} {img_size}px  |  {len(class_names)} clases",
                text_color=TEXT_LO,
            )
            self._log(f"OK   {fname} ({tipo}, {img_size}px, {len(class_names)} clases)")
        except Exception as e:
            self._log(f"ERR  {e}")
            self.lbl_model_status.configure(text=f"Error: {e}", text_color=FAIL)
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
        self._log(f"{'OK' if ok else 'WARN'}  {msg}")
        self._update_snippet()

    def _desconectar_plc(self):
        plc_bridge.disconnect()
        self._update_plc_status()
        self._log("---  PLC desconectado")

    def _toggle_config(self):
        self.config_visible = not self.config_visible
        if self.config_visible:
            self.config_frame.grid()
            self.btn_config_toggle.configure(text="Ocultar",
                                             border_color=BORDER_HI)
        else:
            self.config_frame.grid_remove()
            self.btn_config_toggle.configure(text="Configuracion",
                                             border_color=BORDER)

    def _on_threshold_change(self, value):
        self.threshold = float(value)
        self.lbl_threshold.configure(text=f"{self.threshold:.0%}")

    # ══════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════

    def _update_plc_status(self):
        if plc_bridge.connected:
            self.plc_status_label.configure(
                text=f"PLC  {plc_bridge.ams_net_id}:{plc_bridge.port}",
                text_color=OK,
            )
        else:
            self.plc_status_label.configure(
                text="PLC: Desconectado", text_color=TEXT_DIM,
            )

    def _update_snippet(self):
        ams = self.inp_ams.get().strip() or "5.80.201.232.1.1"
        port = self.inp_port.get().strip() or "851"
        var_c = self.inp_var_clase.get().strip() or "GVL.nResultadoClase"
        var_f = self.inp_var_conf.get().strip() or "GVL.rConfianza"

        snippet = (
            f"import pyads\n\n"
            f"plc = pyads.Connection('{ams}', {port})\n"
            f"plc.open()\n\n"
            f"plc.write_by_name('{var_c}', 1, pyads.PLCTYPE_INT)\n"
            f"plc.write_by_name('{var_f}', 0.95, pyads.PLCTYPE_REAL)\n\n"
            f"plc.close()"
        )

        self.snippet_textbox.configure(state="normal")
        self.snippet_textbox.delete("1.0", "end")
        self.snippet_textbox.insert("1.0", snippet)
        self.snippet_textbox.configure(state="disabled")

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        line = f"{ts}  {msg}\n"
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", line)
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")

    def _on_close(self):
        self.camera_running = False
        if self.cap is not None:
            self.cap.release()
        self.destroy()


def main():
    app = HMIApp()
    app.mainloop()


if __name__ == "__main__":
    main()
