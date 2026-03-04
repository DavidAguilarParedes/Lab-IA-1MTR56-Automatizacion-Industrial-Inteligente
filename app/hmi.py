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

# ── Paleta ──
# Más contraste entre capas, no tan oscuro en el fondo base
BG = "#1c1c21"             # fondo ventana
BG_SURFACE = "#242429"     # paneles principales
BG_RAISED = "#2e2e35"      # cards elevadas, inputs
BG_INSET = "#131316"       # áreas hundidas (cámara, snippet, log)
BORDER_SUBTLE = "#38383f"  # bordes normales
BORDER_ACCENT = "#4a4a54"  # bordes con más presencia

# Texto — mayor rango de contraste
TEXT_HI = "#f0f0f3"        # títulos, clase detectada
TEXT = "#c8c8ce"           # texto normal
TEXT_LO = "#8a8a94"        # labels secundarios
TEXT_DIM = "#5c5c66"       # hints, placeholders

# Funcional
OK = "#34d399"             # emerald-400 — conectado, inspección OK
OK_SOFT = "#065f46"        # emerald-900 — hover
WARN = "#fbbf24"           # amber-400
FAIL = "#f87171"           # red-400
ACCENT = "#60a5fa"         # blue-400 — botón principal
ACCENT_HOVER = "#3b82f6"   # blue-500 — hover
LIVE_DOT = "#f87171"       # rojo para indicador live


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

        # Ventana
        self.title("PUCP — Control de Calidad")
        self.geometry("1200x720")
        self.minsize(900, 550)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Config panel visible
        self.config_visible = False

        # Live dot animation state
        self._live_dot_on = True

        # Construir UI
        self._build_topbar()
        self._build_main()
        self._build_config()

        # Ocultar config por defecto
        self.config_frame.grid_remove()

        # Iniciar cámara
        self._start_camera()

        # Iniciar live dot pulse
        self._pulse_live_dot()

    # ══════════════════════════════════════
    #  LAYOUT
    # ══════════════════════════════════════

    def _build_topbar(self):
        """Barra superior compacta: título + live dot + PLC status + config."""
        self.topbar = ctk.CTkFrame(self, height=36, corner_radius=0,
                                   fg_color=BG_SURFACE)
        self.topbar.grid(row=0, column=0, sticky="ew")
        self.topbar.grid_columnconfigure(2, weight=1)
        self.topbar.grid_propagate(False)

        # Línea de acento superior (2px)
        accent_line = ctk.CTkFrame(self, height=2, corner_radius=0,
                                   fg_color=ACCENT)
        accent_line.grid(row=0, column=0, sticky="new")
        accent_line.lift()

        ctk.CTkLabel(
            self.topbar, text="PUCP — Control de Calidad",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT,
        ).grid(row=0, column=0, padx=(14, 0), pady=0, sticky="w")

        # Live dot (pulsa cuando cámara activa)
        self.live_dot = ctk.CTkLabel(
            self.topbar, text="●", font=ctk.CTkFont(size=9),
            text_color=LIVE_DOT, width=16,
        )
        self.live_dot.grid(row=0, column=1, padx=(6, 0))

        # PLC status (derecha)
        self.plc_status_label = ctk.CTkLabel(
            self.topbar, text="PLC: Desconectado",
            font=ctk.CTkFont(size=11), text_color=TEXT_DIM,
        )
        self.plc_status_label.grid(row=0, column=2, padx=10, sticky="e")

        # Botón config
        self.btn_config_toggle = ctk.CTkButton(
            self.topbar, text="Configuracion", width=100, height=24,
            font=ctk.CTkFont(size=11),
            fg_color="transparent", border_width=1, border_color=BORDER_SUBTLE,
            text_color=TEXT_LO, hover_color=BG_RAISED,
            corner_radius=3,
            command=self._toggle_config,
        )
        self.btn_config_toggle.grid(row=0, column=3, padx=(0, 10), pady=6)

    def _build_main(self):
        """Área principal: cámara (izq) + resultado (der)."""
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=1, column=0, sticky="nsew", padx=6, pady=(3, 3))
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=7)
        self.main_frame.grid_columnconfigure(1, weight=3)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # ── Panel cámara (izquierda) ──
        cam_frame = ctk.CTkFrame(self.main_frame, fg_color=BG_INSET,
                                 corner_radius=4, border_width=1,
                                 border_color=BORDER_SUBTLE)
        cam_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
        cam_frame.grid_rowconfigure(0, weight=1)
        cam_frame.grid_columnconfigure(0, weight=1)

        self.cam_label = ctk.CTkLabel(cam_frame, text="Iniciando camara...",
                                      font=ctk.CTkFont(size=12),
                                      text_color=TEXT_DIM)
        self.cam_label.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        self.fps_label = ctk.CTkLabel(cam_frame, text="",
                                      font=ctk.CTkFont(family="Courier", size=9),
                                      text_color=TEXT_DIM,
                                      fg_color=BG_INSET)
        self.fps_label.grid(row=0, column=0, sticky="se", padx=8, pady=6)

        # ── Panel resultado (derecha) ──
        self.right_frame = ctk.CTkFrame(self.main_frame, fg_color=BG_SURFACE,
                                        corner_radius=4, border_width=1,
                                        border_color=BORDER_SUBTLE)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=(3, 0))
        self.right_frame.grid_columnconfigure(0, weight=1)
        # El log (row 8) se expande para llenar el espacio sobrante
        self.right_frame.grid_rowconfigure(8, weight=1)

        # --- Sección resultado ---
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

        # Barra de confianza (custom frame)
        self.bar_track = ctk.CTkFrame(self.right_frame, height=14,
                                      fg_color=BG_INSET, corner_radius=3)
        self.bar_track.grid(row=2, column=0, padx=14, pady=(0, 2), sticky="ew")
        self.bar_track.grid_propagate(False)

        self.bar_fill = ctk.CTkFrame(self.bar_track, height=14,
                                     fg_color=TEXT_DIM, corner_radius=3)
        self.bar_fill.place(relx=0, rely=0, relwidth=0.0, relheight=1.0)

        # Porcentaje + frames en una fila
        stats_row = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        stats_row.grid(row=3, column=0, padx=14, pady=(2, 0), sticky="ew")
        stats_row.grid_columnconfigure(0, weight=1)

        self.lbl_conf = ctk.CTkLabel(
            stats_row, text="—",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=TEXT_DIM, anchor="w",
        )
        self.lbl_conf.grid(row=0, column=0, sticky="w")

        self.lbl_frames = ctk.CTkLabel(
            stats_row, text="0/10",
            font=ctk.CTkFont(family="Courier", size=10),
            text_color=TEXT_DIM, anchor="e",
        )
        self.lbl_frames.grid(row=0, column=1, sticky="e")

        # --- Botón EJECUTAR ---
        ctk.CTkFrame(self.right_frame, height=1, fg_color=BORDER_SUBTLE).grid(
            row=4, column=0, sticky="ew", padx=10, pady=(10, 0))

        self.btn_forzar = ctk.CTkButton(
            self.right_frame, text="EJECUTAR INSPECCION",
            font=ctk.CTkFont(size=12, weight="bold"),
            height=40, corner_radius=4,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color="white",
            command=self._forzar,
        )
        self.btn_forzar.grid(row=5, column=0, padx=14, pady=(10, 0), sticky="ew")

        # --- Modelo cargado (inline) ---
        self.lbl_model_status = ctk.CTkLabel(
            self.right_frame, text="Sin modelo",
            font=ctk.CTkFont(size=10), text_color=TEXT_DIM, anchor="w",
        )
        self.lbl_model_status.grid(row=6, column=0, padx=14, pady=(6, 0), sticky="w")

        # --- Log ---
        ctk.CTkFrame(self.right_frame, height=1, fg_color=BORDER_SUBTLE).grid(
            row=7, column=0, sticky="ew", padx=10, pady=(8, 0))

        self.log_textbox = ctk.CTkTextbox(
            self.right_frame,
            font=ctk.CTkFont(family="Courier", size=10),
            fg_color=BG_INSET, text_color=TEXT_LO,
            corner_radius=3, border_width=0,
            state="disabled", wrap="none",
        )
        self.log_textbox.grid(row=8, column=0, padx=8, pady=(4, 8), sticky="nsew")

    def _build_config(self):
        """Panel de configuración colapsable."""
        self.config_frame = ctk.CTkFrame(self, fg_color=BG_SURFACE,
                                         corner_radius=4, border_width=1,
                                         border_color=BORDER_SUBTLE)
        self.config_frame.grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 6))
        self.config_frame.grid_columnconfigure(0, weight=1)

        # ── Cards: Modelo + PLC lado a lado ──
        cards = ctk.CTkFrame(self.config_frame, fg_color="transparent")
        cards.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        cards.grid_columnconfigure((0, 1), weight=1)

        # --- MODELO ---
        cm = ctk.CTkFrame(cards, fg_color=BG_RAISED, corner_radius=4,
                           border_width=1, border_color=BORDER_SUBTLE)
        cm.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
        cm.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(cm, text="MODELO", font=ctk.CTkFont(size=9, weight="bold"),
                     text_color=TEXT_DIM).grid(
            row=0, column=0, columnspan=3, padx=10, pady=(8, 4), sticky="w")

        model_choices = list(self.modelos.keys()) if self.modelos else ["(sin modelos)"]
        self.dd_modelo = ctk.CTkOptionMenu(
            cm, values=model_choices,
            font=ctk.CTkFont(size=11), height=26,
            fg_color=BG_INSET, button_color=BORDER_SUBTLE,
            button_hover_color=BORDER_ACCENT,
            dropdown_fg_color=BG_RAISED, dropdown_hover_color=BG_SURFACE,
            text_color=TEXT,
        )
        self.dd_modelo.grid(row=1, column=0, columnspan=2, sticky="ew",
                            padx=(10, 4), pady=(0, 8))

        self.btn_cargar = ctk.CTkButton(
            cm, text="Cargar", width=60, height=26,
            font=ctk.CTkFont(size=11),
            fg_color=BG_INSET, hover_color=BORDER_SUBTLE,
            text_color=TEXT, corner_radius=3,
            command=self._cargar_modelo,
        )
        self.btn_cargar.grid(row=1, column=2, padx=(0, 10), pady=(0, 8))

        # --- CONEXION PLC ---
        cp = ctk.CTkFrame(cards, fg_color=BG_RAISED, corner_radius=4,
                           border_width=1, border_color=BORDER_SUBTLE)
        cp.grid(row=0, column=1, sticky="nsew", padx=(3, 0))
        cp.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(cp, text="CONEXION PLC", font=ctk.CTkFont(size=9, weight="bold"),
                     text_color=TEXT_DIM).grid(
            row=0, column=0, columnspan=4, padx=10, pady=(8, 4), sticky="w")

        # AMS + Puerto en fila
        row_plc = ctk.CTkFrame(cp, fg_color="transparent")
        row_plc.grid(row=1, column=0, columnspan=4, sticky="ew",
                     padx=10, pady=(0, 4))
        row_plc.grid_columnconfigure(0, weight=3)
        row_plc.grid_columnconfigure(1, weight=1)

        self.inp_ams = ctk.CTkEntry(
            row_plc, font=ctk.CTkFont(size=11), height=26,
            fg_color=BG_INSET, border_color=BORDER_SUBTLE, text_color=TEXT,
            placeholder_text="AMS Net ID",
            placeholder_text_color=TEXT_DIM,
        )
        self.inp_ams.insert(0, PLC_DEFAULTS["ams_net_id"])
        self.inp_ams.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        self.inp_port = ctk.CTkEntry(
            row_plc, font=ctk.CTkFont(size=11), width=55, height=26,
            fg_color=BG_INSET, border_color=BORDER_SUBTLE, text_color=TEXT,
            placeholder_text="Puerto",
            placeholder_text_color=TEXT_DIM,
        )
        self.inp_port.insert(0, str(PLC_DEFAULTS["port"]))
        self.inp_port.grid(row=0, column=1, sticky="ew")

        # Botones
        btn_plc = ctk.CTkFrame(cp, fg_color="transparent")
        btn_plc.grid(row=2, column=0, columnspan=4, padx=10, pady=(0, 8), sticky="w")

        self.btn_conectar = ctk.CTkButton(
            btn_plc, text="Conectar", width=72, height=26,
            font=ctk.CTkFont(size=10),
            fg_color=BG_INSET, hover_color=OK_SOFT,
            text_color=OK, corner_radius=3,
            command=self._conectar_plc,
        )
        self.btn_conectar.grid(row=0, column=0, padx=(0, 4))

        self.btn_desconectar = ctk.CTkButton(
            btn_plc, text="Desconectar", width=82, height=26,
            font=ctk.CTkFont(size=10),
            fg_color="transparent", hover_color=BG_INSET,
            text_color=TEXT_DIM, corner_radius=3,
            command=self._desconectar_plc,
        )
        self.btn_desconectar.grid(row=0, column=1)

        # ── Variables PLC + Threshold (fila compacta) ──
        vars_row = ctk.CTkFrame(self.config_frame, fg_color="transparent")
        vars_row.grid(row=1, column=0, sticky="ew", padx=10, pady=(2, 2))
        vars_row.grid_columnconfigure((1, 3, 5), weight=1)

        ctk.CTkLabel(vars_row, text="Var clase",
                     font=ctk.CTkFont(size=9), text_color=TEXT_DIM).grid(
            row=0, column=0, padx=(0, 4), sticky="w")
        self.inp_var_clase = ctk.CTkEntry(
            vars_row, font=ctk.CTkFont(family="Courier", size=10),
            height=24, fg_color=BG_INSET, border_color=BORDER_SUBTLE,
            text_color=TEXT,
        )
        self.inp_var_clase.insert(0, PLC_DEFAULTS["var_clase"])
        self.inp_var_clase.grid(row=0, column=1, sticky="ew", padx=(0, 10))

        ctk.CTkLabel(vars_row, text="Var confianza",
                     font=ctk.CTkFont(size=9), text_color=TEXT_DIM).grid(
            row=0, column=2, padx=(0, 4), sticky="w")
        self.inp_var_conf = ctk.CTkEntry(
            vars_row, font=ctk.CTkFont(family="Courier", size=10),
            height=24, fg_color=BG_INSET, border_color=BORDER_SUBTLE,
            text_color=TEXT,
        )
        self.inp_var_conf.insert(0, PLC_DEFAULTS["var_confianza"])
        self.inp_var_conf.grid(row=0, column=3, sticky="ew", padx=(0, 10))

        ctk.CTkLabel(vars_row, text="Umbral",
                     font=ctk.CTkFont(size=9), text_color=TEXT_DIM).grid(
            row=0, column=4, padx=(0, 4))
        self.slider_threshold = ctk.CTkSlider(
            vars_row, from_=0.1, to=1.0, number_of_steps=18,
            height=12, fg_color=BG_INSET, progress_color=BORDER_ACCENT,
            button_color=TEXT_LO, button_hover_color=TEXT,
            command=self._on_threshold_change,
        )
        self.slider_threshold.set(self.threshold)
        self.slider_threshold.grid(row=0, column=5, sticky="ew", padx=(0, 4))
        self.lbl_threshold = ctk.CTkLabel(
            vars_row, text=f"{self.threshold:.0%}",
            font=ctk.CTkFont(family="Courier", size=10),
            text_color=TEXT_LO, width=30,
        )
        self.lbl_threshold.grid(row=0, column=6)

        # ── Snippet de código PLC ──
        ctk.CTkLabel(
            self.config_frame, text="CODIGO PLC (pyads)",
            font=ctk.CTkFont(size=9, weight="bold"), text_color=TEXT_DIM,
            anchor="w",
        ).grid(row=2, column=0, padx=10, pady=(6, 0), sticky="w")

        self.snippet_textbox = ctk.CTkTextbox(
            self.config_frame, height=85,
            font=ctk.CTkFont(family="Courier", size=11),
            fg_color=BG_INSET, text_color="#93c5fd",
            corner_radius=3, border_width=1, border_color=BORDER_SUBTLE,
            state="disabled",
        )
        self.snippet_textbox.grid(row=3, column=0, padx=10, pady=(2, 10),
                                  sticky="ew")
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

            time.sleep(0.01)

    def _update_camera(self):
        """Actualiza el canvas de cámara desde el thread principal."""
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
        """Actualiza labels de clase/confianza con suavizado temporal."""
        if len(self.pred_buffer) == 0:
            return

        smooth = np.mean(list(self.pred_buffer), axis=0)
        idx = int(np.argmax(smooth))
        conf = float(smooth[idx])

        nombre = self.class_names[idx] if idx < len(self.class_names) else f"Clase {idx}"
        self.lbl_clase.configure(text=nombre)

        if conf >= self.threshold:
            bar_color, text_color = OK, OK
        elif conf >= self.threshold * 0.7:
            bar_color, text_color = WARN, WARN
        else:
            bar_color, text_color = FAIL, FAIL

        self.bar_fill.configure(fg_color=bar_color)
        self.bar_fill.place(relx=0, rely=0, relwidth=max(0.01, conf),
                            relheight=1.0)
        self.lbl_conf.configure(text=f"{conf:.0%}", text_color=text_color)
        self.lbl_frames.configure(
            text=f"{len(self.frame_buffer)}/{BUFFER_SIZE}")

    def _pulse_live_dot(self):
        """Pulsa el punto rojo de 'live' en la topbar."""
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
        """Inspección multi-frame: promedia buffer -> resultado -> envía PLC."""
        if self.model is None:
            self._log("Sin modelo cargado")
            return

        frames = list(self.frame_buffer)
        if not frames:
            self._log("Buffer vacio")
            return

        # Flash visual en el botón
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

        # Restaurar botón
        self.btn_forzar.configure(fg_color=ACCENT, text="EJECUTAR INSPECCION")

        if not all_preds:
            self._log("ERR  Sin predicciones")
            return

        mean_preds = np.mean(all_preds, axis=0)
        idx = int(np.argmax(mean_preds))
        conf = float(mean_preds[idx])
        nombre = self.class_names[idx] if idx < len(self.class_names) else f"Clase {idx}"

        self._log(f"OK   {nombre}  {conf:.0%}  [{len(all_preds)}f]")

        # Actualizar resultado con flash de color
        bar_color = OK if conf >= self.threshold else FAIL
        self.lbl_clase.configure(text=nombre, text_color="white")
        self.bar_fill.configure(fg_color=bar_color)
        self.bar_fill.place(relx=0, rely=0, relwidth=max(0.01, conf),
                            relheight=1.0)
        self.lbl_conf.configure(text=f"{conf:.0%}", text_color=bar_color)

        # Flash: briefly highlight the result section background
        self.right_frame.configure(border_color=bar_color)
        self.after(800, lambda: self.right_frame.configure(
            border_color=BORDER_SUBTLE))
        # Fade class text back to normal
        self.after(500, lambda: self.lbl_clase.configure(text_color=TEXT_HI))

        # Enviar al PLC
        var_clase = self.inp_var_clase.get().strip()
        var_conf = self.inp_var_conf.get().strip()

        if plc_bridge.connected:
            ok, msg = plc_bridge.enviar_resultado(idx, conf, var_clase, var_conf)
            self._log(f"{'OK' if ok else 'WARN'}  {msg}")
        else:
            self._log(f"---  PLC offline  clase={idx}  conf={conf:.2f}")
            self._log(f"     write('{var_clase}', {idx})")
            self._log(f"     write('{var_conf}', {conf:.4f})")

    def _cargar_modelo(self):
        label = self.dd_modelo.get()
        if label not in self.modelos:
            self._log("ERR  Modelo no encontrado")
            return

        info = self.modelos[label]
        self.btn_cargar.configure(state="disabled", text="...")
        self.update()

        try:
            names, sz, prep = _load_model_hmi(info["path"])
            from app.ui.tab_hmi import hmi_state
            self.model = hmi_state["model"]
            self.class_names = names
            self.img_size = sz
            self.preprocessing = prep
            self.pred_buffer.clear()

            tipo = "MobileNet" if prep == "mobilenet" else "CNN"
            fname = os.path.basename(info["path"])
            self.lbl_model_status.configure(
                text=f"{fname}  |  {tipo} {sz}px  |  {len(names)} clases",
                text_color=TEXT_LO,
            )
            self._log(f"OK   {fname} ({tipo}, {sz}px, {len(names)} clases)")
        except Exception as e:
            self._log(f"ERR  {e}")
            self.lbl_model_status.configure(text=f"Error: {e}",
                                            text_color=FAIL)
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
                                             border_color=BORDER_ACCENT)
        else:
            self.config_frame.grid_remove()
            self.btn_config_toggle.configure(text="Configuracion",
                                             border_color=BORDER_SUBTLE)

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
        """Actualiza el snippet de código pyads dinámico."""
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
        """Agrega línea al log."""
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
