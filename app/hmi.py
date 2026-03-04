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

# ── Paleta industrial ──
# Fondo base
BG_PRIMARY = "#18181b"     # zinc-900 — fondo ventana
BG_SURFACE = "#1e1e22"     # superficie de paneles
BG_CARD = "#252529"        # cards elevadas
BG_INPUT = "#2a2a2f"       # inputs / textboxes
BG_INSET = "#141417"       # áreas empotradas (snippet, log)
BORDER = "#333338"         # bordes sutiles

# Texto
TEXT_PRIMARY = "#e4e4e7"   # zinc-200
TEXT_SECONDARY = "#a1a1aa" # zinc-400
TEXT_MUTED = "#71717a"     # zinc-500
TEXT_DIM = "#52525b"       # zinc-600

# Acentos funcionales (desaturados, industriales)
OK_COLOR = "#22c55e"       # verde operativo — solo para "conectado / OK"
OK_DIM = "#166534"         # verde apagado para hover
WARN_COLOR = "#eab308"     # ámbar — precaución
FAIL_COLOR = "#ef4444"     # rojo — fallo / desconectado
ACCENT = "#3b82f6"         # azul — acción primaria (botón forzar)
ACCENT_HOVER = "#2563eb"   # azul hover

# Snippet
SNIPPET_TEXT = "#93c5fd"   # azul claro monospace


class HMIApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.configure(fg_color=BG_PRIMARY)

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
        self.topbar = ctk.CTkFrame(self, height=48, corner_radius=0,
                                   fg_color=BG_SURFACE, border_width=0)
        self.topbar.grid(row=0, column=0, sticky="ew")
        self.topbar.grid_columnconfigure(1, weight=1)

        # Acento izquierdo (línea fina de color)
        ctk.CTkFrame(self.topbar, width=3, fg_color=ACCENT,
                     corner_radius=0).grid(row=0, column=0, sticky="ns", rowspan=1)

        ctk.CTkLabel(
            self.topbar, text="  PUCP  —  Control de Calidad",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=TEXT_PRIMARY,
        ).grid(row=0, column=0, padx=(12, 10), pady=10, sticky="w")

        # PLC status
        self.plc_status_label = ctk.CTkLabel(
            self.topbar, text="  PLC: Desconectado",
            font=ctk.CTkFont(size=12), text_color=TEXT_MUTED,
        )
        self.plc_status_label.grid(row=0, column=1, padx=10, sticky="e")

        # Botón config
        self.btn_config_toggle = ctk.CTkButton(
            self.topbar, text="Configuracion", width=110, height=30,
            font=ctk.CTkFont(size=12),
            fg_color="transparent", border_width=1, border_color=BORDER,
            text_color=TEXT_SECONDARY, hover_color=BG_CARD,
            command=self._toggle_config,
        )
        self.btn_config_toggle.grid(row=0, column=2, padx=(5, 12), pady=9)

    def _build_main(self):
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(4, 4))
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=7)
        self.main_frame.grid_columnconfigure(1, weight=3)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # ── Panel cámara (izquierda) ──
        cam_frame = ctk.CTkFrame(self.main_frame, fg_color=BG_INSET,
                                 corner_radius=6, border_width=1,
                                 border_color=BORDER)
        cam_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        cam_frame.grid_rowconfigure(0, weight=1)
        cam_frame.grid_columnconfigure(0, weight=1)

        self.cam_label = ctk.CTkLabel(cam_frame, text="Iniciando camara...",
                                      font=ctk.CTkFont(size=13),
                                      text_color=TEXT_DIM)
        self.cam_label.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        self.fps_label = ctk.CTkLabel(cam_frame, text="",
                                      font=ctk.CTkFont(family="Courier", size=10),
                                      text_color=TEXT_DIM)
        self.fps_label.grid(row=0, column=0, sticky="ne", padx=10, pady=6)

        # ── Panel resultado (derecha) ──
        right_frame = ctk.CTkFrame(self.main_frame, fg_color=BG_SURFACE,
                                   corner_radius=6, border_width=1,
                                   border_color=BORDER)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        right_frame.grid_columnconfigure(0, weight=1)

        # Encabezado resultado
        ctk.CTkLabel(
            right_frame, text="RESULTADO",
            font=ctk.CTkFont(size=11), text_color=TEXT_DIM, anchor="w",
        ).grid(row=0, column=0, padx=16, pady=(14, 0), sticky="w")

        # Clase
        self.lbl_clase = ctk.CTkLabel(
            right_frame, text="—",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color=TEXT_PRIMARY,
        )
        self.lbl_clase.grid(row=1, column=0, padx=16, pady=(2, 6), sticky="ew")

        # Barra de confianza
        self.bar_track = ctk.CTkFrame(right_frame, height=16, fg_color=BG_INSET,
                                      corner_radius=4)
        self.bar_track.grid(row=2, column=0, padx=16, pady=(0, 2), sticky="ew")
        self.bar_track.grid_columnconfigure(0, weight=1)
        self.bar_track.grid_propagate(False)

        self.bar_fill = ctk.CTkFrame(self.bar_track, height=16, fg_color=TEXT_DIM,
                                     corner_radius=4)
        self.bar_fill.place(relx=0, rely=0, relwidth=0.0, relheight=1.0)

        # Porcentaje
        self.lbl_conf = ctk.CTkLabel(
            right_frame, text="—",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color=TEXT_MUTED,
        )
        self.lbl_conf.grid(row=3, column=0, padx=16, pady=(2, 4), sticky="ew")

        # Frames counter
        self.lbl_frames = ctk.CTkLabel(
            right_frame, text="Buffer: 0 / 10 frames",
            font=ctk.CTkFont(size=11), text_color=TEXT_DIM,
        )
        self.lbl_frames.grid(row=4, column=0, padx=16, pady=(0, 8), sticky="ew")

        # Línea divisoria
        ctk.CTkFrame(right_frame, height=1, fg_color=BORDER).grid(
            row=5, column=0, sticky="ew", padx=12, pady=4)

        # Botón FORZAR
        self.btn_forzar = ctk.CTkButton(
            right_frame, text="EJECUTAR INSPECCION",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=44, corner_radius=4,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            text_color="white",
            command=self._forzar,
        )
        self.btn_forzar.grid(row=6, column=0, padx=16, pady=(8, 8), sticky="ew")

        # Línea divisoria
        ctk.CTkFrame(right_frame, height=1, fg_color=BORDER).grid(
            row=7, column=0, sticky="ew", padx=12, pady=4)

        # Log PLC
        ctk.CTkLabel(
            right_frame, text="LOG",
            font=ctk.CTkFont(size=10), text_color=TEXT_DIM, anchor="w",
        ).grid(row=8, column=0, padx=16, pady=(4, 0), sticky="w")

        self.log_textbox = ctk.CTkTextbox(
            right_frame, height=110,
            font=ctk.CTkFont(family="Courier", size=10),
            fg_color=BG_INSET, text_color=TEXT_SECONDARY,
            corner_radius=4, border_width=1, border_color=BORDER,
            state="disabled",
        )
        self.log_textbox.grid(row=9, column=0, padx=16, pady=(2, 14), sticky="ew")

    def _build_config(self):
        self.config_frame = ctk.CTkFrame(self, fg_color=BG_SURFACE,
                                         corner_radius=6, border_width=1,
                                         border_color=BORDER)
        self.config_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        self.config_frame.grid_columnconfigure(0, weight=1)

        # ── Fila superior: dos cards lado a lado ──
        cards_row = ctk.CTkFrame(self.config_frame, fg_color="transparent")
        cards_row.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 4))
        cards_row.grid_columnconfigure((0, 1), weight=1)

        # --- Card MODELO ---
        card_modelo = ctk.CTkFrame(cards_row, fg_color=BG_CARD, corner_radius=4,
                                   border_width=1, border_color=BORDER)
        card_modelo.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        card_modelo.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(card_modelo, text="MODELO",
                     font=ctk.CTkFont(size=10),
                     text_color=TEXT_DIM).grid(
            row=0, column=0, columnspan=3, padx=12, pady=(10, 6), sticky="w")

        ctk.CTkLabel(card_modelo, text="Archivo",
                     font=ctk.CTkFont(size=11), text_color=TEXT_SECONDARY).grid(
            row=1, column=0, padx=(12, 6), pady=4, sticky="w")

        model_choices = list(self.modelos.keys()) if self.modelos else ["(sin modelos)"]
        self.dd_modelo = ctk.CTkOptionMenu(
            card_modelo, values=model_choices,
            font=ctk.CTkFont(size=11), width=220, height=28,
            fg_color=BG_INPUT, button_color=BORDER,
            button_hover_color=TEXT_DIM,
            dropdown_fg_color=BG_CARD, dropdown_hover_color=BG_INPUT,
            text_color=TEXT_PRIMARY,
        )
        self.dd_modelo.grid(row=1, column=1, sticky="ew", padx=(0, 6), pady=4)

        self.btn_cargar = ctk.CTkButton(
            card_modelo, text="Cargar", width=70, height=28,
            font=ctk.CTkFont(size=11),
            fg_color=BG_INPUT, hover_color=BORDER,
            text_color=TEXT_PRIMARY, border_width=1, border_color=BORDER,
            corner_radius=4,
            command=self._cargar_modelo,
        )
        self.btn_cargar.grid(row=1, column=2, padx=(0, 12), pady=4)

        self.lbl_model_info = ctk.CTkLabel(
            card_modelo, text="Ningun modelo cargado",
            font=ctk.CTkFont(size=10), text_color=TEXT_DIM, anchor="w",
        )
        self.lbl_model_info.grid(row=2, column=0, columnspan=3,
                                 padx=12, pady=(0, 10), sticky="w")

        # --- Card PLC ---
        card_plc = ctk.CTkFrame(cards_row, fg_color=BG_CARD, corner_radius=4,
                                border_width=1, border_color=BORDER)
        card_plc.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        card_plc.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(card_plc, text="CONEXION PLC",
                     font=ctk.CTkFont(size=10),
                     text_color=TEXT_DIM).grid(
            row=0, column=0, columnspan=4, padx=12, pady=(10, 6), sticky="w")

        ctk.CTkLabel(card_plc, text="AMS Net ID",
                     font=ctk.CTkFont(size=11), text_color=TEXT_SECONDARY).grid(
            row=1, column=0, padx=(12, 6), pady=4, sticky="w")
        self.inp_ams = ctk.CTkEntry(
            card_plc, font=ctk.CTkFont(size=11), height=28,
            fg_color=BG_INPUT, border_color=BORDER, text_color=TEXT_PRIMARY,
        )
        self.inp_ams.insert(0, PLC_DEFAULTS["ams_net_id"])
        self.inp_ams.grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=4)

        ctk.CTkLabel(card_plc, text="Puerto",
                     font=ctk.CTkFont(size=11), text_color=TEXT_SECONDARY).grid(
            row=1, column=2, padx=(0, 4), pady=4)
        self.inp_port = ctk.CTkEntry(
            card_plc, font=ctk.CTkFont(size=11), width=55, height=28,
            fg_color=BG_INPUT, border_color=BORDER, text_color=TEXT_PRIMARY,
        )
        self.inp_port.insert(0, str(PLC_DEFAULTS["port"]))
        self.inp_port.grid(row=1, column=3, padx=(0, 12), pady=4)

        # Botones conectar/desconectar
        btn_row = ctk.CTkFrame(card_plc, fg_color="transparent")
        btn_row.grid(row=2, column=0, columnspan=4, padx=12, pady=(2, 10), sticky="w")

        self.btn_conectar = ctk.CTkButton(
            btn_row, text="Conectar", width=80, height=28,
            font=ctk.CTkFont(size=11),
            fg_color=BG_INPUT, hover_color=OK_DIM,
            text_color=OK_COLOR, border_width=1, border_color=OK_DIM,
            corner_radius=4,
            command=self._conectar_plc,
        )
        self.btn_conectar.grid(row=0, column=0, padx=(0, 6))

        self.btn_desconectar = ctk.CTkButton(
            btn_row, text="Desconectar", width=90, height=28,
            font=ctk.CTkFont(size=11),
            fg_color="transparent", hover_color=BG_INPUT,
            text_color=TEXT_DIM, border_width=1, border_color=BORDER,
            corner_radius=4,
            command=self._desconectar_plc,
        )
        self.btn_desconectar.grid(row=0, column=1)

        # ── Fila inferior: Variables + Threshold + Snippet ──
        bottom_row = ctk.CTkFrame(self.config_frame, fg_color="transparent")
        bottom_row.grid(row=1, column=0, sticky="ew", padx=12, pady=(4, 4))
        bottom_row.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

        # Variables PLC (inline)
        ctk.CTkLabel(bottom_row, text="Var clase",
                     font=ctk.CTkFont(size=10), text_color=TEXT_DIM).grid(
            row=0, column=0, padx=(0, 4), sticky="w")
        self.inp_var_clase = ctk.CTkEntry(
            bottom_row, font=ctk.CTkFont(family="Courier", size=11),
            height=26, fg_color=BG_INPUT, border_color=BORDER,
            text_color=TEXT_PRIMARY,
        )
        self.inp_var_clase.insert(0, PLC_DEFAULTS["var_clase"])
        self.inp_var_clase.grid(row=0, column=1, sticky="ew", padx=(0, 12))

        ctk.CTkLabel(bottom_row, text="Var confianza",
                     font=ctk.CTkFont(size=10), text_color=TEXT_DIM).grid(
            row=0, column=2, padx=(0, 4), sticky="w")
        self.inp_var_conf = ctk.CTkEntry(
            bottom_row, font=ctk.CTkFont(family="Courier", size=11),
            height=26, fg_color=BG_INPUT, border_color=BORDER,
            text_color=TEXT_PRIMARY,
        )
        self.inp_var_conf.insert(0, PLC_DEFAULTS["var_confianza"])
        self.inp_var_conf.grid(row=0, column=3, sticky="ew", padx=(0, 12))

        # Threshold
        thresh_frame = ctk.CTkFrame(bottom_row, fg_color="transparent")
        thresh_frame.grid(row=0, column=4, sticky="ew")
        thresh_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(thresh_frame, text="Umbral",
                     font=ctk.CTkFont(size=10), text_color=TEXT_DIM).grid(
            row=0, column=0, padx=(0, 6))
        self.slider_threshold = ctk.CTkSlider(
            thresh_frame, from_=0.1, to=1.0, number_of_steps=18,
            height=14, fg_color=BG_INSET, progress_color=BORDER,
            button_color=TEXT_SECONDARY, button_hover_color=TEXT_PRIMARY,
            command=self._on_threshold_change,
        )
        self.slider_threshold.set(self.threshold)
        self.slider_threshold.grid(row=0, column=1, sticky="ew", padx=(0, 6))
        self.lbl_threshold = ctk.CTkLabel(
            thresh_frame, text=f"{self.threshold:.0%}",
            font=ctk.CTkFont(family="Courier", size=11),
            text_color=TEXT_SECONDARY, width=35,
        )
        self.lbl_threshold.grid(row=0, column=2)

        # ── Snippet de código PLC ──
        snippet_header = ctk.CTkFrame(self.config_frame, fg_color="transparent")
        snippet_header.grid(row=2, column=0, padx=12, pady=(8, 0), sticky="w")

        ctk.CTkLabel(
            snippet_header, text="CODIGO PLC (pyads)",
            font=ctk.CTkFont(size=10), text_color=TEXT_DIM, anchor="w",
        ).grid(row=0, column=0, sticky="w")

        self.snippet_textbox = ctk.CTkTextbox(
            self.config_frame, height=95,
            font=ctk.CTkFont(family="Courier", size=11),
            fg_color=BG_INSET, text_color=SNIPPET_TEXT,
            corner_radius=4, border_width=1, border_color=BORDER,
            state="disabled",
        )
        self.snippet_textbox.grid(row=3, column=0, padx=12, pady=(4, 12), sticky="ew")
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
                                     text_color=FAIL_COLOR)
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

        # Color funcional (no semáforo saturado)
        if conf >= self.threshold:
            bar_color = OK_COLOR
            text_color = OK_COLOR
        elif conf >= self.threshold * 0.7:
            bar_color = WARN_COLOR
            text_color = WARN_COLOR
        else:
            bar_color = FAIL_COLOR
            text_color = FAIL_COLOR

        self.bar_fill.configure(fg_color=bar_color)
        self.bar_fill.place(relx=0, rely=0, relwidth=max(0.01, conf),
                            relheight=1.0)
        self.lbl_conf.configure(text=f"{conf:.0%}", text_color=text_color)
        self.lbl_frames.configure(
            text=f"Buffer: {len(self.frame_buffer)} / {BUFFER_SIZE} frames")

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
            self._log("Buffer vacio — esperando frames")
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
            self._log("ERR  No se pudo clasificar ningun frame")
            return

        # Promedio
        mean_preds = np.mean(all_preds, axis=0)
        idx = int(np.argmax(mean_preds))
        conf = float(mean_preds[idx])
        nombre = self.class_names[idx] if idx < len(self.class_names) else f"Clase {idx}"

        self._log(f"OK   {nombre}  {conf:.0%}  [{len(all_preds)} frames]")

        # Actualizar UI
        bar_color = OK_COLOR if conf >= self.threshold else FAIL_COLOR
        text_color = bar_color
        self.lbl_clase.configure(text=nombre)
        self.bar_fill.configure(fg_color=bar_color)
        self.bar_fill.place(relx=0, rely=0, relwidth=max(0.01, conf),
                            relheight=1.0)
        self.lbl_conf.configure(text=f"{conf:.0%}", text_color=text_color)

        # Enviar al PLC
        var_clase = self.inp_var_clase.get().strip()
        var_conf = self.inp_var_conf.get().strip()

        if plc_bridge.connected:
            ok, msg = plc_bridge.enviar_resultado(idx, conf, var_clase, var_conf)
            self._log(f"{'OK' if ok else 'WARN'}  {msg}")
        else:
            self._log(f"---  PLC offline  clase={idx}  conf={conf:.2f}")
            self._log(f"     write_by_name('{var_clase}', {idx}, PLCTYPE_INT)")
            self._log(f"     write_by_name('{var_conf}', {conf:.4f}, PLCTYPE_REAL)")

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

            tipo = "MobileNetV2" if prep == "mobilenet" else "CNN custom"
            self.lbl_model_info.configure(
                text=f"{os.path.basename(info['path'])}  {tipo}  {sz}x{sz}  {len(names)} clases",
                text_color=TEXT_SECONDARY
            )
            self._log(f"OK   Modelo: {os.path.basename(info['path'])} ({tipo}, {sz}px)")
        except Exception as e:
            self._log(f"ERR  {e}")
            self.lbl_model_info.configure(text=f"Error: {e}", text_color=FAIL_COLOR)
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
            self.btn_config_toggle.configure(
                text="Ocultar", border_color=TEXT_DIM)
        else:
            self.config_frame.grid_remove()
            self.btn_config_toggle.configure(
                text="Configuracion", border_color=BORDER)

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
                text_color=OK_COLOR,
            )
        else:
            self.plc_status_label.configure(
                text="PLC: Desconectado",
                text_color=TEXT_DIM,
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

    def _log(self, msg):
        """Agrega línea al log PLC visible."""
        ts = time.strftime("%H:%M:%S")
        line = f"{ts}  {msg}\n"
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
