# Laboratorio de Inteligencia Artificial — PUCP

Laboratorio de clasificación de imágenes con Redes Neuronales Convolucionales (CNN).

## Estructura del proyecto

```
pucp/
├── app/                                # Aplicación web Gradio (2 tabs)
│   ├── main.py                         # Entry point: python -m app.main
│   ├── config.py                       # Constantes y estado compartido
│   ├── datos.py                        # Utilidades de dataset
│   ├── modelo.py                       # MobileNetV2 + CNN custom
│   ├── plc.py                          # Cliente OPC UA (para scripts)
│   └── ui/                             # Tabs de la interfaz
│       ├── tab_datos.py                # Tab 1: Crear/cargar dataset
│       └── tab_probar.py               # Tab 2: Probar predicciones
│
├── labs/                                # Notebooks de laboratorio (guiados)
│   ├── lab1_cnn_basica.ipynb            # Lab 1: CNN básica (botellas)
│   ├── lab2_data_augmentation.ipynb     # Lab 2: Data augmentation
│   └── lab3_clasificacion_tapitas.ipynb # Lab 3: Clasificación de tapitas
│
├── proyecto/                            # Notebooks para el proyecto del alumno
│   ├── entrenar.ipynb                   # Entrenar CNN/MobileNetV2 (código visible)
│   └── probador.ipynb                   # Teachable Machine interactivo
│
├── scripts/                             # Scripts de soporte
│   ├── capturar_clases.py               # Captura imágenes por clase desde cámara
│   ├── dividir_video.py                 # Extrae frames de video para dataset
│   ├── prueba_video.py                  # Prueba el modelo con cámara en vivo
│   ├── inferencia_plc.py                # Plantilla: inferencia + comunicación PLC
│   └── simular_plc.py                   # Simulador OPC UA para pruebas
│
├── data/                                # Datasets
│   ├── botellas/                        # Dataset pre-cargado (labs 1-2)
│   │   ├── glass/
│   │   └── plastic/
│   └── tapitas/                         # Dataset del alumno (lab 3)
│       └── (creado con dividir_video.py)
│
└── docs/                                # Documentación
    └── guia_2025-2_IA.pdf
```

## Instalación rápida

### Opción A: Con `uv` (recomendado)

```bash
# Instalar uv (si no lo tiene)
# Windows:
irm https://astral.sh/uv/install.ps1 | iex
# Linux/Mac:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Configurar entorno
uv sync
```

### Opción B: Con pip

```bash
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# .\.venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

## Labs (guiados por el profesor)

Notebooks con instrucciones paso a paso. Abrir en Jupyter y seguir las celdas:

1. `labs/lab1_cnn_basica.ipynb` — CNN básica con botellas glass/plastic
2. `labs/lab2_data_augmentation.ipynb` — Técnicas de Data Augmentation
3. `labs/lab3_clasificacion_tapitas.ipynb` — Proyecto de tapitas

```bash
uv run jupyter notebook labs/
```

## Proyecto del alumno

Flujo completo: capturar dataset, entrenar modelo, probar, integrar con PLC.

### 1. Capturar dataset

```bash
# Opción A: App web (Tab 1 — Datos)
uv run python -m app.main
# Se abre en http://localhost:7860

# Opción B: Script CLI
python scripts/capturar_clases.py rojo azul verde --tiempo 15
```

### 2. Entrenar modelo

Abra `proyecto/entrenar.ipynb` en Jupyter y ejecute celda por celda.
El alumno **ve y modifica** todo el código: data augmentation, arquitectura
(CNN custom o MobileNetV2 con transfer learning), hiperparámetros, evaluación.

```bash
uv run jupyter notebook proyecto/entrenar.ipynb
```

### 3. Probar modelo

```bash
# Opción A: App web (Tab 2 — Probar)
uv run python -m app.main

# Opción B: Cámara en vivo
python scripts/prueba_video.py  # completar configuración al inicio
```

### 4. Integración PLC

Edite `scripts/inferencia_plc.py` — el alumno escribe su propia lógica de
comunicación con el PLC (los TODOs indican dónde). El código de inferencia
(cargar modelo, predecir) ya está implementado.

```bash
# Probar con simulador:
uv run python scripts/simular_plc.py    # terminal 1
python scripts/inferencia_plc.py         # terminal 2
```

## Uso de Jupyter

```bash
# Con uv:
uv run jupyter notebook

# Con pip (entorno activado):
jupyter notebook
```

## Scripts CLI

```bash
# Extraer frames de video
python scripts/dividir_video.py amarillo --video ruta/video.mp4 --salida data/tapitas/amarillo

# Capturar imágenes con cámara
python scripts/capturar_clases.py rojo azul verde --tiempo 15

# Prueba en vivo con modelo
python scripts/prueba_video.py  # completar configuración al inicio

# Inferencia + PLC (plantilla)
python scripts/inferencia_plc.py  # completar TODOs
```
