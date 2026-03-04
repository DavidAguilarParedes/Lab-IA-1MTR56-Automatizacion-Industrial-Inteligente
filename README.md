# Laboratorio de Inteligencia Artificial — PUCP

Clasificacion de imagenes con Redes Neuronales Convolucionales (CNN).
Incluye labs guiados, proyecto del alumno con entrenamiento propio, y un HMI industrial para integracion con PLC Beckhoff.

---

## Requisitos

- **Python 3.10** (no usar 3.12 ni 3.13, TensorFlow no es compatible)
- **Windows 10/11** con PowerShell (tambien funciona en Linux/Mac)
- Camara web (para captura de datos y HMI en vivo)

> Si no tiene Python 3.10, descargarlo de https://www.python.org/downloads/release/python-31011/
> Al instalar, **marcar la casilla "Add Python to PATH"**.

---

## Instalacion paso a paso (Windows PowerShell)

### 1. Descargar el proyecto

```powershell
git clone https://github.com/DavidAguilarParedes/Lab-IA-1MTR56-Automatizacion-Industrial-Inteligente.git
cd Lab-IA-1MTR56-Automatizacion-Industrial-Inteligente
```

Si no tiene `git`, puede descargar el ZIP desde GitHub y descomprimirlo.

### 2. Crear entorno virtual e instalar dependencias

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **Nota:** Si PowerShell no permite ejecutar scripts, ejecute primero:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

> **Nota:** La instalacion de TensorFlow puede tardar varios minutos. Es normal.

### 3. Verificar que todo funciona

```powershell
python -c "import tensorflow; print('TensorFlow OK:', tensorflow.__version__)"
python -c "import cv2; print('OpenCV OK:', cv2.__version__)"
```

---

## Como abrir cada cosa

**Importante:** Siempre activar el entorno virtual antes de ejecutar cualquier comando:

```powershell
cd Lab-IA-1MTR56-Automatizacion-Industrial-Inteligente
.\.venv\Scripts\Activate.ps1
```

Cuando el entorno esta activo, vera `(.venv)` al inicio de la linea de comandos.

### Aplicacion web (Gradio) — desarrollo y pruebas

```powershell
python -m app.main
```

Se abre en el navegador en http://localhost:7860. Tiene 3 pestanas:
1. **Datos** — Crear dataset desde camara
2. **Probar** — Probar modelo con imagen o camara
3. **HMI** — Version web del HMI (para desarrollo)

### HMI Industrial (CustomTkinter) — produccion

```powershell
python -m app.hmi
```

Abre una ventana de escritorio con:
- Camara en vivo con prediccion en tiempo real
- Panel de resultado con clase y confianza
- Selector de modelo (dropdown + buscar archivo)
- Conexion a PLC Beckhoff via pyads
- Snippet de codigo PLC que se actualiza en vivo

### Jupyter Notebooks — labs y proyecto

```powershell
jupyter notebook
```

Se abre en el navegador. Navegar a la carpeta deseada (`labs/` o `proyecto/`).

---

## Estructura del proyecto

```
pucp/
├── app/                          # Aplicacion
│   ├── main.py                   # App web Gradio: python -m app.main
│   ├── hmi.py                    # HMI standalone: python -m app.hmi
│   ├── config.py                 # Constantes y configuracion
│   ├── datos.py                  # Utilidades de dataset
│   ├── modelo.py                 # Arquitecturas CNN y MobileNetV2
│   ├── plc.py                    # Comunicacion PLC Beckhoff (pyads)
│   └── ui/                       # Pestanas de la interfaz web
│       ├── tab_datos.py          # Tab 1: Crear/cargar dataset
│       ├── tab_probar.py         # Tab 2: Probar predicciones
│       ├── tab_hmi.py            # Tab 3: HMI en Gradio
│       └── pred_html.py          # Renderizado HTML de predicciones
│
├── labs/                          # Notebooks guiados por el profesor
│   ├── lab1_cnn_basica.ipynb      # Lab 1: CNN basica (botellas)
│   ├── lab2_data_augmentation.ipynb  # Lab 2: Data Augmentation
│   └── lab3_clasificacion_tapitas.ipynb  # Lab 3: Tapitas
│
├── proyecto/                      # Proyecto del alumno
│   ├── entrenar.ipynb             # Entrenar su propio modelo
│   ├── probador.ipynb             # Probar modelo interactivamente
│   └── modelos/                   # Modelos entrenados (ejemplo incluido)
│
├── scripts/                       # Scripts auxiliares
│   ├── capturar_clases.py         # Capturar imagenes por clase
│   ├── dividir_video.py           # Extraer frames de un video
│   ├── prueba_video.py            # Probar modelo con camara en vivo
│   ├── inferencia_plc.py          # Plantilla: inferencia + PLC
│   └── simular_plc.py             # Simulador PLC para pruebas
│
├── data/                          # Datasets
│   ├── botellas/                  # Dataset ejemplo (glass/plastic)
│   └── tapitas/                   # Datos del alumno
│
├── docs/                          # Documentacion del curso
│   └── guia_2025-2_IA.pdf
│
├── requirements.txt               # Dependencias (pip install -r requirements.txt)
└── pyproject.toml                 # Configuracion del proyecto
```

---

## Flujo de trabajo del proyecto

### Paso 1: Capturar dataset

Usar la app web (Tab 1 — Datos) o el script:

```powershell
python scripts/capturar_clases.py rojo azul verde --tiempo 15
```

### Paso 2: Entrenar modelo

Abrir `proyecto/entrenar.ipynb` en Jupyter y ejecutar celda por celda.
El alumno ve y modifica: data augmentation, arquitectura, hiperparametros.

```powershell
jupyter notebook proyecto/entrenar.ipynb
```

### Paso 3: Probar modelo

```powershell
# Opcion A: App web (Tab 2)
python -m app.main

# Opcion B: Camara en vivo (script)
python scripts/prueba_video.py
```

### Paso 4: HMI + PLC

```powershell
python -m app.hmi
```

1. Seleccionar modelo en el panel derecho (dropdown o "Buscar...")
2. Hacer clic en "Cargar"
3. La camara muestra prediccion en tiempo real
4. Abrir "Config PLC" para conectar al PLC Beckhoff
5. "EJECUTAR INSPECCION" clasifica multiples frames y envia resultado al PLC

---

## Solucion de problemas comunes

### "python no se reconoce como comando"
Python no esta en el PATH. Reinstalar Python y marcar "Add Python to PATH".

### "No module named tensorflow"
El entorno virtual no esta activado. Ejecutar:
```powershell
.\.venv\Scripts\Activate.ps1
```

### "cannot be loaded because running scripts is disabled"
Ejecutar en PowerShell:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### "No se pudo abrir la camara"
- Verificar que la camara no este siendo usada por otra aplicacion
- En laptops con doble camara, cambiar `CAMERA_INDEX = 0` por `1` en `app/hmi.py`

### La instalacion de TensorFlow falla
- Verificar que tiene Python 3.10 (no 3.12 ni 3.13)
- Verificar con: `python --version`

### "pip install" tarda mucho
Es normal. TensorFlow es un paquete grande (~500 MB). Esperar a que termine.

---

## Instalacion alternativa con uv (avanzado)

Si prefiere usar `uv` en vez de pip:

```powershell
# Instalar uv
irm https://astral.sh/uv/install.ps1 | iex

# Instalar todo automaticamente
uv sync

# Ejecutar comandos con uv
uv run python -m app.main
uv run python -m app.hmi
uv run jupyter notebook
```
