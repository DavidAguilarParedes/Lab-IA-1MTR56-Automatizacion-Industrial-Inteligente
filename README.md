# Laboratorio de Inteligencia Artificial — PUCP

Laboratorio de clasificación de imágenes con Redes Neuronales Convolucionales (CNN).

## Estructura del proyecto

```
pucp/
├── labs/                                # Notebooks de laboratorio
│   ├── lab1_cnn_basica.ipynb            # CNN básica (botellas glass/plastic)
│   ├── lab2_data_augmentation.ipynb     # Data augmentation
│   ├── lab3_clasificacion_tapitas.ipynb # Proyecto: clasificación de tapitas
│   └── probador.ipynb                   # Teachable Machine: entrenar/probar modelos
│
├── scripts/                             # Scripts de soporte
│   ├── dividir_video.py                 # Extrae frames de video para dataset
│   └── prueba_video.py                  # Prueba el modelo con cámara en vivo
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

## Flujo de trabajo

### Labs 1 y 2 (enseñanza)
1. Abra `labs/lab1_cnn_basica.ipynb` en Jupyter
2. Ejecute las celdas para aprender CNN básica
3. Luego abra `labs/lab2_data_augmentation.ipynb` para aprender Data Augmentation

### Lab 3 (proyecto de tapitas)
1. Grabe videos de tapitas en la **zona de inspección** con el celular fijo
2. Ejecute `scripts/dividir_video.py` para extraer frames al directorio `data/tapitas/`
3. Abra `labs/lab3_clasificacion_tapitas.ipynb` y siga las instrucciones
4. Exporte el modelo entrenado (.h5)
5. Abra `labs/probador.ipynb` para entrenar/probar modelos interactivamente (estilo Teachable Machine)
6. Configure y ejecute `scripts/prueba_video.py` para probar en tiempo real

## Uso de Jupyter

```bash
# Con uv:
uv run jupyter notebook

# Con pip (entorno activado):
jupyter notebook
```

Luego navegue a la carpeta `labs/` y abra el notebook deseado.
