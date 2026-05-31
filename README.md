# Laboratorio de IA — Clasificacion de Imagenes con CNN

## Instalacion

### 1. Instalar Python 3.10

Descargar de https://www.python.org/downloads/release/python-31011/

**IMPORTANTE:** Al instalar, marcar la casilla **"Add Python to PATH"**.

### 2. Descargar este proyecto

Descargar el ZIP desde GitHub (boton verde "Code" > "Download ZIP") y descomprimirlo.

O si tiene git:
```
git clone https://github.com/DavidAguilarParedes/Lab-IA-1MTR56-Automatizacion-Industrial-Inteligente.git
```

### 3. Instalar dependencias

Abrir la carpeta del proyecto en **Visual Studio Code**. Luego abrir una terminal (Menu: Terminal > New Terminal) y ejecutar:

```
python -m venv .venv
```

VS Code detectara el entorno virtual y preguntara si desea usarlo. Hacer clic en **"Yes"**.

Cerrar la terminal y abrir una nueva (para que use el entorno virtual). Luego:

```
pip install -r requirements.txt
```

La instalacion puede tardar varios minutos por TensorFlow. Es normal.

### 4. Verificar

En la terminal de VS Code:
```
python -c "import tensorflow; print(tensorflow.__version__)"
```

Si imprime la version (ej: `2.20.0`), todo esta listo.

---

## Como usar

### Notebooks (labs y proyecto)

Abrir cualquier archivo `.ipynb` en VS Code y ejecutar las celdas.

**Parte 1 — IA** (`parte1_ia/`):
- `notebook1_cnn_desde_cero.ipynb` — CNN desde cero
- `notebook2_data_augmentation.ipynb` — Data augmentation
- `notebook3_clasificador_tapitas.ipynb` — Clasificador de tapitas (CNN, dataset propio)

**Parte 2 — Visión** (`parte2_vision/`):
- `notebook4_transfer_learning.ipynb` — Entrenar con Transfer Learning (MobileNetV2)
- `notebook5_probar_modelo.ipynb` — Probar modelo en vivo
- `notebook6_envio_beckhoff.ipynb` — Enviar predicciones al PLC Beckhoff (ADS)

### Integración con Beckhoff (PLC)

La demostración con el brazo se hace desde `notebook6_envio_beckhoff.ipynb`
(cámara → predicción → 3 BOOLs por ADS). Requiere `pyads` y TwinCAT.

1. Seleccionar modelo en el panel derecho y hacer clic en "Cargar"
2. La camara muestra la prediccion en tiempo real
3. "EJECUTAR INSPECCION" clasifica y envia resultado al PLC
4. Para conectar al PLC: abrir "Config PLC" e ingresar AMS Net ID

---

## Solucion de problemas

| Problema | Solucion |
|---|---|
| `python` no se reconoce | Reinstalar Python marcando "Add Python to PATH" |
| No module named tensorflow | Cerrar y reabrir la terminal de VS Code |
| La camara no abre | Cerrar otras apps que usen la camara |
| pip install tarda mucho | Normal, TensorFlow es grande (~500 MB) |
| TensorFlow falla al instalar | Verificar que tiene Python 3.10 (`python --version`) |
