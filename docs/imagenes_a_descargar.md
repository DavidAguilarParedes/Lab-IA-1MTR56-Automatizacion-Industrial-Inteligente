# Imágenes para las guías — lista de descarga y atribución

> Cada figura referenciada como `[FIGURA N: ...]` en las guías debe reemplazarse por una de las
> imágenes listadas abajo. Guardar en `docs/assets/` con el nombre sugerido (col. "Archivo destino").
> Las marcadas como **Generar** se producen ejecutando el código indicado en el notebook.

## Guía 1 — CNN básica

| # | Tipo | Descripción | Fuente sugerida | URL / cómo obtenerla | Archivo destino | Atribución a poner al pie |
|---|---|---|---|---|---|---|
| 1 | Diagrama propio | Flujo end-to-end: cámara → CNN → 3 BOOLs → Beckhoff → brazo | **Diagrama propio** | Crear en draw.io o Excalidraw. Plantilla 16:9, paleta PUCP. | `g1_fig01_pipeline.png` | Fuente: elaboración propia |
| 2 | Comparativa | Feature extraction tradicional vs CNN aprende features | Stanford CS231n | https://cs231n.github.io/classification/ — buscar imagen de comparativa | `g1_fig02_classical_vs_cnn.png` | Fuente: Stanford CS231n (Karpathy et al.) |
| 3 | Diagrama arquitectura | CNN completa input→conv→pool→...→softmax con etiquetas | **NN-SVG** | https://alexlenail.me/NN-SVG/LeNet.html — exportar SVG, ajustar para nuestras dimensiones (64×64×3) | `g1_fig03_cnn_arch.svg` | Fuente: NN-SVG (Lenail, 2019) — adaptación |
| 4 | Animación convolución | Kernel 3×3 deslizándose sobre imagen | setosa.io | https://setosa.io/ev/image-kernels/ — captura GIF o frames clave | `g1_fig04_conv_anim.gif` | Fuente: Powell — setosa.io |
| 5 | Stride / padding | Comparativa visual stride=1 vs 2; padding valid vs same | TensorFlow docs | https://www.tensorflow.org/api_docs/python/tf/nn/convolution — o CS231n | `g1_fig05_stride_padding.png` | Fuente: TensorFlow docs |
| 6 | Funciones activación | Gráficas ReLU, Leaky ReLU, Sigmoid, Tanh | Wikipedia | https://en.wikipedia.org/wiki/Activation_function — figura comparativa | `g1_fig06_activations.png` | Fuente: Wikipedia (CC BY-SA) |
| 7 | Pooling | Max pool vs Avg pool con números | CS231n | https://cs231n.github.io/convolutional-networks/#pool | `g1_fig07_pooling.png` | Fuente: Stanford CS231n |
| 8 | Fully connected | Esquema de capa densa con conexiones | NN-SVG | https://alexlenail.me/NN-SVG/index.html — variante FCNN | `g1_fig08_dense.svg` | Fuente: NN-SVG |
| 9 | Training loop | Diagrama circular forward → loss → backward → optimizer | **Diagrama propio** | Excalidraw o draw.io | `g1_fig09_training_loop.png` | Fuente: elaboración propia |
| 10 | Curvas underfit/fit/overfit | 3 paneles típicos de loss vs epoch | Towards Data Science / Karpathy | https://karpathy.github.io/2019/04/25/recipe/ o blog TDS | `g1_fig10_underfit_overfit.png` | Fuente: A. Karpathy (2019) |
| 11 | Arquitectura del lab | Diagrama de la red usada (Conv32-MaxPool-Conv32-MaxPool-Flatten-Dense64-Dense2) | **Generar** | En notebook: `tf.keras.utils.plot_model(model, show_shapes=True, to_file='g1_fig11.png')` | `g1_fig11_model_arch.png` | Fuente: elaboración propia |
| 12 | Estructura de carpetas | Captura del filesystem `data/botellas/train|validation` | **Captura propia** | Captura del repo (VSCode tree o `tree` en terminal) | `g1_fig12_dataset_tree.png` | Fuente: elaboración propia |
| 13 | Curvas entrenamiento | Resultado esperado de loss/accuracy de train/val | **Generar** | En notebook tras entrenar — matplotlib | `g1_fig13_curves_expected.png` | Fuente: elaboración propia |
| 14 | Matriz confusión | Resultado esperado 2×2 vidrio/plástico | **Generar** | En notebook — seaborn.heatmap | `g1_fig14_confmat_expected.png` | Fuente: elaboración propia |
| 15 | Antes/después G1→G2 | CNN simple limitada vs MobileNetV2 robusto | **Diagrama propio** | Excalidraw | `g1_fig15_bridge_g2.png` | Fuente: elaboración propia |

## Recursos generales para producir imágenes propias

| Herramienta | Para qué | URL |
|---|---|---|
| **NN-SVG** | Diagramas de redes neuronales editables | https://alexlenail.me/NN-SVG/ |
| **Excalidraw** | Diagramas a mano alzada con estilo "didáctico" | https://excalidraw.com |
| **draw.io / diagrams.net** | Diagramas profesionales con paleta corporativa | https://app.diagrams.net |
| **setosa.io/ev** | Visualizaciones interactivas de kernels e ML | https://setosa.io/ev/ |
| **Stanford CS231n notes** | Material de referencia académico (citar) | https://cs231n.github.io/ |
| **tf.keras.utils.plot_model** | Diagrama generado del modelo Keras | docs TF |

## Notas de licencia

- **CS231n** — material educacional, OK para uso académico citando.
- **NN-SVG** — generador, las salidas son tuyas. Citar al autor por cortesía.
- **Wikipedia** — figuras suelen ser CC BY-SA. Mantener atribución y licencia.
- **TensorFlow / Keras docs** — Apache 2.0 / CC BY 4.0 según figura, citar.
- **Karpathy blog** — uso académico citando.

> ⚠️ Si una figura encontrada **no tiene licencia clara**, NO la uses. Genera la tuya
> con NN-SVG / Excalidraw / código del notebook.
