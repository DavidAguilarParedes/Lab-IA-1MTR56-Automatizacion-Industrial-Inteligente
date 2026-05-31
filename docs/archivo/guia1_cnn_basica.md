<!--
================================================================================
PONTIFICIA UNIVERSIDAD CATÓLICA DEL PERÚ
FACULTAD DE CIENCIAS E INGENIERÍA — INGENIERÍA MECATRÓNICA
2026-1 · 1MTR56 — AUTOMATIZACIÓN INDUSTRIAL INTELIGENTE B
================================================================================
Documento fuente en Markdown. Exportar a PDF con tipografía Montserrat
(secundaria oficial PUCP). Reemplazar los bloques [FIGURA N] por las
imágenes listadas en docs/imagenes_a_descargar.md antes de generar el PDF.
================================================================================
-->

<div align="center">

# Laboratorio combinado de Inteligencia Artificial y Visión por Computadora

## Guía 1: Redes Neuronales Convolucionales (CNN) — Fundamentos

</div>

---

### Contenido

1. Objetivos
2. Materiales
3. Visión por Computadora
   3.1 Definición
   3.2 Visión por Computadora utilizando Feature Extraction
   3.3 Visión por Computadora utilizando Inteligencia Artificial
4. Redes Neuronales Convolucionales
   4.1 Capas convolucionales
   4.2 Stride y padding
   4.3 Capas de pooling
   4.4 Funciones de activación
   4.5 Capas fully connected
5. Proceso de entrenamiento
   5.1 Inicialización
   5.2 Forward propagation
   5.3 Función de pérdida
   5.4 Backpropagation
   5.5 Optimización
6. Evaluación del modelo
   6.1 Conjunto de validación
   6.2 Métricas de rendimiento
7. Experiencia guiada: clasificador de botellas
   7.1 Preparación del dataset
   7.2 Diseño y compilación de la arquitectura
   7.3 Entrenamiento
   7.4 Matriz de confusión y métricas
   7.5 Predicción sobre imágenes individuales
8. Anexos
   A1. Optimizadores comunes
   A2. Funciones de pérdida
   A3. Parámetros del objeto `model`
   A4. Métricas de evaluación

---

## 1. Objetivos

- Introducir al estudiante en los fundamentos teóricos de la visión por computadora moderna basada en aprendizaje profundo.
- Familiarizar al estudiante con la arquitectura y los componentes de una Red Neuronal Convolucional (CNN).
- Comprender el proceso de entrenamiento de una CNN: forward propagation, función de pérdida, backpropagation y optimización.
- Implementar en Python (TensorFlow/Keras) una CNN simple para clasificación de imágenes.
- Evaluar el rendimiento de un modelo entrenado mediante curvas de aprendizaje y matriz de confusión.

## 2. Materiales

- Computadora con Python 3.10 o superior y entorno virtual del laboratorio activo.
- Dataset de botellas (vidrio y plástico), provisto en `data/botellas/`.
- Notebook de trabajo: `labs/lab1_cnn_basica.ipynb`.
- Librerías: `tensorflow`, `numpy`, `matplotlib`, `seaborn`.

---

## 3. Visión por Computadora

### 3.1 Definición

La visión por computadora es una disciplina dentro de la inteligencia artificial cuyo objetivo es dotar a las máquinas de la capacidad de interpretar el mundo visual de forma análoga a como lo hace el sistema visual humano. Mediante la combinación de cámaras, algoritmos y poder de cómputo, esta disciplina busca extraer información significativa a partir de imágenes y secuencias de video.

Las aplicaciones industriales son numerosas: control de calidad por inspección visual, identificación y trazabilidad de productos, asistencia a operarios mediante realidad aumentada, supervisión de procesos críticos, y guiado de robots colaborativos. En todos estos casos, la imagen actúa como una señal sensorial cuya interpretación habilita la toma de decisiones automatizada.

![](assets/g1/g1_fig01_pipeline.png)

*Figura 1. Pipeline del laboratorio combinado. Fuente: elaboración propia.*

### 3.2 Visión por Computadora utilizando Feature Extraction

El enfoque tradicional de la visión por computadora se basa en el diseño manual de descriptores que capturan información geométrica o estadística de la imagen. Entre las técnicas más representativas se encuentran la segmentación por color en el espacio HSV, la transformada de Hough para detección de líneas y círculos, y descriptores como SIFT (*Scale-Invariant Feature Transform*) y SURF (*Speeded Up Robust Features*) para la detección de puntos de interés invariantes a escala y rotación.

Estos métodos resultan eficaces y computacionalmente eficientes cuando las condiciones del entorno son controladas: iluminación estable, fondo conocido y geometría predecible de los objetos. Sin embargo, su principal limitación es la fragilidad ante variaciones no anticipadas. Cambios de iluminación, oclusiones parciales, deformaciones, o la introducción de nuevas clases de objetos típicamente requieren un rediseño manual del pipeline.

![](assets/g1/g1_fig02_classical.png)

*Figura 2. Visión por computadora clásica con features diseñadas manualmente. Fuente: elaboración propia.*

### 3.3 Visión por Computadora utilizando Inteligencia Artificial

El surgimiento del aprendizaje profundo, y en particular de las Redes Neuronales Convolucionales (CNN), modificó radicalmente este paradigma. En lugar de diseñar manualmente los descriptores, una CNN aprende **directamente de los datos** qué características son relevantes para la tarea. Las primeras capas tienden a aprender detectores de bordes y texturas, mientras que las capas más profundas aprenden representaciones cada vez más abstractas (partes, objetos enteros).

Este enfoque presenta tres ventajas frente al método tradicional. Primero, permite manejar variaciones complejas (iluminación, ángulos, escalas) sin reformular el algoritmo. Segundo, escala de manera natural a problemas multiclase. Tercero, aprovecha la disponibilidad creciente de grandes datasets y de cómputo paralelo en GPU, lo que ha permitido alcanzar y superar el desempeño humano en numerosas tareas de clasificación visual.

![](assets/g1/g1_fig03_cnn_pipeline.png)

*Figura 3. Visión por computadora moderna basada en CNN end-to-end. Fuente: elaboración propia.*

> *Analogía mecatrónica.* En control clásico, el diseñador define manualmente la función de transferencia del controlador. En control basado en aprendizaje, los parámetros del controlador se ajustan a partir de datos. Las CNN ocupan el papel de un "controlador adaptativo" sobre el dominio visual.

---

## 4. Redes Neuronales Convolucionales

Las CNN son una clase de redes neuronales especializadas en el procesamiento de datos con estructura de rejilla, como imágenes (rejilla 2D de píxeles con canales de color). Se diferencian de las redes neuronales totalmente conectadas tradicionales por la introducción de **capas convolucionales**, que aprovechan la localidad espacial y la invariancia traslacional inherentes a las imágenes.

![](assets/g1/g1_fig04_cnn_arch.png)

*Figura 4. Arquitectura completa de la CNN utilizada en la experiencia guiada. Fuente: elaboración propia.*

### 4.1 Capas convolucionales

Una capa convolucional aplica un conjunto de **kernels** (también llamados filtros) sobre la imagen de entrada. Cada kernel es una pequeña matriz de pesos (típicamente 3×3 o 5×5) que se desliza espacialmente sobre la entrada realizando, en cada posición, una multiplicación elemento a elemento seguida de una suma. El resultado es un **mapa de características** (*feature map*) que resalta la presencia del patrón codificado en el kernel.

Durante el entrenamiento, los pesos de los kernels se ajustan automáticamente para que la red aprenda los patrones más útiles para la tarea. Una capa con $N$ filtros produce $N$ mapas de características, formando un tensor de salida de profundidad $N$.

![](assets/g1/g1_fig05_convolution.png)

*Figura 5. Operación de convolución 2D. El kernel Sobel-x detecta gradientes horizontales. Fuente: elaboración propia.*

### 4.2 Stride y padding

El **stride** indica el paso con el que el kernel se desplaza sobre la imagen. Un stride de 1 produce una salida del mismo tamaño espacial que la entrada (asumiendo padding adecuado), mientras que un stride de 2 reduce las dimensiones espaciales a la mitad.

El **padding** consiste en añadir píxeles (típicamente con valor cero) en los bordes de la imagen antes de la convolución. El modo `same` ajusta el padding para preservar las dimensiones espaciales de salida; el modo `valid` no añade padding y reduce la salida en función del tamaño del kernel.

![](assets/g1/g1_fig06_stride_padding.png)

*Figura 6. Efectos de stride y padding sobre el tamaño espacial de la salida. Fuente: elaboración propia.*

### 4.3 Capas de pooling

Las capas de pooling reducen las dimensiones espaciales de los mapas de características, conservando la información más relevante. **Max pooling** retiene el valor máximo dentro de cada ventana, mientras que **average pooling** retiene el promedio. El pooling cumple tres funciones: reducir el número de parámetros y el costo computacional de las capas posteriores, mitigar el sobreajuste, e introducir un grado de invariancia a pequeñas traslaciones.

![](assets/g1/g1_fig07_pooling.png)

*Figura 7. Comparativa entre max pooling y average pooling sobre una entrada 4×4. Fuente: elaboración propia.*

### 4.4 Funciones de activación

Las funciones de activación introducen no-linealidades en la red, sin las cuales una pila de capas convolucionales colapsaría a una única transformación lineal y la red sería incapaz de aprender relaciones complejas. La función más utilizada es **ReLU** (*Rectified Linear Unit*), definida como $f(x) = \max(0, x)$, por su simplicidad computacional y por mitigar el problema del desvanecimiento del gradiente. Variantes como Leaky ReLU permiten pequeños valores negativos, atenuando el problema de las "neuronas muertas".

![](assets/g1/g1_fig08_activations.png)

*Figura 8. Funciones de activación más utilizadas en redes neuronales convolucionales. Fuente: elaboración propia.*

### 4.5 Capas fully connected

Al final de la red, las capas fully connected (también llamadas densas) combinan las características aprendidas por las capas convolucionales y producen la salida final. En problemas de clasificación, la última capa densa tiene tantas neuronas como clases, y emplea la función de activación **softmax**, que normaliza las salidas para que sumen 1, interpretándose como una distribución de probabilidad sobre las clases.

*El esquema de capas fully connected es el mostrado en la Figura 4 (bloques Dense 64 y Dense 2).*

---

## 5. Proceso de entrenamiento

El entrenamiento de una CNN consiste en ajustar iterativamente los pesos de la red (los valores de los kernels y de las capas densas) para minimizar una función de pérdida que mide la discrepancia entre las predicciones de la red y las etiquetas reales.

![](assets/g1/g1_fig09_training_loop.png)

*Figura 9. Ciclo de entrenamiento iterativo de una red neuronal. Fuente: elaboración propia.*

### 5.1 Inicialización

Los pesos de la red se inicializan típicamente con valores aleatorios pequeños, siguiendo esquemas como **Glorot** (Xavier) o **He**, diseñados para que la varianza de las activaciones se mantenga estable a lo largo de las capas. Una inicialización adecuada es crítica para garantizar una convergencia eficiente.

### 5.2 Forward propagation

La imagen de entrada se propaga capa por capa a través de la red. Cada capa transforma su entrada según los pesos actuales y la función de activación correspondiente, produciendo finalmente la predicción $\hat{y}$.

### 5.3 Función de pérdida

La función de pérdida cuantifica el error de la predicción respecto a la etiqueta real. Para clasificación multiclase se utiliza típicamente la **entropía cruzada** (*categorical cross-entropy*), que mide la divergencia entre la distribución predicha y la distribución real (codificada como vector one-hot). Para regresión se emplean el error cuadrático medio (MSE) o el error absoluto medio (MAE).

*Para una interpretación gráfica de la entropía cruzada y otras funciones de pérdida, ver la referencia [Goodfellow et al., Deep Learning, Cap. 5].*

### 5.4 Backpropagation

El algoritmo de backpropagation calcula el gradiente de la función de pérdida respecto a cada uno de los pesos de la red, aplicando la regla de la cadena del cálculo diferencial. Estos gradientes indican la dirección y magnitud en que cada peso debe ajustarse para reducir el error.

*El flujo de gradientes en backpropagation se representa por las flechas inversas del ciclo mostrado en la Figura 9.*

### 5.5 Optimización

Los **optimizadores** son los algoritmos que actualizan los pesos a partir de los gradientes calculados. El más simple es el descenso de gradiente estocástico (**SGD**), que aplica la regla $w \leftarrow w - \eta \cdot \nabla L$, donde $\eta$ es la tasa de aprendizaje (*learning rate*). Variantes adaptativas como **Adam** o **RMSprop** ajustan automáticamente la tasa de aprendizaje por parámetro, y son particularmente efectivas en redes profundas. Adam es, en la práctica, el optimizador por defecto en la mayoría de aplicaciones.

> *Analogía mecatrónica.* El optimizador puede interpretarse como un controlador que regula la trayectoria de los pesos hacia el mínimo de la función de pérdida. La tasa de aprendizaje cumple el papel de la ganancia proporcional: una ganancia excesiva genera inestabilidad, mientras que una ganancia insuficiente produce convergencia lenta.

Una lista de optimizadores y sus casos de uso se presenta en el **Anexo A1**.

---

## 6. Evaluación del modelo

### 6.1 Conjunto de validación

El dataset disponible se divide en al menos dos subconjuntos disjuntos: **entrenamiento** (típicamente 70–80% de los datos) y **validación** (20–30% restante). El modelo ajusta sus pesos únicamente sobre el conjunto de entrenamiento, y se evalúa periódicamente sobre el conjunto de validación para estimar su capacidad de **generalización** — es decir, su desempeño sobre datos no observados durante el entrenamiento.

La discrepancia entre el desempeño de entrenamiento y el de validación es un indicador clave: si el modelo obtiene alto desempeño en entrenamiento pero bajo en validación, se encuentra en régimen de **sobreajuste** (*overfitting*) y ha memorizado el dataset sin aprender los patrones generales.

![](assets/g1/g1_fig10_curves.png)

*Figura 10. Diagnóstico de entrenamiento mediante curvas de loss. Fuente: elaboración propia.*

### 6.2 Métricas de rendimiento

Para problemas de clasificación, las métricas más utilizadas son:

- **Accuracy:** proporción de predicciones correctas. Adecuada cuando las clases están balanceadas.
- **Precision:** de las predicciones positivas, qué fracción es realmente positiva.
- **Recall:** de los positivos reales, qué fracción fue correctamente detectada.
- **F1-score:** media armónica entre precision y recall, útil cuando existe desbalance de clases.
- **Matriz de confusión:** tabla cruzada de clases reales vs. predichas, que permite identificar confusiones específicas entre pares de clases.

Un detalle de métricas adicionales se presenta en el **Anexo A4**.

---

## 7. Experiencia guiada: clasificador de botellas

En esta experiencia se construirá una CNN para clasificar imágenes de botellas en dos categorías: **vidrio** y **plástico**. El propósito es asimilar el flujo completo de trabajo (preparación del dataset, definición del modelo, entrenamiento, evaluación y predicción) sobre un problema reducido, antes de abordar la **Guía 2**, donde se introducirá el aprendizaje por transferencia (*transfer learning*) sobre un problema más complejo.

La arquitectura propuesta es deliberadamente sencilla: dos capas convolucionales con max pooling, una capa fully connected intermedia, y una capa de salida con activación softmax. La entrada tiene dimensiones 64×64×3 (imagen RGB).

*La arquitectura propuesta para esta experiencia ya fue presentada en la Figura 4.*

### 7.1 Preparación del dataset

El dataset se organiza en subcarpetas, una por clase, dentro de los directorios `train/` y `validation/`. Esta estructura permite que `ImageDataGenerator` infiera automáticamente las etiquetas.

```
data/botellas/
├── train/
│   ├── vidrio/      (~80% de las imágenes de vidrio)
│   └── plastico/    (~80% de las imágenes de plástico)
└── validation/
    ├── vidrio/      (~20% restantes)
    └── plastico/    (~20% restantes)
```

*Figura 11. Estructura de carpetas del dataset de botellas.*

**Script 1:** Generadores de datos para entrenamiento y validación.

```python
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir = 'data/botellas'

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen   = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
    os.path.join(base_dir, 'train'),
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

val_set = val_datagen.flow_from_directory(
    os.path.join(base_dir, 'validation'),
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)
```

La operación `rescale=1./255` normaliza los valores de los píxeles al rango $[0, 1]$. El argumento `class_mode='categorical'` indica que las etiquetas se entregarán en formato one-hot, compatible con la pérdida `categorical_crossentropy`.

### 7.2 Diseño y compilación de la arquitectura

**Script 2:** Definición y compilación del modelo.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')   # 2 clases: vidrio, plastico
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

La compilación define tres elementos esenciales: el **optimizador** (`adam`), la **función de pérdida** (`categorical_crossentropy`, adecuada para clasificación multiclase con etiquetas one-hot) y la **métrica de evaluación** (`accuracy`). Alternativas para cada uno se detallan en los **Anexos A1 y A2**.

### 7.3 Entrenamiento

**Script 3:** Entrenamiento del modelo y registro de métricas.

```python
history = model.fit(
    train_set,
    steps_per_epoch=train_set.samples // train_set.batch_size,
    epochs=20,
    validation_data=val_set,
    validation_steps=val_set.samples // val_set.batch_size
)

loss, accuracy = model.evaluate(val_set)
print(f'Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')
```

El parámetro `epochs` indica el número de veces que el modelo recorre el conjunto de entrenamiento completo. Un valor demasiado bajo conduce a **underfitting** (el modelo no alcanza a aprender); un valor demasiado alto, a **overfitting**. El parámetro `steps_per_epoch` determina cuántos batches se procesan por época.

![](assets/g1/g1_fig12_training_curves.png)

*Figura 12. Curvas de aprendizaje esperadas tras 20 epochs de entrenamiento. Fuente: elaboración propia.*

### 7.4 Matriz de confusión y métricas

**Script 4:** Construcción de la matriz de confusión.

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

predictions = model.predict(val_set, verbose=1)
y_pred = tf.argmax(predictions, axis=1).numpy()
y_true = val_set.classes

num_classes = len(val_set.class_indices)
confusion_mtx = np.zeros((num_classes, num_classes), dtype=np.int32)
for t, p in zip(y_true, y_pred):
    confusion_mtx[t][p] += 1

class_names = list(val_set.class_indices.keys())

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicción')
plt.ylabel('Etiqueta real')
plt.title('Matriz de confusión')
plt.show()
```

![](assets/g1/g1_fig11_confmat.png)

*Figura 13. Matriz de confusión esperada sobre el conjunto de validación. Fuente: elaboración propia.*

### 7.5 Predicción sobre imágenes individuales

**Script 5:** Función de predicción para una imagen arbitraria.

```python
import numpy as np
from tensorflow.keras.preprocessing import image

class_indices = {'plastico': 0, 'vidrio': 1}
class_labels = {v: k for k, v in class_indices.items()}

def predict_image(img_path, model, class_labels):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array, verbose=0)
    idx = int(np.argmax(prediction))
    label = class_labels[idx]
    confidence = float(prediction[0][idx])
    return f'Predicción: {label} (Confianza: {confidence:.2f})'

print(predict_image('data/botellas/validation/vidrio/ejemplo.jpg', model, class_labels))
```

Nótese que el diccionario `class_indices` debe respetar el orden alfabético asignado por `flow_from_directory`. Para tres clases de colores, por ejemplo, sería `{'amarillo': 0, 'azul': 1, 'rojo': 2}` — convención que se retomará en la **Guía 2**.

---

## 8. Anexos

### Anexo A1. Optimizadores comunes

| Optimizador | Descripción | Cuándo utilizarlo |
|---|---|---|
| **SGD** | Descenso de gradiente estocástico. Actualiza los pesos en la dirección opuesta al gradiente, escalado por la tasa de aprendizaje. | Modelos sencillos; control fino del LR; problemas donde Adam genera oscilaciones. |
| **Adam** | Combina momentum y RMSprop adaptativo. | Default razonable para la mayoría de problemas modernos, incluida visión por computadora. |
| **RMSprop** | Divide el gradiente por una media móvil de magnitudes recientes. | Redes recurrentes; gradientes con magnitud muy variable. |
| **Adagrad** | Aumenta el LR para parámetros poco frecuentes y lo reduce para los más comunes. | Datos dispersos o con muchas features categóricas. |
| **Adadelta** | Extensión de Adagrad que evita la disminución indefinida del LR. | Cuando Adagrad converge demasiado rápido a un LR pequeño. |
| **Adamax** | Variante de Adam basada en la norma infinito. | Redes profundas donde Adam es inestable. |
| **Nadam** | Adam con momento de Nesterov. | Problemas que se benefician del *lookahead* del momento de Nesterov. |

### Anexo A2. Funciones de pérdida

| Función de pérdida | Descripción | Cuándo utilizarla |
|---|---|---|
| **BinaryCrossentropy** | Distancia entre la salida binaria predicha y la etiqueta real. | Clasificación binaria con activación sigmoid. |
| **CategoricalCrossentropy** | Distancia entre la distribución softmax predicha y la etiqueta one-hot. | Clasificación multiclase con etiquetas one-hot. |
| **SparseCategoricalCrossentropy** | Equivalente a `CategoricalCrossentropy` pero acepta etiquetas enteras directamente. | Clasificación multiclase sin necesidad de codificar one-hot. |
| **MeanSquaredError (MSE)** | Promedio de los errores cuadrados. | Regresión; penaliza fuertemente errores grandes. |
| **MeanAbsoluteError (MAE)** | Promedio de los errores absolutos. | Regresión robusta a outliers. |
| **Huber** | Híbrida entre MSE y MAE. | Regresión robusta donde se desea suavidad en el origen. |
| **Hinge** | Pérdida usada en máquinas de vectores de soporte. | Clasificación binaria con margen máximo. |
| **CosineSimilarity** | Similitud por coseno entre vectores. | Cuando importa la dirección y no la magnitud (embeddings). |

### Anexo A3. Parámetros del objeto `model`

| Capa / argumento | Opciones | Sintaxis |
|---|---|---|
| `Conv2D` — número de filtros | 32, 64, 128, 256, … | `Conv2D(32, ...)` |
| `Conv2D` — tamaño de kernel | (3,3), (5,5), (7,7) | `Conv2D(32, (3,3))` |
| `Conv2D` — activación | `relu`, `elu`, `selu`, `swish`, `sigmoid`, `tanh` | `Conv2D(32, (3,3), activation='relu')` |
| `Conv2D` — `input_shape` | `(H, W, C)`. Solo en la primera capa. | `Conv2D(32, (3,3), input_shape=(64,64,3))` |
| `MaxPooling2D` — `pool_size` | (2,2), (3,3) | `MaxPooling2D(pool_size=(2,2))` |
| `MaxPooling2D` — `strides` | (1,1), (2,2), … | `MaxPooling2D(pool_size=(2,2), strides=(2,2))` |
| `Flatten` | Sin parámetros. | `Flatten()` |
| `Dense` — `units` | 32, 64, 128, 256, 512 | `Dense(128)` |
| `Dense` — activación | `relu`, `softmax`, `sigmoid`, `linear` | `Dense(128, activation='relu')` |
| `Dropout` — `rate` | 0.1 a 0.5 | `Dropout(0.5)` |
| Capa de salida | `Dense(N, activation='softmax')` para N clases | `Dense(2, activation='softmax')` |
| `model.compile` — `optimizer` | `adam`, `sgd`, `rmsprop`, … | ver Anexo A1 |
| `model.compile` — `loss` | `categorical_crossentropy`, `sparse_categorical_crossentropy`, … | ver Anexo A2 |
| `model.compile` — `metrics` | `['accuracy']`, `['precision']`, … | ver Anexo A4 |

### Anexo A4. Métricas de evaluación

| Métrica | Descripción | Cuándo utilizarla |
|---|---|---|
| **Accuracy** | Proporción de predicciones correctas. | Clasificación con clases balanceadas. |
| **BinaryAccuracy** | Accuracy adaptada a salida sigmoid binaria. | Clasificación binaria. |
| **CategoricalAccuracy** | Accuracy para etiquetas one-hot. | Clasificación multiclase con `categorical_crossentropy`. |
| **SparseCategoricalAccuracy** | Accuracy para etiquetas enteras. | Clasificación multiclase con `sparse_categorical_crossentropy`. |
| **TopKCategoricalAccuracy** | Acierto si la clase real está entre las K más probables. | Clasificación con muchas clases (ImageNet, etc.). |
| **Precision** | TP / (TP + FP). | Cuando interesa minimizar falsos positivos. |
| **Recall** | TP / (TP + FN). | Cuando interesa minimizar falsos negativos. |
| **F1-score** | Media armónica de precision y recall. | Clases desbalanceadas. |
| **AUC** | Área bajo la curva ROC. | Clasificación binaria con probabilidades. |
| **MSE / MAE** | Error cuadrático / absoluto medio. | Regresión. |

---

*Fin de la Guía 1. Continúa con la Guía 2: Transfer Learning con MobileNetV2, clasificación de tapitas e integración con IPC Beckhoff vía pyADS.*
