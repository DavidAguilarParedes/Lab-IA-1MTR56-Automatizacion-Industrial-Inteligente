<!--
================================================================================
PONTIFICIA UNIVERSIDAD CATÓLICA DEL PERÚ
FACULTAD DE CIENCIAS E INGENIERÍA — INGENIERÍA MECATRÓNICA
2026-1 · 1MTR56 — AUTOMATIZACIÓN INDUSTRIAL INTELIGENTE B
================================================================================
Ficha de Evaluación — Guía 1 (CNN, fundamentos)
Plantilla institucional adaptada de FE_2026-1_IA.docx
================================================================================
-->

<div align="center">

# Ficha de Evaluación

## Laboratorio combinado de Inteligencia Artificial y Visión por Computadora

### Guía 1: Redes Neuronales Convolucionales — Fundamentos

</div>

---

**Horario:** _______________ **Fecha:** _______________ **Mesa de trabajo:** ______

### Identificación de la pareja

| Nombres completos | Código PUCP |
|---|---|
| Alumno A: | |
| Alumno B: | |

---

**Nota total Guía 1:** _______ / 15 puntos *(componente formativo — 30% del laboratorio)*

---

## Parte práctica

| Sección | Criterio | Puntaje |
|---|---|---:|
| **Actividad 1. Preparación conceptual** | Responder en formato breve (2–4 líneas cada una) las preguntas conceptuales del notebook sobre el rol de la capa convolucional, la función de activación ReLU y la analogía del learning rate con la ganancia proporcional de un controlador. | 2 pts |
| **Actividad 2. Implementación de la arquitectura** | Adjuntar la salida de `model.summary()` y verificar que el número total de parámetros coincide con el esperado para la arquitectura propuesta en la sección 7.2. | 1 pt |
| | Adjuntar el diagrama del modelo generado con `tf.keras.utils.plot_model`. | 1 pt |
| **Actividad 3. Entrenamiento y análisis** | Adjuntar las curvas de loss y accuracy de entrenamiento y validación. Comentar si se observa underfitting, ajuste adecuado o overfitting, justificando con la forma de las curvas. | 2 pts |
| | Adjuntar la matriz de confusión sobre el conjunto de validación y reportar la accuracy final. Comentar las confusiones específicas, si las hay. | 1.5 pts |
| **Actividad 4. Experimentación con hiperparámetros** | Reportar tres experimentos con learning rates distintos (orden de magnitud: 1e-2, 1e-4, 1e-5). Adjuntar las curvas correspondientes y justificar el comportamiento observado en cada caso. | 2.5 pts |
| **Actividad 5. Evaluación sobre datos propios** | Adjuntar cuatro fotografías capturadas por la pareja en condiciones variadas (iluminación, fondo, ángulo). Las fotografías deben incluir botellas de vidrio y de plástico, al menos una de cada en condiciones distintas a las del dataset de entrenamiento. | 1.5 pts |
| | Reportar en una tabla: imagen, clase real, clase predicha y nivel de confianza. Identificar en cuáles fotografías el modelo predice correctamente y en cuáles falla. | 1.5 pts |
| **Actividad 6. Discusión y análisis crítico** | Formular al menos dos hipótesis técnicas sobre las causas de los errores observados en la Actividad 5 (por ejemplo, sesgos del dataset, sensibilidad a la iluminación, limitaciones de la arquitectura). Cada hipótesis debe estar justificada con argumentos técnicos. | 1.5 pts |
| | Proponer al menos dos estrategias para mejorar el desempeño del modelo sin modificar la arquitectura propuesta. Argumentar la elección. | 0.5 pts |

---

## Material a adjuntar a la entrega

1. Notebook `lab1_cnn_basica.ipynb` completamente ejecutado (formato `.ipynb` y exportación `.html`).
2. Carpeta `fotos_g1_propias/` con las cuatro fotografías capturadas por la pareja.
3. Capturas solicitadas en las actividades 2 a 5 incrustadas en la presente ficha.

---

## Criterios generales de evaluación

- Las respuestas deben presentar argumentación técnica, no descripciones literales del procedimiento.
- Las figuras adjuntadas deben estar correctamente referenciadas y comentadas en el texto.
- La nomenclatura debe ser consistente con la utilizada en la Guía 1.
- Se valorará la claridad expositiva y la profundidad del análisis crítico (Actividad 6).
