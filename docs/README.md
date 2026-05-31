# Documentos del laboratorio combinado IA + Visión

Guías y ficha de evaluación del laboratorio de Inteligencia Artificial y
Visión por Computadora (1MTR56 — Automatización Industrial Inteligente B).
La **fuente de verdad es LaTeX** (`latex/*.tex`); los PDF se generan con
XeLaTeX.

## Estructura

```
latex/
├── plantilla_pucp.tex     Estilos institucionales (fuentes, portada, header/footer)
├── guia1.tex              Guía 1 — CNN, fundamentos (notebooks lab1 + lab2)
├── guia2.tex              Guía 2 — Transfer Learning + Beckhoff (proyecto/entrenar + probar_modelo)
└── fe_lab.tex             Ficha de Evaluación única del laboratorio (G1 /10 + G2 /10 = /20)

docs/
├── assets/                Figuras embebidas (g1/, g2/, branding/)
├── build/                 PDFs generados (no versionar)
├── archivo/               Guías y FE históricas (versiones anteriores)
└── imagenes_a_descargar.md  Catálogo de figuras + atribución
```

## Evaluación

Una sola ficha por pareja (`fe_lab.tex`). La nota del laboratorio es la
suma de ambas guías, **sobre 20**:

- **Guía 1** (CNN, fundamentos): 10 puntos.
- **Guía 2** (Transfer Learning + Beckhoff): 10 puntos.

## Flujo de trabajo

1. Editar el contenido en LaTeX (`latex/*.tex`).
2. Generar los PDF con `./scripts/build_latex.sh` desde la raíz del proyecto.
3. Los PDF quedan en `docs/build/`. Ese es el entregable final para los alumnos.

## Comandos

```bash
# Generar todos los documentos
./scripts/build_latex.sh

# Generar solo uno
./scripts/build_latex.sh guia1
./scripts/build_latex.sh guia2
./scripts/build_latex.sh fe_lab
```

## Requisitos

- `xelatex` (TeX Live 2022 o superior).
- `fonts-liberation` (Liberation Sans = Arial libre; cae a DejaVu Sans si falta).
- `python3` + `matplotlib` (solo para regenerar las figuras de G1 con
  `scripts/generar_figuras_g1.py`).
