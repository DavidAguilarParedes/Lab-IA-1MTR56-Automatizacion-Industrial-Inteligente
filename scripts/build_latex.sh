#!/usr/bin/env bash
# =============================================================================
# Generador de PDFs en LaTeX para guías y FE del laboratorio combinado
# PUCP · Ing. Mecatrónica · 1MTR56 — Automatización Industrial Inteligente B
#
# Compila con XeLaTeX (necesario para fontspec / Liberation Sans).
# Doble pasada por documento para asegurar TOC y referencias cruzadas.
#
# Uso:
#   ./scripts/build_latex.sh                 # genera todos los documentos
#   ./scripts/build_latex.sh guia1           # solo G1
#   ./scripts/build_latex.sh fe_lab          # solo la ficha de evaluación
#   ./scripts/build_latex.sh guia1 guia2     # múltiples
#
# Requisitos:
#   - xelatex (TeX Live 2022 o superior)
#   - fonts-liberation (Liberation Sans = Arial libre)
#   - python3 + matplotlib   (solo si hay que regenerar figuras matplotlib)
#
# Salida: docs/build/*.pdf
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LATEX="$ROOT/latex"
BUILD="$ROOT/docs/build"
mkdir -p "$BUILD"

# Verificar dependencias mínimas
command -v xelatex >/dev/null 2>&1 || { echo "ERROR: xelatex no instalado. Instalar TeX Live." >&2; exit 1; }
fc-list | grep -qi "liberation sans" || \
    echo "ADVERTENCIA: Liberation Sans no detectada. Se usará DejaVu Sans como fallback." >&2

# Documentos disponibles (clave → archivo .tex)
declare -A DOCUMENTS=(
    [guia1]="guia1.tex"
    [guia2]="guia2.tex"
    [fe_lab]="fe_lab.tex"
    [guia_lab]="guia_laboratorio.tex"
)

# Verificar si las figuras matplotlib están presentes
verificar_figuras_g1() {
    local fig_dir="$ROOT/docs/assets/g1"
    if [[ ! -f "$fig_dir/g1_fig05_convolution.png" ]]; then
        echo "Figuras matplotlib de G1 no encontradas; regenerando..."
        ( cd "$ROOT" && python scripts/generar_figuras_g1.py )
    fi
}

build_one() {
    local key="$1"
    local src="${DOCUMENTS[$key]:-}"
    if [[ -z "$src" ]]; then
        echo "WARN: documento '$key' no reconocido (claves: ${!DOCUMENTS[*]})" >&2
        return 1
    fi
    local input="$LATEX/$src"
    if [[ ! -f "$input" ]]; then
        echo "SKIP: $input no existe todavía" >&2
        return 0
    fi

    # Verificar dependencias específicas por documento
    case "$key" in
        guia1|guia_lab) verificar_figuras_g1 ;;
    esac

    local base="${src%.tex}"
    echo "→ Compilando $src (doble pasada)..."
    ( cd "$LATEX" && \
      xelatex -interaction=nonstopmode "$src" > /dev/null && \
      xelatex -interaction=nonstopmode "$src" > /dev/null )

    mv -f "$LATEX/$base.pdf" "$BUILD/$base.pdf"
    # Limpieza de archivos auxiliares
    rm -f "$LATEX/$base."{aux,log,out,toc}
    echo "  ✓ $BUILD/$base.pdf"
}

# Punto de entrada
if [[ $# -eq 0 ]]; then
    for key in "${!DOCUMENTS[@]}"; do
        build_one "$key" || true
    done
else
    for key in "$@"; do
        build_one "$key"
    done
fi

echo ""
echo "✔ PDFs generados en $BUILD/"
ls -lh "$BUILD"/*.pdf 2>/dev/null | awk '{printf "  %-40s %s\n", $9, $5}'
