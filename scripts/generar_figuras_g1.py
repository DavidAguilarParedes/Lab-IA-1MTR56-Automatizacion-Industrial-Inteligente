"""
Generador de figuras matplotlib para la Guía 1 — CNN básica.

Estilo: académico con paleta de colores PUCP + complementarios.
Tipografía Liberation Sans (Arial libre) para consistencia con LaTeX.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Paleta institucional + complementarios académicos
# -----------------------------------------------------------------------------
PUCP_BLUE   = "#1F3864"   # azul institucional (primario)
PUCP_GOLD   = "#D4A017"   # dorado complementario
EMERALD     = "#2E8B57"   # verde académico
PURPLE      = "#7C3AED"   # violeta acento
CORAL       = "#E07A5F"   # coral suave
TEAL        = "#2A9D8F"   # turquesa
GREY_DK     = "#1A1A1A"
GREY_MD     = "#5A5A5A"
GREY_LT     = "#CFCFCF"

mpl.rcParams.update({
    "font.family": "Liberation Sans",
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.titlecolor": PUCP_BLUE,
    "axes.labelsize": 10,
    "axes.edgecolor": GREY_DK,
    "axes.labelcolor": GREY_DK,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": GREY_DK,
    "ytick.color": GREY_DK,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "lines.linewidth": 1.7,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": GREY_LT,
    "legend.fontsize": 9.5,
    "figure.dpi": 150,
})

OUT = Path("docs/assets/g1")
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {path}")


# -----------------------------------------------------------------------------
# Figura 5 — Operación de convolución 2D (con color)
# -----------------------------------------------------------------------------
def fig05_convolution():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2),
                             gridspec_kw={'wspace': 0.30})
    from matplotlib.patches import Rectangle

    np.random.seed(7)
    img = np.random.randint(20, 200, (5, 5))

    ax = axes[0]
    ax.imshow(img, cmap="Blues", vmin=0, vmax=255)
    for (i, j), v in np.ndenumerate(img):
        ax.text(j, i, v, ha="center", va="center", fontsize=9.5,
                color=GREY_DK if img[i, j] < 130 else "white")
    ax.add_patch(Rectangle((-0.5, -0.5), 3, 3, fill=False,
                           edgecolor=PUCP_GOLD, lw=2.5))
    ax.set_title("Entrada (5 × 5)")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    ax.imshow(kernel, cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    for (i, j), v in np.ndenumerate(kernel):
        ax.text(j, i, f"{v:+d}", ha="center", va="center",
                fontsize=12, color=GREY_DK, fontweight="bold")
    ax.set_title("Kernel Sobel-x (3 × 3)")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[2]
    out = np.zeros((3, 3), dtype=int)
    for i in range(3):
        for j in range(3):
            out[i, j] = int((img[i:i+3, j:j+3] * kernel).sum())
    abs_max = abs(out).max()
    ax.imshow(out, cmap="Greens", vmin=-abs_max, vmax=abs_max)
    for (i, j), v in np.ndenumerate(out):
        ax.text(j, i, v, ha="center", va="center", fontsize=10,
                color=GREY_DK if abs(v) < abs_max*0.6 else "white")
    ax.set_title("Feature map (3 × 3)")
    ax.set_xticks([]); ax.set_yticks([])

    save(fig, "g1_fig05_convolution.png")


# -----------------------------------------------------------------------------
# Figura 7 — Max vs Average pooling (con color)
# -----------------------------------------------------------------------------
def fig07_pooling():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2),
                             gridspec_kw={'wspace': 0.30})
    from matplotlib.patches import Rectangle

    inp = np.array([[1, 3, 2, 9],
                    [5, 6, 7, 8],
                    [3, 1, 0, 4],
                    [2, 5, 3, 6]])

    ax = axes[0]
    ax.imshow(inp, cmap="Blues", vmin=0, vmax=10)
    for (i, j), v in np.ndenumerate(inp):
        ax.text(j, i, v, ha="center", va="center",
                color=GREY_DK if v < 6 else "white", fontsize=12)
    for (x, y) in [(-0.5, -0.5), (1.5, -0.5), (-0.5, 1.5), (1.5, 1.5)]:
        ax.add_patch(Rectangle((x, y), 2, 2, fill=False,
                               edgecolor=PUCP_GOLD, lw=2.0))
    ax.set_title("Entrada 4 × 4 (ventanas 2 × 2)")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    mx = np.array([[6, 9], [5, 6]])
    ax.imshow(mx, cmap="Greens", vmin=0, vmax=10)
    for (i, j), v in np.ndenumerate(mx):
        ax.text(j, i, v, ha="center", va="center",
                color=GREY_DK if v < 6 else "white",
                fontsize=15, fontweight="bold")
    ax.set_title("Max pooling")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[2]
    av = np.array([[3.75, 6.5], [2.75, 3.25]])
    ax.imshow(av, cmap="Oranges", vmin=0, vmax=10)
    for (i, j), v in np.ndenumerate(av):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color=GREY_DK if v < 5 else "white",
                fontsize=13, fontweight="bold")
    ax.set_title("Average pooling")
    ax.set_xticks([]); ax.set_yticks([])

    save(fig, "g1_fig07_pooling.png")


# -----------------------------------------------------------------------------
# Figura 8 — Funciones de activación (con color)
# -----------------------------------------------------------------------------
def fig08_activations():
    x = np.linspace(-4, 4, 400)
    relu       = np.maximum(0, x)
    leaky_relu = np.where(x > 0, x, 0.1 * x)
    sigmoid    = 1 / (1 + np.exp(-x))
    tanh       = np.tanh(x)

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.plot(x, relu,       label="ReLU",       color=PUCP_BLUE, lw=2.2)
    ax.plot(x, leaky_relu, label="Leaky ReLU", color=PUCP_GOLD, lw=2.0, linestyle="--")
    ax.plot(x, sigmoid,    label="Sigmoid",    color=EMERALD,   lw=1.8)
    ax.plot(x, tanh,       label="Tanh",       color=PURPLE,    lw=1.8, linestyle="-.")
    ax.axhline(0, color=GREY_DK, lw=0.5)
    ax.axvline(0, color=GREY_DK, lw=0.5)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.legend(loc="upper left")
    ax.set_ylim(-1.3, 4)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    save(fig, "g1_fig08_activations.png")


# -----------------------------------------------------------------------------
# Figura 10 — Underfit / fit / overfit (con color)
# -----------------------------------------------------------------------------
def fig10_curves():
    epochs = np.arange(1, 31)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)

    train1 = 0.7 - 0.15 * (1 - np.exp(-epochs / 30))
    val1   = 0.72 - 0.12 * (1 - np.exp(-epochs / 30))
    axes[0].plot(epochs, train1, color=PUCP_BLUE, lw=2.0, label="entrenamiento")
    axes[0].plot(epochs, val1,   color=PUCP_GOLD, lw=2.0, ls="--", label="validación")
    axes[0].set_title("Underfitting", color=CORAL)

    train2 = 0.7 * np.exp(-epochs / 10) + 0.05
    val2   = 0.75 * np.exp(-epochs / 11) + 0.08
    axes[1].plot(epochs, train2, color=PUCP_BLUE, lw=2.0)
    axes[1].plot(epochs, val2,   color=PUCP_GOLD, lw=2.0, ls="--")
    axes[1].set_title("Ajuste adecuado", color=EMERALD)

    train3 = 0.7 * np.exp(-epochs / 8) + 0.02
    val3   = 0.7 * np.exp(-epochs / 8) + 0.05
    val3[12:] += np.linspace(0, 0.25, len(epochs) - 12)
    axes[2].plot(epochs, train3, color=PUCP_BLUE, lw=2.0)
    axes[2].plot(epochs, val3,   color=PUCP_GOLD, lw=2.0, ls="--")
    axes[2].axvline(12, color=CORAL, lw=1.2, ls=":")
    axes[2].text(13, 0.55, "inicio del\noverfit", color=CORAL, fontsize=9)
    axes[2].set_title("Overfitting", color=CORAL)

    for ax in axes:
        ax.set_xlabel("Época")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_ylim(0, 0.85)
    axes[0].set_ylabel("Pérdida (loss)")
    axes[0].legend(loc="upper right")
    save(fig, "g1_fig10_curves.png")


# -----------------------------------------------------------------------------
# Figura 11 — Matriz de confusión (con color)
# -----------------------------------------------------------------------------
def fig11_confmat():
    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    cm = np.array([[42, 3], [5, 50]])
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    classes = ["plástico", "vidrio"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Etiqueta real")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center",
                fontsize=20, fontweight="bold",
                color="white" if v > cm.max() / 2 else PUCP_BLUE)
    ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
    ax.grid(which="minor", color=GREY_LT, linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    plt.colorbar(im, ax=ax, fraction=0.045)
    save(fig, "g1_fig11_confmat.png")


# -----------------------------------------------------------------------------
# Figura 12 — Curvas esperadas (con color)
# -----------------------------------------------------------------------------
def fig12_training_curves():
    np.random.seed(11)
    epochs = np.arange(1, 21)
    train_loss = 0.7 * np.exp(-epochs / 6) + 0.05 + np.random.normal(0, 0.012, 20)
    val_loss   = 0.75 * np.exp(-epochs / 7) + 0.10 + np.random.normal(0, 0.018, 20)
    train_acc  = 1 - train_loss * 0.9
    val_acc    = 1 - val_loss * 0.95

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ax.plot(epochs, train_loss, color=PUCP_BLUE, lw=2.0, label="entrenamiento")
    ax.plot(epochs, val_loss,   color=PUCP_GOLD, lw=2.0, ls="--", label="validación")
    ax.set_xlabel("Época"); ax.set_ylabel("Pérdida (loss)")
    ax.set_title("Pérdida")
    ax.grid(True, alpha=0.3, linewidth=0.5); ax.legend()

    ax = axes[1]
    ax.plot(epochs, train_acc, color=EMERALD,   lw=2.0, label="entrenamiento")
    ax.plot(epochs, val_acc,   color=PUCP_GOLD, lw=2.0, ls="--", label="validación")
    ax.set_xlabel("Época"); ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.grid(True, alpha=0.3, linewidth=0.5); ax.set_ylim(0, 1); ax.legend()
    save(fig, "g1_fig12_training_curves.png")


# -----------------------------------------------------------------------------
# Figura nueva — Transfer Learning con MobileNetV2
# -----------------------------------------------------------------------------
def fig13_transfer_learning():
    """Diagrama de TL: backbone pre-entrenado + head reemplazada."""
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_xlim(0, 11); ax.set_ylim(0, 3); ax.axis("off")

    from matplotlib.patches import FancyBboxPatch

    # Backbone congelado (gris)
    backbone = FancyBboxPatch((0.4, 0.6), 5.5, 1.6, boxstyle="round,pad=0.05",
                              linewidth=1.5, edgecolor=GREY_DK,
                              facecolor="#E8E8E8")
    ax.add_patch(backbone)
    ax.text(3.15, 1.7, "Backbone pre-entrenado",
            ha="center", fontsize=11, fontweight="bold", color=GREY_DK)
    ax.text(3.15, 1.25, "MobileNetV2 (ImageNet, $\\approx$1.3M imágenes)",
            ha="center", fontsize=9.5, color=GREY_MD, style="italic")
    ax.text(3.15, 0.9, "🔒 pesos congelados (feature extraction)",
            ha="center", fontsize=8.5, color=CORAL)

    # Head nueva (color PUCP)
    head = FancyBboxPatch((6.5, 0.6), 4.2, 1.6, boxstyle="round,pad=0.05",
                          linewidth=1.5, edgecolor=PUCP_BLUE,
                          facecolor=PUCP_BLUE)
    ax.add_patch(head)
    ax.text(8.6, 1.7, "Cabezal nuevo",
            ha="center", fontsize=11, fontweight="bold", color="white")
    ax.text(8.6, 1.25, "GlobalAveragePool + Dense(3)",
            ha="center", fontsize=9.5, color="white")
    ax.text(8.6, 0.9, "entrenable sobre nuestro dataset",
            ha="center", fontsize=8.5, color=PUCP_GOLD)

    # Flecha
    ax.annotate("", xy=(6.4, 1.4), xytext=(5.95, 1.4),
                arrowprops=dict(arrowstyle="->", color=GREY_DK, lw=1.8))

    # Etiquetas top
    ax.text(3.15, 2.55, "Conocimiento transferido",
            ha="center", fontsize=10, color=PUCP_BLUE, style="italic")
    ax.text(8.6, 2.55, "Especialización al problema",
            ha="center", fontsize=10, color=PUCP_BLUE, style="italic")

    save(fig, "g1_fig13_transfer_learning.png")


# -----------------------------------------------------------------------------
# Limpieza
# -----------------------------------------------------------------------------
def limpiar_antiguos():
    for nombre in ("g1_fig01_pipeline.png", "g1_fig02_classical.png",
                   "g1_fig03_cnn_pipeline.png", "g1_fig04_cnn_arch.png",
                   "g1_fig06_stride_padding.png", "g1_fig09_training_loop.png"):
        p = OUT / nombre
        if p.exists():
            p.unlink()


if __name__ == "__main__":
    print(f"Regenerando figuras en {OUT}/...\n")
    limpiar_antiguos()
    fig05_convolution()
    fig07_pooling()
    fig08_activations()
    fig10_curves()
    fig11_confmat()
    fig12_training_curves()
    fig13_transfer_learning()
    print(f"\n✔ {len(list(OUT.glob('*.png')))} figuras en {OUT}/")
