"""
PUCP — Sistema de Clasificación Industrial

Entry point: python -m app.main
"""

import gradio as gr

from app.ui import tab_datos, tab_probar, tab_hmi

CSS = """
footer { display: none !important; }

.gradio-container {
    background: #f5f5f5 !important;
    max-width: 960px !important;
}

/* Tarjetas blancas */
.card {
    background: white !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    border: none !important;
    padding: 1rem 1.2rem !important;
}

/* Botón de captura */
.btn-record {
    min-height: 44px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}

/* Botones pequeños discretos (eliminar) */
.btn-sm {
    font-size: 0.8rem !important;
    padding: 0.3rem 0.8rem !important;
    min-height: 32px !important;
    color: #d93025 !important;
    background: transparent !important;
    border: 1px solid #dadce0 !important;
}
.btn-sm:hover {
    background: #fce8e6 !important;
}

/* Galería compacta */
.tm-gallery {
    min-height: 100px;
}

/* ══════════════════════════════════
   Barras de predicción animadas
   ══════════════════════════════════ */

.pred-result {
    padding: 0.5rem 0;
}

.pred-winner {
    text-align: center;
    margin-bottom: 0.8rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid #f0f0f0;
}
.pred-winner-name {
    display: block;
    font-size: 1.4rem;
    font-weight: 700;
    color: #1a73e8;
}
.pred-winner-conf {
    font-size: 2rem;
    font-weight: 800;
    color: #1a73e8;
}

.pred-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 5px;
}
.pred-name {
    min-width: 70px;
    font-size: 0.82rem;
    text-align: right;
    color: #444;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.pred-track {
    flex: 1;
    height: 22px;
    background: #f0f0f0;
    border-radius: 6px;
    overflow: hidden;
}
.pred-fill {
    height: 100%;
    border-radius: 6px;
    min-width: 2px;
}
.pred-fill.top {
    background: linear-gradient(90deg, #1a73e8, #4285f4);
}
.pred-fill.other {
    background: #c8ddf8;
}
.pred-fill.grow {
    animation: barGrow 0.5s cubic-bezier(0.22, 1, 0.36, 1);
    transform-origin: left;
}
@keyframes barGrow {
    from { transform: scaleX(0); }
    to { transform: scaleX(1); }
}
.pred-pct {
    min-width: 45px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #555;
    text-align: right;
}
.pred-empty {
    color: #999;
    padding: 2rem 1rem;
    text-align: center;
    font-size: 0.9rem;
}
.pred-error {
    color: #d93025;
    padding: 1.5rem 1rem;
    text-align: center;
    font-size: 0.9rem;
    background: #fce8e6;
    border-radius: 8px;
}

/* Indicador LIVE */
.live-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: #d93025;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
    animation: livePulse 1s ease-in-out infinite;
}
@keyframes livePulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* Contador de captura */
.capture-count {
    font-size: 2rem;
    font-weight: 800;
    color: #1a73e8;
    text-align: center;
    animation: countPop 0.2s ease-out;
}
@keyframes countPop {
    from { transform: scale(1.2); }
    to { transform: scale(1); }
}
"""

THEME = gr.themes.Default(
    primary_hue="blue",
    neutral_hue="gray",
    radius_size="lg",
    text_size="md",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)


def create_app():
    with gr.Blocks(title="PUCP - Clasificación") as app:

        gr.Markdown("## PUCP — Clasificación Industrial")

        with gr.Tabs():
            with gr.Tab("1. Datos"):
                tab_datos.create()
            with gr.Tab("2. Probar"):
                tab_probar.create()
            with gr.Tab("3. HMI"):
                tab_hmi.create()

    return app


def main():
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        theme=THEME,
        css=CSS,
    )


if __name__ == "__main__":
    main()
