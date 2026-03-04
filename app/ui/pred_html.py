"""Renderizado HTML de barras de predicción con animaciones CSS."""


def render_prediction(preds_dict, animate=True, live=False):
    """Genera HTML de barras de predicción.

    Args:
        preds_dict: dict {class_name: confidence} o None.
        animate: True para animar barras (click individual).
        live: True para mostrar indicador LIVE (streaming).
    """
    if preds_dict is None:
        if live:
            return '<div class="pred-empty">Activa la cámara para ver predicciones</div>'
        return '<div class="pred-empty">Sube una imagen y haz clic en Predecir</div>'

    items = sorted(preds_dict.items(), key=lambda x: x[1], reverse=True)
    if not items:
        return '<div class="pred-empty">Sin resultados</div>'

    top_name, top_conf = items[0]
    grow = " grow" if animate else ""

    html = '<div class="pred-result">'

    # Winner
    html += '<div class="pred-winner">'
    if live:
        html += '<span class="live-dot"></span>'
    html += f'<span class="pred-winner-name">{top_name}</span>'
    html += f'<span class="pred-winner-conf">{top_conf:.1%}</span>'
    html += '</div>'

    # Bars
    for i, (name, conf) in enumerate(items[:5]):
        pct = max(conf * 100, 0.5)
        fill_cls = "top" if i == 0 else "other"
        html += (
            f'<div class="pred-row">'
            f'<div class="pred-name">{name}</div>'
            f'<div class="pred-track">'
            f'<div class="pred-fill {fill_cls}{grow}" style="width:{pct:.1f}%"></div>'
            f'</div>'
            f'<div class="pred-pct">{pct:.1f}%</div>'
            f'</div>'
        )

    html += '</div>'
    return html


def render_error(msg):
    """Muestra un mensaje de error en el panel de predicción."""
    return f'<div class="pred-error">{msg}</div>'
