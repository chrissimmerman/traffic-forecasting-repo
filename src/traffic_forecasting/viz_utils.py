from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go


def save_plotly(fig: go.Figure, html_out: Path, png_out: Path | None = None, width: int = 1500, height: int = 520) -> None:
    html_out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        html_out,
        include_plotlyjs='cdn',
        config={'responsive': True, 'displayModeBar': False},
    )

    if png_out is not None:
        png_out.parent.mkdir(parents=True, exist_ok=True)
        # Requires `kaleido`
        fig.write_image(png_out, width=width, height=height, scale=2)
