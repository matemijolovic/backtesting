"""Plotly figures for the UI. One y-axis per chart, thin marks, muted chrome.

Colours are the validated reference palette (categorical slots in fixed
order; diverging blue <-> red around a gray midpoint). Charts use
transparent backgrounds so Streamlit's light/dark theme shows through.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
REFERENCE = "#898781"  # benchmark / context lines
NEGATIVE = "#e34948"
GRID = "rgba(137,135,129,0.18)"
AXIS = "rgba(137,135,129,0.45)"
DIVERGING = [
    [0.0, "#b3261e"],
    [0.25, "#e66767"],
    [0.5, "#d9d8d2"],
    [0.75, "#5598e7"],
    [1.0, "#1c5cab"],
]
MAX_POINTS = 4000


def _thin(series: pd.Series, n: int = MAX_POINTS) -> pd.Series:
    """Downsample for plotting; keep the extremes of each chunk visible."""
    if len(series) <= n:
        return series
    step = int(np.ceil(len(series) / n))
    return series.iloc[::step]


def _layout(fig: go.Figure, *, height: int = 360, y_title: str = "", y_format: str | None = None, log: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, ticks="outside", tickcolor=AXIS)
    fig.update_yaxes(
        gridcolor=GRID,
        zeroline=False,
        linecolor=AXIS,
        title_text=y_title,
        tickformat=y_format,
        type="log" if log else "-",  # "-" = auto, so categorical heatmap rows survive
    )
    return fig


def equity(result, *, log: bool = False) -> go.Figure:
    fig = go.Figure()
    bench = _thin(result.benchmark)
    eq = _thin(result.equity)
    fig.add_trace(
        go.Scatter(x=bench.index, y=bench.values, name="Buy & hold", line=dict(color=REFERENCE, width=1.5, dash="dot"),
                   hovertemplate="%{y:,.0f}")
    )
    fig.add_trace(
        go.Scatter(x=eq.index, y=eq.values, name=result.config.strategy, line=dict(color=SERIES[0], width=2),
                   hovertemplate="%{y:,.0f}")
    )
    return _layout(fig, height=380, y_title="Equity (USDT)", y_format=",.0f", log=log)


def drawdown(result) -> go.Figure:
    from backtester.metrics import drawdown as dd_fn

    fig = go.Figure()
    bench = _thin(dd_fn(result.benchmark))
    dd = _thin(dd_fn(result.equity))
    fig.add_trace(go.Scatter(x=bench.index, y=bench.values, name="Buy & hold", line=dict(color=REFERENCE, width=1.5, dash="dot"),
                             hovertemplate="%{y:.1%}"))
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name=result.config.strategy, fill="tozeroy",
                             line=dict(color=NEGATIVE, width=1.5), fillcolor="rgba(227,73,72,0.15)",
                             hovertemplate="%{y:.1%}"))
    return _layout(fig, height=240, y_title="Drawdown", y_format=".0%")


def exposure(result) -> go.Figure:
    exp = _thin(result.exposure)
    fig = go.Figure(
        go.Scatter(x=exp.index, y=exp.values, name="Net exposure", line=dict(color=SERIES[0], width=1.5, shape="hv"),
                   fill="tozeroy", fillcolor="rgba(42,120,214,0.12)", hovertemplate="%{y:.2f}x")
    )
    fig.update_layout(showlegend=False)
    return _layout(fig, height=200, y_title="Exposure (x equity)", y_format=".1f")


def price_with_trades(result) -> go.Figure:
    close = _thin(result.close)
    fig = go.Figure(
        go.Scatter(x=close.index, y=close.values, name="Close", line=dict(color=REFERENCE, width=1.25),
                   hovertemplate="%{y:,.2f}")
    )
    fills = result.fills
    if fills is not None and not fills.empty and "ts_last" in fills:
        ts = pd.to_datetime(fills["ts_last"], utc=True)
        px = pd.to_numeric(fills["avg_px"], errors="coerce")
        side = fills["side"].astype(str)
        for label, mask, color, symbol in (
            ("Buy", side == "BUY", SERIES[2], "triangle-up"),
            ("Sell", side == "SELL", SERIES[7], "triangle-down"),
        ):
            if mask.any():
                fig.add_trace(
                    go.Scatter(x=ts[mask], y=px[mask], mode="markers", name=label,
                               marker=dict(color=color, size=8, symbol=symbol, opacity=0.75 if mask.sum() > 200 else 1.0,
                                           line=dict(width=1, color="rgba(255,255,255,0.8)")),
                               hovertemplate=f"{label} %{{y:,.2f}}<extra></extra>")
                )
    return _layout(fig, height=340, y_title="Price", y_format=",.2f")


def monthly_heatmap(table: pd.DataFrame) -> go.Figure:
    if table.empty:
        return go.Figure()
    values = table.to_numpy(dtype=float)
    lim = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
    text = np.where(np.isfinite(values), np.vectorize(lambda v: f"{v:+.1%}")(np.nan_to_num(values)), "")
    fig = go.Figure(
        go.Heatmap(
            z=values, x=list(table.columns), y=[str(y) for y in table.index], colorscale=DIVERGING,
            zmid=0, zmin=-lim, zmax=lim, text=text, texttemplate="%{text}", textfont=dict(size=11),
            hovertemplate="%{y} %{x}: %{z:+.2%}<extra></extra>", xgap=2, ygap=2, showscale=False,
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig = _layout(fig, height=60 + 34 * len(table))
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", type="category")
    return fig


def grid_heatmap(frame: pd.DataFrame, value: str = "sharpe", fmt: str = "+.2f") -> go.Figure:
    """Rows = symbol, cols = interval (ordered by bar length), cells = median ``value``."""
    from backtester.data import interval_minutes

    pivot = frame.pivot_table(index="symbol", columns="interval", values=value, aggfunc="median")
    pivot = pivot[sorted(pivot.columns, key=interval_minutes)]
    values = pivot.to_numpy(dtype=float)
    lim = max(np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0, 1e-9)
    text = np.where(np.isfinite(values), np.vectorize(lambda v: format(v, fmt))(np.nan_to_num(values)), "n/a")
    fig = go.Figure(
        go.Heatmap(
            z=values, x=list(pivot.columns), y=list(pivot.index), colorscale=DIVERGING, zmid=0, zmin=-lim, zmax=lim,
            text=text, texttemplate="%{text}", textfont=dict(size=13), xgap=2, ygap=2,
            hovertemplate="%{y} · %{x}: %{z:" + fmt + "}<extra></extra>", showscale=False,
        )
    )
    fig = _layout(fig, height=80 + 46 * len(pivot))
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", autorange="reversed", type="category")
    fig.update_xaxes(type="category")
    return fig


def sharpe_strip(frame: pd.DataFrame, by: str = "interval") -> go.Figure:
    """Every run's Sharpe as a dot, grouped by ``by`` (dispersion view)."""
    from backtester.data import interval_minutes

    fig = go.Figure()
    groups = list(dict.fromkeys(frame[by]))
    if by == "interval":
        groups = sorted(groups, key=interval_minutes)
    fig.add_trace(
        go.Box(
            x=frame[by], y=frame["sharpe"], boxpoints="all", jitter=0.4, pointpos=0, marker=dict(color=SERIES[0], size=8, opacity=0.8),
            line=dict(color=REFERENCE, width=1), fillcolor="rgba(0,0,0,0)", name="Sharpe",
            customdata=np.stack([frame["symbol"], frame["interval"], frame["offset"]], axis=1),
            hovertemplate="%{customdata[0]} %{customdata[1]} %{customdata[2]}<br>Sharpe %{y:.2f}<extra></extra>",
        )
    )
    bh = frame.groupby(by)["bh_sharpe"].median()
    fig.add_trace(go.Scatter(x=bh.index, y=bh.values, mode="markers", name="Buy & hold (median)",
                             marker=dict(color=REFERENCE, size=12, symbol="line-ew-open", line=dict(width=2, color=REFERENCE)),
                             hovertemplate="Buy & hold %{y:.2f}<extra></extra>"))
    fig.update_xaxes(categoryorder="array", categoryarray=groups, type="category")
    fig.add_hline(y=0, line=dict(color=AXIS, width=1))
    fig = _layout(fig, height=320, y_title="Sharpe (daily, annualised)", y_format=".1f")
    fig.update_layout(hovermode="closest")
    return fig


def curves(named: dict[str, pd.Series], *, normalize: bool = True, highlight: str | None = None,
           reference: pd.Series | None = None, log: bool = False) -> go.Figure:
    """Overlay of equity curves. Up to 8 get categorical colours; more are drawn as a gray fan."""
    fig = go.Figure()
    many = len(named) > len(SERIES)
    for i, (name, series) in enumerate(named.items()):
        if series is None or len(series) < 2:
            continue
        s = _thin(series / series.iloc[0] if normalize else series)
        if many:
            line = dict(color="rgba(137,135,129,0.35)", width=1)
            show = False
        else:
            line = dict(color=SERIES[i % len(SERIES)], width=2)
            show = True
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name, line=line, showlegend=show,
                                 hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>" if many else "%{y:.2f}"))
    if reference is not None and len(reference) > 1:
        r = _thin(reference / reference.iloc[0] if normalize else reference)
        fig.add_trace(go.Scatter(x=r.index, y=r.values, name=highlight or "Portfolio", line=dict(color=SERIES[0], width=2.5)))
    fig = _layout(fig, height=380, y_title="Growth of 1" if normalize else "Equity", y_format=".2f", log=log)
    if many:
        fig.update_layout(hovermode="closest")
    return fig


def candles(bars: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Candlestick(
            x=bars.index, open=bars["open"], high=bars["high"], low=bars["low"], close=bars["close"],
            increasing=dict(line=dict(color=SERIES[2], width=1)), decreasing=dict(line=dict(color=SERIES[7], width=1)),
            name="OHLC",
        )
    )
    fig.update_layout(xaxis_rangeslider_visible=False, showlegend=False)
    return _layout(fig, height=380, y_title="Price", y_format=",.2f")
