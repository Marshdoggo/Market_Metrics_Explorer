import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


def _size_rank_values(df: pd.DataFrame, size_metric: str) -> pd.Series:
    values = pd.to_numeric(df[size_metric], errors="coerce")
    if "Drawdown" in size_metric:
        values = values.abs()
    return values.rank(pct=True)


def scatter_xy(
    df: pd.DataFrame,
    x: str,
    y: str,
    color_by: str='Sector',
    title: str='',
    size_metric: str | None = None,
):
    fig, ax = plt.subplots()
    sizes = None
    if size_metric and size_metric in df.columns:
        ranks = _size_rank_values(df, size_metric)
        sizes = ranks.fillna(0.15).mul(220).add(25)
    if color_by in df.columns:
        groups = df.groupby(color_by)
        for name, grp in groups:
            group_sizes = sizes.loc[grp.index] if sizes is not None else None
            ax.scatter(grp[x], grp[y], s=group_sizes, label=name, alpha=0.8, edgecolors='none')
        ax.legend(fontsize=8, ncols=2)
    else:
        ax.scatter(df[x], df[y], s=sizes, alpha=0.8, edgecolors='none')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title or f"{y} vs {x}")
    ax.grid(True, alpha=0.3)
    return fig, ax


def _marker_sizes(df: pd.DataFrame, size_metric: str | None) -> pd.Series | None:
    if not size_metric or size_metric not in df.columns:
        return None
    ranks = _size_rank_values(df, size_metric)
    return ranks.fillna(0.0).mul(38).add(7)


def scatter_xy_interactive(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = '',
    color_by: str | None = 'Sector',
    size_metric: str | None = None,
):
    """
    Interactive scatter with hover labels.
    Expects df to include columns: x, y, and ideally 'Ticker','Name','Sector','SubIndustry'.
    """
    plot_df = df.copy()
    marker_size_col = None
    if size_metric and size_metric in plot_df.columns:
        marker_size_col = "_MarkerSize"
        plot_df[marker_size_col] = _marker_sizes(plot_df, size_metric)

    # Build hover_data dynamically if columns exist
    hover_data = {}
    for col in ['Ticker', 'Name', 'Sector', 'SubIndustry']:
        if col in plot_df.columns:
            hover_data[col] = True
    if size_metric and size_metric in plot_df.columns:
        hover_data[size_metric] = True
    if color_by and color_by in plot_df.columns and color_by not in hover_data:
        hover_data[color_by] = True
    if marker_size_col:
        hover_data[marker_size_col] = False

    fig = px.scatter(
        plot_df,
        x=x,
        y=y,
        color=color_by if color_by in plot_df.columns else None,
        size=marker_size_col,
        size_max=45,
        hover_name='Ticker' if 'Ticker' in plot_df.columns else None,
        hover_data=hover_data if hover_data else None,
        title=title or f"{y} vs {x}",
    )
    if marker_size_col:
        fig.update_traces(marker=dict(opacity=0.85))
    else:
        fig.update_traces(marker=dict(size=8, opacity=0.85))
    fig.update_layout(legend_title_text=color_by or '')
    return fig
