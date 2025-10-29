import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

def scatter_xy(df: pd.DataFrame, x: str, y: str, color_by: str='Sector', title: str=''):
    fig, ax = plt.subplots()
    if color_by in df.columns:
        groups = df.groupby(color_by)
        for name, grp in groups:
            ax.scatter(grp[x], grp[y], label=name, alpha=0.8, edgecolors='none')
        ax.legend(fontsize=8, ncols=2)
    else:
        ax.scatter(df[x], df[y], alpha=0.8, edgecolors='none')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title or f"{y} vs {x}")
    ax.grid(True, alpha=0.3)
    return fig, ax
def scatter_xy_interactive(df: pd.DataFrame, x: str, y: str, title: str = ''):
    """
    Interactive scatter with hover labels.
    Expects df to include columns: x, y, and ideally 'Ticker','Name','Sector','SubIndustry'.
    """
    # Build hover_data dynamically if columns exist
    hover_data = {}
    for col in ['Ticker', 'Name', 'Sector', 'SubIndustry']:
        if col in df.columns:
            hover_data[col] = True

    fig = px.scatter(
        df,
        x=x,
        y=y,
        color='Sector' if 'Sector' in df.columns else None,
        hover_name='Ticker' if 'Ticker' in df.columns else None,
        hover_data=hover_data if hover_data else None,
        title=title or f"{y} vs {x}",
    )
    fig.update_traces(marker=dict(size=8, opacity=0.85))
    fig.update_layout(legend_title_text='Sector')
    return fig