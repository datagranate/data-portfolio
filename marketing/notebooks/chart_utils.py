import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import to_hex
from typing import List, Optional, Tuple, Union

# --- Configuration and defaults ---
POMEGRANATE_PALETTE = [
    "#C41E3A", # Primary Red
    "#004E7C", # Ocean
    "#2A9D8F", # Teal 
    "#DAA520", # Gold
    "#596275", # Slate Grey
    "#8B0000", # Garnet
    "#FA8072", # Coral
    "#E9C46A", # Yellow
    "#E8D8C0", # Pale Sand
    "#264653"  # Electric Blue
]

PLOT_CONFIG = {
    "font_size": 10,
    "bar_height": 0.5,
    "edge_color": "none",
    "text_format": "{:.1f}%",
    "label_threshold": 3.0,
    "text_min_val": 8.0, # Threshold for white text
    "spine_left": False, # keep left spine
    "spine_top": True,
    "spine_right": True,
    "spine_bottom": True,
    "legend_loc": "center left",
    "legend_anchor": (1.0, 0.5),
    "legend_frameon": False,
}

def _get_text_color(rgb: tuple) -> str:
    # calculates luminance of an RGB color and returns 'white' or 'black' based on contrast requirements
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return "white" if luminance < 0.5 else "black"

import matplotlib.colors as mcolors

def _generate_color_map(segments: List[str], cmap: str) -> dict:
    # Generates a consistent color map for a list of segments
    # Converts hex codes to RGB tuples for compatibility with text color logic.
    if cmap == "pomegranate":
        palette_hex = POMEGRANATE_PALETTE
    else:
        # Fallback to standard Seaborn if named cmap
        palette_hex = sns.color_palette(cmap, n_colors=len(segments)).as_hex()
    
    # Convert Hex -> RGB yuple (0-1 range)
    color_map = {}
    for i, seg in enumerate(segments):
        hex_color = palette_hex[i % len(palette_hex)]
        # mcolors.to_rgb returns a tuple of floats (0.0 to 1.0)
        rgb_tuple = mcolors.to_rgb(hex_color)
        color_map[seg] = rgb_tuple
        
    return color_map

def set_pomegranate_theme():
    palette_list = [
        "#C41E3A", "#004E7C", "#2A9D8F", "#DAA520", "#596275",
        "#8B0000", "#FA8072", "#E9C46A", "#E8D8C0", "#264653"
    ]
    sns.set_palette(palette_list)
    sns.set_theme(style="whitegrid")
    
    # force the palette again just to be safe (sometimes set_theme resets it)
    sns.set_palette(palette_list)
    
    return "Pomegranate theme applied."

def _draw_segment_and_label(
    ax: plt.Axes, 
    y_pos: float, 
    left: float, 
    val: float, 
    color: tuple, 
    config: dict
) -> float:
    # Draws a single bar segment and optional text label, returns the updated 'left' position for the next segment
    
    # Draw bar
    ax.barh(
        [y_pos], 
        [val], 
        left=left, 
        height=config["bar_height"], 
        color=color,
        edgecolor=config["edge_color"]
    )

    # Draw label if threshold met
    if val >= config["label_threshold"]:
        x = left + val / 2
        text_color = _get_text_color(color)
        ax.text(
            x, y_pos, 
            config["text_format"].format(val),
            va="center", ha="center",
            fontsize=config["font_size"],
            color=text_color
        )
    return left + val

def _setup_axes(
    ax: plt.Axes, 
    y_ticks: List, 
    y_labels: List[str], 
    xlabel: str, 
    ylabel: str, 
    title: Optional[str] = None,
    config: dict = None
):
    # Common setup for axes, limits, and despine
    if config is None:
        config = PLOT_CONFIG
    
    ax.set_xlim(0, 100)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(l) for l in y_labels])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if title:
        ax.set_title(title)

    sns.despine(
        ax=ax, 
        left=config["spine_left"], 
        top=config["spine_top"], 
        right=config["spine_right"], 
        bottom=config["spine_bottom"]
    )

def _create_legend(
    ax: plt.Axes, 
    segments: List[str], 
    color_map: dict
):
    
    # create a shared legend for plot
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[s]) for s in segments]
    ax.legend(
        handles, 
        segments, 
        loc=PLOT_CONFIG["legend_loc"], 
        bbox_to_anchor=PLOT_CONFIG["legend_anchor"], 
        frameon=PLOT_CONFIG["legend_frameon"]
    )

# --- Public Functions ---

def plot_100pct_stacked_hbar(
    df: pd.DataFrame, 
    col_list: List[str],
    cmap: str = "pomegranate",
    label_min_pct: float = 3.0,
    figsize: Tuple[int, int] = (12, 2.8),
    reverse_segments: bool = False,
    title: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    # Single 100% stacked horizontal bar representing total distribution of sums of columns
    # Returns: fig, ax
    
    # Update config if needed
    config = PLOT_CONFIG.copy()
    config["label_threshold"] = label_min_pct

    # Prep data
    overall = df[col_list].sum(axis=0)
    pct = (overall / overall.sum()) * 100
    
    if reverse_segments:
        pct = pct.sort_values(ascending=True)
    else:
        pct = pct.sort_values(ascending=False)

    segments = list(pct.index)
    color_map = _generate_color_map(segments, cmap)

    # plotting
    fig, ax = plt.subplots(figsize=figsize)
    y_pos = 0
    left = 0

    for seg in segments:
        val = pct[seg]
        color = color_map[seg]
        
        left = _draw_segment_and_label(
            ax, y_pos, left, val, color, config
        )

    # set up axes, legend, finalise
    _setup_axes(
        ax, 
        y_ticks=[y_pos], 
        y_labels=[""], 
        xlabel="Percent of total (%)", 
        ylabel="", 
        title=title or "Distribution (100% stacked)"
    )
    
    _create_legend(ax, segments, color_map)
    
    plt.tight_layout()
    return fig, ax

def plot_100pct_stacked_hbar_by_group(
    df: pd.DataFrame,
    segment_cols: List[str],
    group_col: str,
    cmap: str = "pomegranate",
    label_min_pct: float = 3.0,
    figsize: Tuple[int, int] = (12, 3.2),
    group_order: Optional[List] = None,
    reverse_segments: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    # Single 100% stacked horizontal bar representing total distribution of sums of columns, grouped
    # Returns: fig, ax
    
    config = PLOT_CONFIG.copy()
    config["label_threshold"] = label_min_pct

    # prep data
    group_sum = df.groupby(group_col)[list(segment_cols)].sum()
    pct = group_sum.div(group_sum.sum(axis=1), axis=0) * 100

    if group_order is None:
        group_order = list(pct.index)
    else:
        group_order = list(group_order)

    # determine segment order
    pct_order = pct.sum(axis=0).sort_values(ascending=False)
    segments = list(pct_order.index)
    if reverse_segments:
        segments = segments[::-1]

    color_map = _generate_color_map(segments, cmap)
    y_positions = list(range(len(group_order)))

    # plot
    fig, ax = plt.subplots(figsize=figsize)

    for yi, grp in zip(y_positions, group_order):
        left = 0
        for seg in segments:
            val = float(pct.loc[grp, seg]) if seg in pct.columns else 0.0
            color = color_map.get(seg)
            
            if color:
                left = _draw_segment_and_label(
                    ax, yi, left, val, color, config
                )

    # set up axes, legend, finalise
    _setup_axes(
        ax, 
        y_ticks=y_positions, 
        y_labels=[str(g) for g in group_order], 
        xlabel="Percent of total (%)", 
        ylabel=group_col, 
        title=f"Distribution (100% stacked) split by {group_col}"
    )

    _create_legend(ax, segments, color_map)
    
    plt.tight_layout()
    return fig, ax

def plot_distribution_bar(
    df: pd.DataFrame,
    col: str,
    cmap: str = "pomegranate",
    label_min_pct: float = 3.0,
    figsize: Tuple[int, int] = (12, 2.8),
    reverse_segments: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    # Single 100% stacked horizontal bar chart showing the distribution of unique values in a single column
    # Validates that there are <= 10 unique values (raises ValueError if not)
    # Returns fig, ax
       
    # Validate column existence
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame.")
    
    # Count occurrences
    counts = df[col].value_counts()
    unique_count = len(counts)
    
    # Validation: Max 10 unique values
    if unique_count > 10:
        raise ValueError(
            f"Column '{col}' has {unique_count} unique values. "
            "This chart type supports a maximum of 10 unique values to avoid clutter. "
            "Consider grouping categories or using a different visualisation."
        )
    
    if unique_count == 0:
        raise ValueError(f"Column '{col}' contains no data (empty or all NaN).")

    # prep data
    pct = (counts / counts.sum()) * 100
    
    # Sort segments
    if reverse_segments:
        pct = pct.sort_values(ascending=True)
    else:
        pct = pct.sort_values(ascending=False)

    segments = list(pct.index)
    color_map = _generate_color_map(segments, cmap)

    # update config if needed
    config = PLOT_CONFIG.copy()
    config["label_threshold"] = label_min_pct

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    y_pos = 0
    left = 0

    for seg in segments:
        val = pct[seg]
        color = color_map[seg]
        
        left = _draw_segment_and_label(
            ax, y_pos, left, val, color, config
        )

    # set up axes, legend, finalise
    _setup_axes(
        ax, 
        y_ticks=[y_pos], 
        y_labels=[""], 
        xlabel="Percent of total (%)", 
        ylabel="", 
        title=f"Distribution of '{col}'"
    )
    
    _create_legend(ax, segments, color_map)
    
    plt.tight_layout()
    return fig, ax

# Show the plot
def show_plot(fig, ax):
    plt.show()
    return fig, ax