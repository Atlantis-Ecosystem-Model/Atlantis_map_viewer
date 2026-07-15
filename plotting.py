"""Static (matplotlib) map / profile / time-series plots for Atlantis output,
for scripting and notebook use. See app.py for the interactive viewer."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ncdata import AtlantisOutput


def plot_map(
    ao: AtlantisOutput,
    varname: str,
    t_index: int = 0,
    layer_index: int = 0,
    layer_type: str = "wc",
    ax=None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    show_labels: bool = True,
):
    if ao.bgm is None:
        raise RuntimeError("No .bgm geometry available; cannot draw a map")
    gdf = ao.bgm.to_geodataframe()
    values = ao.map_values(varname, t_index, layer_index, layer_type)
    gdf = gdf.copy()
    gdf["value"] = values

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    gdf.plot(column="value", cmap=cmap, vmin=vmin, vmax=vmax, edgecolor="black",
              linewidth=0.6, legend=True, ax=ax)

    if show_labels:
        for _, row in gdf.iterrows():
            c = row.geometry.centroid
            ax.annotate(str(row["label"]), (c.x, c.y), ha="center", va="center", fontsize=7)

    meta = ao.var_meta(varname)
    title = f"{meta['long_name']} ({meta['units']})\n{ao.date_label(t_index)}"
    if ao.is_layered(varname):
        title += f" | {layer_type} layer {layer_index} [{ao.depth_label(t_index, 0, layer_index, layer_type)}]"
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    return ax


def plot_profile(
    ao: AtlantisOutput,
    varname: str,
    t_index: int,
    box_index: int,
    layer_type: str = "wc",
    ax=None,
):
    mids, values, bounds = ao.profile(varname, t_index, box_index, layer_type)
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 6))

    # step plot: value held constant across each layer's depth range
    y = np.repeat(bounds, 1)
    step_y = np.empty(2 * len(values))
    step_x = np.empty(2 * len(values))
    step_y[0::2] = bounds[:-1]
    step_y[1::2] = bounds[1:]
    step_x[0::2] = values
    step_x[1::2] = values
    ax.plot(step_x, step_y, marker="o", markersize=3)

    meta = ao.var_meta(varname)
    ax.set_xlabel(f"{meta['long_name']} ({meta['units']})")
    ax.set_ylabel("Depth (m)")
    label = ao.box_labels[box_index]
    ax.set_title(f"{label} | {ao.date_label(t_index)}")
    return ax


def plot_timeseries(
    ao: AtlantisOutput,
    varname: str,
    box_index: int,
    layer_index: int | None = 0,
    layer_type: str = "wc",
    ax=None,
):
    times, values = ao.timeseries(varname, box_index, layer_index, layer_type)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, values)
    meta = ao.var_meta(varname)
    ax.set_ylabel(f"{meta['long_name']} ({meta['units']})")
    ax.set_xlabel("Time")
    label = ao.box_labels[box_index]
    title = f"{label}"
    if ao.is_layered(varname):
        title += f" | {layer_type} layer {layer_index}"
    ax.set_title(title)
    return ax
