"""
Interactive viewer for Atlantis box-model netCDF output + .bgm geometry.

A Python/Dash replica of the relevant parts of CSIRO's "Olive" Java tool
(cmr.apps.olive.*) for this data type: a shaded box Map view (one value per
box at a chosen time + depth layer), a vertical Profile view for a chosen box
(using that box's own non-uniform layer thicknesses), and a Time Series view
for a chosen box + layer across the whole run. Olive has no "Section"
(transect) view for box-model geometry and no vector/current view for
tracers -- box-model data is not a regular grid -- so neither is replicated
here (see the design notes for a summary of what was checked in the original
tool).

Usage:
    python3 app.py --nc /path/to/outputSETAS.nc [--bgm /path/to/VMPA_setas.bgm]
                    [--port 8050]

If --bgm is omitted, the tool looks for the file named in the netCDF
"geometry" global attribute next to the netCDF file (or its parent
directory), matching Olive's own behaviour of resolving the geometry file
relative to the run's netCDF output.
"""
from __future__ import annotations

import argparse

import numpy as np
import plotly.colors
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, no_update

from ncdata import AtlantisOutput

COLORSCALE = "Viridis"
PROFILE_HEIGHT = 560  # profile plot / vertical layer slider are sized to match
FONT_FAMILY = '"Arial Narrow", Helvetica, Arial, sans-serif'

# Plot chrome colours, matched to the panel/card theme in assets/style.css
# (modelled on MT-Hack/frontend's muted blue-grey design language) so the
# Plotly figures read as part of the same interface rather than a bolt-on.
TEXT_COLOR = "#1c2733"
HEADING_COLOR = "#4a5b6b"
GRID_COLOR = "#eef2f5"
LINE_COLOR = "#2c6e9b"


def _base_layout(title, height):
    return dict(
        title=dict(text=title, font=dict(size=14, color=HEADING_COLOR)),
        height=height,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family=FONT_FAMILY, color=TEXT_COLOR),
        hoverlabel=dict(font=dict(family=FONT_FAMILY)),
    )


def build_variable_options(ao: AtlantisOutput):
    names = sorted(ao.variables(), key=lambda n: (ao.var_meta(n)["bmtype"], n))
    options = []
    for n in names:
        meta = ao.var_meta(n)
        label = f"[{meta['bmtype']}] {n} - {meta['long_name']}"
        options.append({"label": label, "value": n})
    return options


def build_time_marks(ao: AtlantisOutput):
    n = ao.ntime
    step = max(1, n // 8)
    return {i: "" for i in range(0, n, step)}


def polygon_xy(geom):
    x, y = geom.exterior.xy
    return list(x), list(y)


def make_map_figure(ao: AtlantisOutput, varname, t_index, layer_index, layer_type, selected_box,
                     vmin=None, vmax=None):
    gdf = ao.bgm.to_geodataframe()
    values = ao.map_values(varname, t_index, layer_index, layer_type)
    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))
    if vmin == vmax:
        vmax = vmin + 1e-9
    # clip to [0, 1]: values outside a user-set fixed range are drawn at the
    # nearest end colour, rather than erroring or extrapolating.
    norm = [min(max((v - vmin) / (vmax - vmin), 0.0), 1.0) for v in values]
    colors = plotly.colors.sample_colorscale(COLORSCALE, norm)

    fig = go.Figure()
    for i, row in enumerate(gdf.itertuples()):
        xs, ys = polygon_xy(row.geometry)
        line_color = "red" if row.box == selected_box else "black"
        line_width = 3 if row.box == selected_box else 1
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                fill="toself",
                fillcolor=colors[i],
                line=dict(color=line_color, width=line_width),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    cx = [row.geometry.centroid.x for row in gdf.itertuples()]
    cy = [row.geometry.centroid.y for row in gdf.itertuples()]
    hover_text = [
        f"{gdf.iloc[i]['label']}<br>{varname} = {values[i]:.4g}" for i in range(len(values))
    ]
    fig.add_trace(
        go.Scatter(
            x=cx,
            y=cy,
            mode="markers",
            # invisible marker: keeps per-box hover text and click-to-select
            # working without drawing a dot on top of each polygon
            marker=dict(size=20, color="rgba(0,0,0,0)"),
            customdata=list(gdf["box"]),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    # dummy trace purely to render a colorbar for the fill colours above
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                colorscale=COLORSCALE,
                cmin=vmin,
                cmax=vmax,
                color=[vmin, vmax],
                showscale=True,
                colorbar=dict(title=varname),
                size=0.0001,
            ),
            hoverinfo="none",
            showlegend=False,
        )
    )

    meta = ao.var_meta(varname)
    title = f"{meta['long_name']} ({meta['units']}) | {ao.date_label(t_index)}"
    if ao.is_layered(varname):
        title += f" | {layer_type} layer {layer_index} [{ao.depth_label(t_index, selected_box, layer_index, layer_type)}]"

    fig.update_layout(
        **_base_layout(title, height=650),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def make_profile_figure(ao: AtlantisOutput, varname, t_index, box_index, layer_type, height=350,
                         xmin=None, xmax=None):
    fig = go.Figure()
    if not ao.is_layered(varname):
        fig.update_layout(**_base_layout(f"{varname} has no depth dimension", height))
        return fig
    mids, values, bounds = ao.profile(varname, t_index, box_index, layer_type)
    step_x = np.empty(2 * len(values))
    step_y = np.empty(2 * len(values))
    step_y[0::2] = bounds[:-1]
    step_y[1::2] = bounds[1:]
    step_x[0::2] = values
    step_x[1::2] = values
    fig.add_trace(go.Scatter(x=step_x, y=step_y, mode="lines+markers", line=dict(color=LINE_COLOR)))
    meta = ao.var_meta(varname)
    xaxis = dict(title=f"{meta['long_name']} ({meta['units']})", gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    if xmin is not None and xmax is not None:
        xaxis["range"] = [xmin, xmax]
    fig.update_layout(
        **_base_layout(f"Profile | {ao.box_labels[box_index]} | {ao.date_label(t_index)}", height),
        xaxis=xaxis,
        yaxis=dict(title="Depth (m)", gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


def make_timeseries_figure(ao: AtlantisOutput, varname, box_index, layer_index, layer_type, t_index,
                            ymin=None, ymax=None):
    times, values = ao.timeseries(varname, box_index, layer_index, layer_type)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=values, mode="lines", line=dict(color=LINE_COLOR)))
    fig.add_vline(x=times[t_index], line=dict(color=HEADING_COLOR))
    meta = ao.var_meta(varname)
    title = f"Time series | {ao.box_labels[box_index]}"
    if ao.is_layered(varname):
        title += f" | {layer_type} layer {layer_index}"
    yaxis = dict(title=f"{meta['long_name']} ({meta['units']})", gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    if ymin is not None and ymax is not None:
        yaxis["range"] = [ymin, ymax]
    fig.update_layout(
        **_base_layout(title, height=350),
        xaxis=dict(title="Time", gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=yaxis,
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


def scale_controls(prefix: str):
    """Autoscale checkbox + min/max number inputs, used above the map legend,
    the time series y-axis, and the profile x-axis."""
    return html.Div(
        [
            dcc.Checklist(
                id=f"{prefix}-autoscale",
                options=[{"label": "Autoscale", "value": "auto"}],
                value=["auto"],
                inline=True,
                style={"display": "inline-block"},
            ),
            html.Label("Min"),
            dcc.Input(id=f"{prefix}-min", type="number", disabled=True, style={"width": "90px"}),
            html.Label("Max"),
            dcc.Input(id=f"{prefix}-max", type="number", disabled=True, style={"width": "90px"}),
        ],
        className="scale-controls",
    )


def build_app(ao: AtlantisOutput) -> Dash:
    app = Dash(__name__)
    app.title = "Atlantis Output Viewer"

    var_options = build_variable_options(ao)
    default_var = "NH3" if "NH3" in ao.variables() else var_options[0]["value"]
    box_options = [{"label": lbl, "value": i} for i, lbl in enumerate(ao.box_labels)]
    has_sediment = ao.sednz > 0

    app.layout = html.Div(
        [
            html.H2("Atlantis Output Viewer", className="app-title"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Variable", className="section-title"),
                            dcc.Dropdown(id="var-dd", options=var_options, value=default_var, clearable=False),
                        ],
                        className="control-group",
                        style={"width": "380px"},
                    ),
                    html.Div(
                        [
                            html.Label("Layer type", className="section-title"),
                            dcc.RadioItems(
                                id="layertype-radio",
                                options=[{"label": "Water column", "value": "wc"}]
                                + ([{"label": "Sediment", "value": "sed"}] if has_sediment else []),
                                value="wc",
                                inline=True,
                            ),
                        ],
                        className="control-group",
                    ),
                    html.Div(
                        [
                            html.Label("Selected box", className="section-title"),
                            dcc.Dropdown(id="box-dd", options=box_options, value=0, clearable=False),
                        ],
                        className="control-group",
                        style={"width": "160px"},
                    ),
                ],
                className="panel controls-row",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [scale_controls("map"), dcc.Graph(id="map-graph")],
                                className="graph-card",
                            ),
                            html.Div(
                                [scale_controls("ts"), dcc.Graph(id="timeseries-graph")],
                                className="graph-card",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label(id="time-label", className="section-title"),
                                            html.Button("Play", id="play-btn", n_clicks=0),
                                        ],
                                        className="time-row",
                                    ),
                                    dcc.Slider(id="time-slider", min=0, max=ao.ntime - 1, step=1, value=0,
                                               marks=build_time_marks(ao), tooltip={"placement": "bottom"}),
                                    dcc.Interval(id="play-interval", interval=400, disabled=True),
                                ],
                                className="panel",
                            ),
                        ],
                        style={"width": "55%", "display": "inline-block", "verticalAlign": "top"},
                    ),
                    html.Div(
                        [
                            html.Label("Layer", className="section-title"),
                            html.Div(
                                dcc.Slider(
                                    id="layer-slider", min=0, max=max(ao.wcnz - 1, 0), step=1, value=0,
                                    marks=None, vertical=True, verticalHeight=PROFILE_HEIGHT,
                                    tooltip={"placement": "right"},
                                ),
                                style={"height": f"{PROFILE_HEIGHT}px", "marginTop": "10px"},
                            ),
                            html.Div(id="layer-label", className="layer-depth-label"),
                        ],
                        className="slider-card",
                        style={"width": "130px", "display": "inline-block", "verticalAlign": "top",
                               "marginLeft": "16px"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [scale_controls("profile"), dcc.Graph(id="profile-graph")],
                                className="graph-card",
                            ),
                        ],
                        style={"width": "30%", "display": "inline-block", "verticalAlign": "top",
                               "marginLeft": "16px"},
                    ),
                ],
            ),
        ],
        className="app-root",
    )

    @app.callback(
        Output("layer-slider", "max"),
        Output("layer-slider", "value"),
        Input("layertype-radio", "value"),
        State("layer-slider", "value"),
    )
    def _update_layer_range(layer_type, current_value):
        n = ao.n_layers(layer_type)
        new_max = max(n - 1, 0)
        new_value = min(current_value, new_max)
        return new_max, new_value

    @app.callback(Output("layer-label", "children"), Input("layer-slider", "value"),
                  Input("layertype-radio", "value"), Input("box-dd", "value"),
                  Input("time-slider", "value"))
    def _layer_label(layer_index, layer_type, box_index, t_index):
        depth = ao.depth_label(t_index, box_index, layer_index, layer_type)
        return f"Layer {layer_index} ({layer_type}) at {ao.box_labels[box_index]}: {depth}"

    @app.callback(Output("time-label", "children"), Input("time-slider", "value"))
    def _time_label(t_index):
        return f"Time: {ao.date_label(t_index)} (step {t_index}/{ao.ntime - 1})"

    @app.callback(
        Output("play-interval", "disabled"),
        Output("play-btn", "children"),
        Input("play-btn", "n_clicks"),
        State("play-interval", "disabled"),
    )
    def _toggle_play(n_clicks, disabled):
        if n_clicks == 0:
            return True, "Play"
        new_disabled = not disabled
        return new_disabled, ("Play" if new_disabled else "Pause")

    @app.callback(
        Output("time-slider", "value"),
        Input("play-interval", "n_intervals"),
        State("time-slider", "value"),
        prevent_initial_call=True,
    )
    def _advance_time(_n, t_index):
        return (t_index + 1) % ao.ntime

    @app.callback(Output("box-dd", "value"), Input("map-graph", "clickData"), prevent_initial_call=True)
    def _select_box_from_click(click_data):
        if not click_data:
            return no_update
        point = click_data["points"][0]
        if "customdata" not in point:
            return no_update
        return int(point["customdata"])

    def _is_auto(autoscale_value):
        return "auto" in (autoscale_value or [])

    # --- autoscale checkbox -> enable/disable + prefill the min/max inputs
    # with the current data range whenever scaling mode or selection changes.

    @app.callback(
        Output("map-min", "value"),
        Output("map-max", "value"),
        Output("map-min", "disabled"),
        Output("map-max", "disabled"),
        Input("map-autoscale", "value"),
        Input("var-dd", "value"),
        Input("time-slider", "value"),
        Input("layer-slider", "value"),
        Input("layertype-radio", "value"),
    )
    def _map_scale_toggle(autoscale_value, varname, t_index, layer_index, layer_type):
        is_auto = _is_auto(autoscale_value)
        li = layer_index if ao.is_layered(varname) else 0
        values = ao.map_values(varname, t_index, li, layer_type)
        data_min, data_max = float(np.nanmin(values)), float(np.nanmax(values))
        if is_auto:
            return data_min, data_max, True, True
        return no_update, no_update, False, False

    @app.callback(
        Output("ts-min", "value"),
        Output("ts-max", "value"),
        Output("ts-min", "disabled"),
        Output("ts-max", "disabled"),
        Input("ts-autoscale", "value"),
        Input("var-dd", "value"),
        Input("box-dd", "value"),
        Input("layer-slider", "value"),
        Input("layertype-radio", "value"),
    )
    def _ts_scale_toggle(autoscale_value, varname, box_index, layer_index, layer_type):
        is_auto = _is_auto(autoscale_value)
        li = layer_index if ao.is_layered(varname) else 0
        _, values = ao.timeseries(varname, box_index, li, layer_type)
        data_min, data_max = float(np.nanmin(values)), float(np.nanmax(values))
        if is_auto:
            return data_min, data_max, True, True
        return no_update, no_update, False, False

    @app.callback(
        Output("profile-min", "value"),
        Output("profile-max", "value"),
        Output("profile-min", "disabled"),
        Output("profile-max", "disabled"),
        Input("profile-autoscale", "value"),
        Input("var-dd", "value"),
        Input("time-slider", "value"),
        Input("box-dd", "value"),
        Input("layertype-radio", "value"),
    )
    def _profile_scale_toggle(autoscale_value, varname, t_index, box_index, layer_type):
        is_auto = _is_auto(autoscale_value)
        if not ao.is_layered(varname):
            return no_update, no_update, True, True
        _, values, _ = ao.profile(varname, t_index, box_index, layer_type)
        data_min, data_max = float(np.nanmin(values)), float(np.nanmax(values))
        if is_auto:
            return data_min, data_max, True, True
        return no_update, no_update, False, False

    # --- figures themselves: use the fixed min/max whenever autoscale is off

    @app.callback(
        Output("map-graph", "figure"),
        Input("var-dd", "value"),
        Input("time-slider", "value"),
        Input("layer-slider", "value"),
        Input("layertype-radio", "value"),
        Input("box-dd", "value"),
        Input("map-autoscale", "value"),
        Input("map-min", "value"),
        Input("map-max", "value"),
    )
    def _update_map(varname, t_index, layer_index, layer_type, box_index, autoscale_value, vmin, vmax):
        li = layer_index if ao.is_layered(varname) else 0
        if _is_auto(autoscale_value):
            vmin = vmax = None
        return make_map_figure(ao, varname, t_index, li, layer_type, box_index, vmin=vmin, vmax=vmax)

    @app.callback(
        Output("profile-graph", "figure"),
        Input("var-dd", "value"),
        Input("time-slider", "value"),
        Input("box-dd", "value"),
        Input("layertype-radio", "value"),
        Input("profile-autoscale", "value"),
        Input("profile-min", "value"),
        Input("profile-max", "value"),
    )
    def _update_profile(varname, t_index, box_index, layer_type, autoscale_value, xmin, xmax):
        if _is_auto(autoscale_value):
            xmin = xmax = None
        return make_profile_figure(ao, varname, t_index, box_index, layer_type, height=PROFILE_HEIGHT,
                                    xmin=xmin, xmax=xmax)

    @app.callback(
        Output("timeseries-graph", "figure"),
        Input("var-dd", "value"),
        Input("box-dd", "value"),
        Input("layer-slider", "value"),
        Input("layertype-radio", "value"),
        Input("time-slider", "value"),
        Input("ts-autoscale", "value"),
        Input("ts-min", "value"),
        Input("ts-max", "value"),
    )
    def _update_timeseries(varname, box_index, layer_index, layer_type, t_index, autoscale_value, ymin, ymax):
        li = layer_index if ao.is_layered(varname) else 0
        if _is_auto(autoscale_value):
            ymin = ymax = None
        return make_timeseries_figure(ao, varname, box_index, li, layer_type, t_index, ymin=ymin, ymax=ymax)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nc", required=True, help="Path to Atlantis output netCDF file")
    parser.add_argument("--bgm", default=None, help="Path to .bgm geometry file (auto-detected if omitted)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    ao = AtlantisOutput(args.nc, args.bgm)
    if ao.bgm is None:
        print("Warning: no .bgm geometry found/loaded; the Map view will not be available.")
    app = build_app(ao)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
