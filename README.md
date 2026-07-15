# Atlantis Output Viewer

A Python replica of the parts of CSIRO's **Olive** Java tool (`Olive2Jar.jar`,
`cmr.apps.olive.*`) needed to look at Atlantis box-model output: irregular
box geometry from a `.bgm` file, and 4D (time x box x depth-layer x variable)
netCDF output such as `outputSETAS.nc`.

## Why Python, and what was replicated

Olive's code was decompiled and read directly (`BMDataAccessor`,
`BMParameters`, `BMPolygonPatch3D`, `ScalarMapDataLayer`/`MapPlotPanel`,
`ProfilePlotPanel`, `TSPlotPanel`, and `Olive.setupFile`'s accessor dispatch)
to confirm exactly what the Atlantis file-type path in Olive does, rather
than guessing:

* Box geometry (polygons, labels, centres, bottom depth) comes only from a
  handful of `.bgm` tags (`box<i>.label/.inside/.vert/.botz`, `bnd_vert`,
  `projection`) -- see `bgm.py`.
* The netCDF `z` dimension is `wcnz` water-column layers followed by `sednz`
  sediment layers (global attributes). Per-box depth boundaries are built by
  cumulatively summing that box's own `dz` (or `nominal_dz`) from the bottom
  up to 0 m -- this is exactly `BMDataAccessor.createZGrids`, reimplemented
  in `ncdata.py:layer_bounds`. Because each box has a different bottom
  depth, "layer index 0" is a different absolute depth range in different
  boxes (a shallow box's bottom layer is a degenerate/zero-thickness
  "padding" layer) -- this is inherent to the data, not a bug.
* The Map view always shows a single (time, layer) slice, one colour per
  box -- Olive never depth-averages for the shaded map.
* The Profile view is a vertical profile for one box at one time, using
  that box's own non-uniform layer thicknesses.
* The Time Series view is one box (+ one layer, for 3D variables) across
  the whole run.
* Olive's **Section** (transect) view and vector/current arrows are gated on
  `GridDataAccessor` (regular-grid ocean models like MOM/CARS) and never
  apply to `BMDataAccessor`/box geometry -- there is no equivalent for
  Atlantis output, in Olive or here.

Python (`xarray` + `geopandas`/`shapely`/`pyproj` + `Dash`/`Plotly`) was
chosen over R because `xarray` maps directly onto the netCDF's native
labelled dimensions (`t`, `b`, `z`) with no reshaping, and
`geopandas`/`shapely` handle the irregular (non-gridded, pre-shapefile) box
polygons and the file's Albers-equal-area `proj4` string natively. An
equivalent R stack (`ncdf4`/`sf`/`shiny`) would work too, but requires more
glue code to reproduce the same box-relative layer math.

## Files

- `bgm.py` -- parses a `.bgm` file into box polygons/labels/centres/botz and
  the model boundary polygon, as a `geopandas.GeoDataFrame` in the file's
  native projected CRS.
- `ncdata.py` -- `AtlantisOutput` wraps an output netCDF file (+ its `.bgm`,
  auto-located from the netCDF's `geometry` global attribute if not given
  explicitly). Provides variable listing/metadata, per-box layer boundaries,
  and `map_values` / `profile` / `timeseries` extraction methods.
- `plotting.py` -- static matplotlib `plot_map` / `plot_profile` /
  `plot_timeseries` functions for scripting or notebook use.
- `app.py` -- interactive Dash web app: Map + Profile + Time series views
  side by side, with variable/box/layer-type dropdowns, a time slider with
  a Play button (animation), and click-to-select-box on the map.

## Install

```
pip install -r requirements.txt
```

(`xarray` reads this classic-format netCDF via its built-in `scipy` engine;
no `netCDF4` package is required for files like `outputSETAS.nc`. If you
have NetCDF4/HDF5-format output instead, also `pip install netCDF4`.)

## Interactive viewer

```
python3 atlantis_viewer_app.py --nc /path/to/outputFilename.nc
```

Then open http://127.0.0.1:8050/. Pass `--bgm /path/to/model.bgm`
explicitly if it isn't next to the netCDF file (or its parent directory).

## Scripting / notebook use

```python
from ncdata import AtlantisOutput
from plotting import plot_map, plot_profile, plot_timeseries

ao = AtlantisOutput("outputSETAS.nc")  # finds VMPA_setas.bgm automatically
plot_map(ao, "NH3", t_index=20, layer_index=4, layer_type="wc")
plot_profile(ao, "NH3", t_index=20, box_index=4)
plot_timeseries(ao, "NH3", box_index=4, layer_index=4)
```

`AtlantisOutput.variables(bmtype=...)` lists variable names (optionally
filtered by `bmtype`, e.g. `"tracer"`, `"phys"`, `"epibenthos"`);
`AtlantisOutput.var_meta(name)` gives `long_name`/`units`/`bmtype`.

## Tested against

- `VMPA_setas.bgm` (11 boxes, Albers equal-area projection) -- box areas
  from the parsed polygons match the file's own reported `box<i>.area`
  values to within floating-point/rounding precision.
- `outputSETAS.nc` (101 time steps, 11 boxes, 7 z-levels = 6 water column +
  1 sediment, 377 output variables) -- map/profile/time-series extraction
  and the full interactive app (including every callback) were exercised
  directly against this file.
