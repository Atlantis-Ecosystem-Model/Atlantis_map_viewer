"""
Reader/accessor for Atlantis box-model netCDF output (e.g. outputSETAS.nc),
replicating the relevant behaviour of cmr.apps.olive.BMDataAccessor from the
Olive Java tool:

  * the z dimension is a concatenation of `wcnz` water-column layers followed
    by `sednz` sediment layers (global attributes "wcnz"/"sednz"); a "layer
    type" (water column vs sediment) plus a 0-based layer index within that
    type addresses a single z slice.
  * per-box vertical layer boundaries are built by cumulatively summing that
    box's `dz` (preferred, time-varying) or `nominal_dz` (fallback, static)
    from the bottom of the box up to the surface (0 m) -- see
    BMDataAccessor.createZGrids in the original tool. This is robust to
    "padding" zero-thickness layers appearing anywhere in the stack (shallow
    boxes do not use every possible layer).
  * the Map view always shows a single (time, layer) slice per box -- Olive
    never depth-averages for the shaded map.
  * Profile view uses a single box's own layer boundaries.
  * Time series view uses a single box (+ single layer, for layered vars).

A companion .bgm file (see bgm.py) provides box polygons/labels for the map
view; it is optional for profile/time-series-only use.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from bgm import parse_bgm, BgmGeometry

_NON_DATA_VARS = {"t"}
_DIM_ONLY_VARS = {"dz", "nominal_dz"}  # geometry-support vars, not shown as "data" by default


class AtlantisOutput:
    def __init__(self, nc_path: str | Path, bgm_path: str | Path | None = None):
        self.nc_path = Path(nc_path)
        try:
            self.ds = xr.open_dataset(self.nc_path, decode_times=True)
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                f"Could not open {self.nc_path} with xarray. If this is a "
                f"NetCDF4/HDF5 file you may need to `pip install netCDF4` or "
                f"`pip install h5netcdf`. Original error: {exc}"
            ) from exc

        self.wcnz = int(self.ds.attrs.get("wcnz", self.ds.sizes.get("z", 0)))
        self.sednz = int(self.ds.attrs.get("sednz", 0))
        self.nbox = int(self.ds.sizes["b"])
        self.nz = int(self.ds.sizes.get("z", 0))
        self.ntime = int(self.ds.sizes["t"])

        self.dz_varname = "dz" if "dz" in self.ds.variables else (
            "nominal_dz" if "nominal_dz" in self.ds.variables else None
        )

        self.bgm: BgmGeometry | None = None
        if bgm_path is not None:
            self.bgm = parse_bgm(bgm_path)
        else:
            geom_name = self.ds.attrs.get("geometry")
            if geom_name:
                for candidate_dir in (self.nc_path.parent, self.nc_path.parent.parent):
                    candidate = candidate_dir / geom_name
                    if candidate.exists():
                        self.bgm = parse_bgm(candidate)
                        break

        if self.bgm is not None and self.bgm.nbox == self.nbox:
            self.box_labels = [b.label for b in self.bgm.boxes]
        else:
            self.box_labels = [f"Box{i}" for i in range(self.nbox)]

    # ---------------------------------------------------------------- meta
    def close(self):
        self.ds.close()

    @property
    def time_values(self) -> np.ndarray:
        return self.ds["t"].values

    def date_label(self, t_index: int) -> str:
        t = self.time_values[t_index]
        ts = np.datetime_as_string(t, unit="s").replace("T", " ")
        return ts

    def variables(self, bmtype: str | None = None) -> list[str]:
        names = []
        for name, da in self.ds.data_vars.items():
            if name in _NON_DATA_VARS or name in _DIM_ONLY_VARS:
                continue
            if "b" not in da.dims:
                continue
            if bmtype is not None and da.attrs.get("bmtype") != bmtype:
                continue
            names.append(name)
        return names

    def bmtypes(self) -> list[str]:
        seen = []
        for name in self.variables():
            bt = self.ds[name].attrs.get("bmtype", "other")
            if bt not in seen:
                seen.append(bt)
        return seen

    def var_meta(self, varname: str) -> dict:
        da = self.ds[varname]
        return {
            "long_name": da.attrs.get("long_name", varname),
            "units": da.attrs.get("units", ""),
            "bmtype": da.attrs.get("bmtype", "other"),
        }

    def is_layered(self, varname: str) -> bool:
        return "z" in self.ds[varname].dims

    def is_time_varying(self, varname: str) -> bool:
        return "t" in self.ds[varname].dims

    # --------------------------------------------------------- layer maths
    def _z_slice(self, layer_type: str) -> slice:
        if layer_type == "wc":
            return slice(0, self.wcnz)
        if layer_type == "sed":
            return slice(self.wcnz, self.wcnz + self.sednz)
        raise ValueError("layer_type must be 'wc' or 'sed'")

    def n_layers(self, layer_type: str) -> int:
        return self.wcnz if layer_type == "wc" else self.sednz

    def dz_for(self, t_index: int, box_index: int, layer_type: str = "wc") -> np.ndarray:
        if self.dz_varname is None:
            raise RuntimeError("No dz/nominal_dz variable found in this file")
        da = self.ds[self.dz_varname]
        zsl = self._z_slice(layer_type)
        if "t" in da.dims:
            vals = da.isel(t=t_index, b=box_index, z=zsl).values
        else:
            vals = da.isel(b=box_index, z=zsl).values
        return np.asarray(vals, dtype=float)

    def layer_bounds(self, t_index: int, box_index: int, layer_type: str = "wc") -> np.ndarray:
        """Cumulative depth boundaries (length n_layers+1, m, negative down,
        0 = surface for 'wc' or sediment surface for 'sed'), replicating
        BMDataAccessor.createZGrids: layer 0 is the deepest, and depth
        boundaries are summed from the bottom of the box up to 0."""
        dz = self.dz_for(t_index, box_index, layer_type)
        n = len(dz)
        bounds = np.empty(n + 1)
        bounds[0] = -dz.sum()
        for j in range(1, n + 1):
            bounds[j] = bounds[j - 1] + dz[j - 1]
        return bounds

    def depth_label(self, t_index: int, box_index: int, layer_index: int, layer_type: str = "wc") -> str:
        bounds = self.layer_bounds(t_index, box_index, layer_type)
        return f"{bounds[layer_index]:.1f} to {bounds[layer_index + 1]:.1f} m"

    # -------------------------------------------------------------- views
    def map_values(
        self,
        varname: str,
        t_index: int,
        layer_index: int = 0,
        layer_type: str = "wc",
    ) -> np.ndarray:
        """Values of `varname` for every box at a single (time, layer) slice --
        exactly what Olive's shaded box map shows (never depth-averaged)."""
        da = self.ds[varname]
        sel = {}
        if "t" in da.dims:
            sel["t"] = t_index
        if "z" in da.dims:
            z_idx = layer_index + (self.wcnz if layer_type == "sed" else 0)
            sel["z"] = z_idx
        return np.asarray(da.isel(**sel).values, dtype=float)

    def profile(
        self, varname: str, t_index: int, box_index: int, layer_type: str = "wc"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (layer_mid_depths, values, layer_bounds) for a vertical
        profile of `varname` at one box and time, using that box's own
        (non-uniform) layer thicknesses."""
        if not self.is_layered(varname):
            raise ValueError(f"{varname} has no z dimension; cannot build a profile")
        da = self.ds[varname]
        zsl = self._z_slice(layer_type)
        sel = {"b": box_index, "z": zsl}
        if "t" in da.dims:
            sel["t"] = t_index
        values = np.asarray(da.isel(**sel).values, dtype=float)
        bounds = self.layer_bounds(t_index, box_index, layer_type)
        mids = (bounds[:-1] + bounds[1:]) / 2.0
        return mids, values, bounds

    def timeseries(
        self,
        varname: str,
        box_index: int,
        layer_index: int | None = 0,
        layer_type: str = "wc",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (times, values) for `varname` at one box across all time
        steps. `layer_index` is ignored for variables without a z dimension."""
        da = self.ds[varname]
        sel = {"b": box_index}
        if "z" in da.dims:
            z_idx = (layer_index or 0) + (self.wcnz if layer_type == "sed" else 0)
            sel["z"] = z_idx
        values = np.asarray(da.isel(**sel).values, dtype=float)
        return self.time_values, values
