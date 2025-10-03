import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import pyproj
import xarray as xr
from metpy.units import units
from omegaconf.errors import ConfigAttributeError, ConfigKeyError

from . import _helpers as h
from . import meteorology_helpers as mh
from . import thermodynamics as td
from .readers.readers import pysondeL1

def _safe_to(da, target_unit: str, default_unit: str | None = None):
    """
    Convert DataArray to `target_unit` robustly.
    - If already a Quantity → .pint.to(target_unit)
    - Else if has attrs['units'] → quantify then convert
    - Else → quantify with default_unit (or target_unit) then convert
    Always returns a Quantity DataArray.
    """
    # Already quantified?
    try:
        return da.pint.to(target_unit)
    except Exception:
        pass

    # Try from declared attribute
    u = getattr(da, "attrs", {}).get("units")
    if u:
        try:
            return da.pint.quantify(u).pint.to(target_unit)
        except Exception:
            pass

    # Last resort: assume a unit (or target if none provided)
    assumed = default_unit or target_unit
    return da.pint.quantify(assumed).pint.to(target_unit)


def _values_in_unit(da, target_unit: str, default_unit: str | None = None):
    """Return numeric values in target_unit if possible; otherwise fall back safely."""
    # Try pint conversion if already quantified
    try:
        return da.pint.to(target_unit).values
    except Exception:
        pass
    # Try from attr
    u = getattr(da, "attrs", {}).get("units")
    if u:
        try:
            return da.pint.quantify(u).pint.to(target_unit).values
        except Exception:
            pass
    # Fallback: assume default or target
    arr = da.values
    if default_unit and default_unit != target_unit:
        if default_unit == "Pa" and target_unit in ("hPa", "mbar"):
            return arr / 100.0
        if default_unit in ("hPa", "mbar") and target_unit == "Pa":
            return arr * 100.0
        if default_unit == "km" and target_unit == "m":
            return arr * 1000.0
    return arr


def prepare_data_for_interpolation(ds, uni, vertical_interpolation_axis, variables, reader=pysondeL1):
    u, v = mh.get_wind_components(ds.wdir, ds.wspd)
    ds["u"] = xr.DataArray(u.data, dims=["level"])
    ds["v"] = xr.DataArray(v.data, dims=["level"])

    if "alt_WGS84" in ds.keys():
        # Convert lat, lon, alt to cartesian coordinates
        ecef = pyproj.Proj(proj="geocent", ellps="WGS84", datum="WGS84")
        lla = pyproj.Proj(proj="latlong", ellps="WGS84", datum="WGS84")
        x, y, z = pyproj.transform(
            lla,
            ecef,
            ds.lon.values,
            ds.lat.values,
            ds["alt_WGS84"].values,
            radians=False,
        )
        for var, val in {"x": x, "y": y, "z": z}.items():
            ds[var] = xr.DataArray(val, dims=["level"])
    else:
        logging.warning(
            "No WGS84 altitude could be found. The averaging of the position might be faulty especially at the 0 meridian and close to the poles"
        )

    # --- Make 'alt' the vertical dimension for the entire dataset ---
    if "level" in ds.dims:
        # ensure 'alt' is a coordinate (if it's currently a data_var)
        if vertical_interpolation_axis in ds and vertical_interpolation_axis not in ds.coords:
            ds = ds.set_coords(vertical_interpolation_axis)

        # preferred path: swap_dims when 'alt' is a 1-D coord over 'level'
        if vertical_interpolation_axis in ds.coords and ds[vertical_interpolation_axis].dims == ("level",):
            ds = ds.swap_dims({"level": vertical_interpolation_axis})
        else:
            # fallback: if an 'alt' data_var conflicts, rename it temporarily, then rename the dim
            if vertical_interpolation_axis in ds.variables and vertical_interpolation_axis not in ds.dims:
                ds = ds.rename({vertical_interpolation_axis: "alt_var"})
            ds = ds.rename({"level": vertical_interpolation_axis})
    # ----------------------------------------------------------------

    # Thermodynamics on the dataset now using 'alt' as the vertical dim
    td.metpy.units.units = uni
    theta = td.calc_theta_from_T(ds["ta"], ds["p"])
    e_s = td.calc_saturation_pressure(ds["ta"], method="wagner_pruss")
    e   = ds["rh"] * e_s
    w   = td.calc_wv_mixing_ratio(ds, e)
    q   = w / (1 + w)

    # Add 'sounding' dim for derived fields to match Level-2 expectations
    w     = w.expand_dims({"sounding": 1})
    q     = q.expand_dims({"sounding": 1})
    theta = theta.expand_dims({"sounding": 1})

    # Optional: drop non-essential coords to keep ds_new clean (alt remains as a dimension)
    ds = ds.reset_coords()
    ds = ds.expand_dims({"sounding": 1})

    # New dataset holding fields needed for interpolation & outputs
    ds_new = xr.Dataset()
    ds_new["mr"]                = w.reset_coords(drop=True)
    ds_new["theta"]             = theta.reset_coords(drop=True)
    ds_new["specific_humidity"] = q.reset_coords(drop=True)

    # NEW: carry PTU geopotential height into L2 (NaNs for Meteomodem)
    #if "height_ptu" in ds:
    #    ds_new["height_ptu"] = ds["height_ptu"]

    # Carry over remaining variables from ds (that aren't already present)
    for var in ds.data_vars:
        if var not in ds_new.data_vars and var not in ds_new.coords:
            try:
                ds_new[var] = ds[var]
            except NameError:
                logging.warning(f"Variable {var} not found.")
                pass

    # Final variable renames according to Level-2 'variables' mapping
    for variable_name_in, variable_name_out in variables:
        try:
            ds_new = ds_new.rename({variable_name_in: variable_name_out})
            # copy attrs from source if present in ds
            if variable_name_in in ds:
                ds_new[variable_name_out].attrs = ds[variable_name_in].attrs
        except (ValueError, KeyError):
            logging.warning(f"Variable {variable_name_in} not found.")
            pass

    return ds, ds_new


def interpolation(ds_new, method, interpolation_grid, vertical_interpolation_axis, sounding, variables, cfg):
    """
    Interpolate/bin Level-2 dataset along GPS altitude 'alt'.

    Parameters
    ----------
    ds_new : xr.Dataset
        Output of prepare_data_for_interpolation(), with 'alt' as vertical dim.
    method : {"linear","bin"}
        Interpolation strategy.
    interpolation_grid : np.ndarray
        Target altitude grid (centers) in meters.
    sounding : object
        Sounding object (for unitregistry etc).
    variables : iterable[(in_name, out_name)]
        Variable rename mapping for outputs.
    cfg : omegaconf.DictConfig
        Full merged config.

    Returns
    -------
    ds_interp : xr.Dataset
    """
    
    if method == "linear":
        print("[pysonde] Using vertical interpolation method: linear")

        # --- clean & sort only by 'alt' ---
        ds_sorted = ds_new.sortby(vertical_interpolation_axis)
        ds_sorted = ds_sorted.where(np.isfinite(ds_sorted[vertical_interpolation_axis]), drop=True)

        # Interpolate on the target alt grid
        ds_interp = ds_sorted.interp(
            {vertical_interpolation_axis: interpolation_grid},
            method="linear"
            )

        print("pressure before:", ds_sorted.pressure.values)
        # --- robust log-p interpolation to preserve p(z) profile ---
        p_name = "pressure" if "pressure" in ds_new else ("p" if "p" in ds_new else None)
        if (p_name is not None) and (vertical_interpolation_axis in ds_new.dims or vertical_interpolation_axis in ds_new.coords):
            p_da = ds_sorted[p_name]
            z_da = ds_sorted[vertical_interpolation_axis]

            # reduce to 1D along alt if needed
            if "sounding" in p_da.dims:
                p_da = p_da.isel(sounding=0)
            if "sounding" in z_da.dims:
                z_da = z_da.isel(sounding=0)

            # numeric arrays with units handled
            p_src = np.asarray(_values_in_unit(p_da, "hPa", default_unit="hPa")).ravel()
            z_src = np.asarray(_values_in_unit(z_da, "m",   default_unit="m")).ravel()

            mfin = np.isfinite(p_src) & np.isfinite(z_src)
            p_src, z_src = p_src[mfin], z_src[mfin]

            if z_src.size >= 2:
                order = np.argsort(z_src)
                z_src, p_src = z_src[order], p_src[order]
                z_out = np.asarray(ds_interp[vertical_interpolation_axis].values).ravel()
                # print("\nz_src:", z_src)
                # print("\np_src:", p_src)
                # print("\nz_out:", z_out)

                # print(f"[pysonde] log-p inputs: p_src={p_src.shape}, z_src={z_src.shape}, z_out={z_out.shape}")
                p_out = mh.pressure_interpolation(p_src, z_src, z_out)  # numeric hPa
                # print("\np_out:", p_out)

                # --- minimal endpoint fix for p_out without changing mh.pressure_interpolation ---
                # fill p_out[0] if it's NaN but within the measured span
                zmin, zmax = z_src[0], z_src[-1]
                # NOTE: z_src is sorted above

                def _fill_endpoint(z_target, idx_target):
                    if np.isnan(p_out[idx_target]) and (zmin <= z_target <= zmax):
                        # find bracketing indices in z_src
                        j_hi = np.searchsorted(z_src, z_target, side="left")
                        if 0 < j_hi < len(z_src):
                            j_lo = j_hi - 1
                            z1, z2 = z_src[j_lo], z_src[j_hi]
                            p1, p2 = p_src[j_lo], p_src[j_hi]
                            if np.isfinite(z1) and np.isfinite(z2) and np.isfinite(p1) and np.isfinite(p2) and (z2 > z1):
                                # log-linear in pressure vs altitude
                                lp = np.interp(z_target, [z1, z2], [np.log(p1), np.log(p2)])
                                p_out[idx_target] = float(np.exp(lp))

                # first bin (usually 0 m)
                _fill_endpoint(z_out[0], 0)
                # last bin
                _fill_endpoint(z_out[-1], -1)
                # print("\np_out out:", p_out)

                ds_interp[p_name] = xr.DataArray(
                    p_out,
                    dims=(vertical_interpolation_axis,),
                    coords={vertical_interpolation_axis: ds_interp[vertical_interpolation_axis].values},
                    attrs={**ds_new[p_name].attrs, "units": "hPa"},
                )
        # print("pressure after:", ds_interp.pressure.values)

        # --- ensure pressure has Pint units downstream ---
        p_name = "pressure" if "pressure" in ds_interp else ("p" if "p" in ds_interp else None)
        if p_name is not None:
            ds_interp[p_name] = _safe_to(ds_interp[p_name], "hPa", default_unit="hPa")
        # print("pressure after after:", ds_interp.pressure.values)

        # --- unit harmonization for other variables ---
        for var_in, var_out in variables:
            try:
                ds_interp[var_out] = ds_interp[var_out].pint.quantify(ds_new[var_out].pint.units)
                ds_interp[var_out] = ds_interp[var_out].pint.to(cfg.level2.variables[var_in].attrs.units)
            except (KeyError, ValueError, ConfigAttributeError):
                pass

        # --- guarantee size-1 'sounding' dim (adjust_ds_after_interpolation expects it) ---
        if "sounding" not in ds_interp.dims:
            ds_interp = ds_interp.expand_dims({"sounding": 1})

        # --- ensure scalar launch_time present (0-D datetime64[ns]) ---
        def _first_finite_dt64_scalar(da):
            """Return first finite datetime64[ns] from a time-like DataArray as 0-D."""
            # reduce any extra dims (e.g., sounding/alt) by taking the first finite element
            vals = da.values
            # to ns int for finite mask
            ns = vals.astype("datetime64[ns]").astype("int64")
            m = np.isfinite(ns)
            if m.any():
                return xr.DataArray(vals[m][0].astype("datetime64[ns]"), dims=())
            return None

        lt_scalar = None

        # 1) prefer a scalar 'launch_time' from the input
        if "launch_time" in ds_new and ds_new["launch_time"].ndim == 0:
            lt_scalar = xr.DataArray(
                ds_new["launch_time"].values.astype("datetime64[ns]"), dims=()
            )
        # 2) fallback: derive from per-level time variables in the ORIGINAL input
        elif any(v in ds_new for v in ("flight_time", "time")):
            for cand in ("flight_time", "time"):
                if cand in ds_new:
                    tvar = ds_new[cand]
                    # drop sounding dim if present
                    if "sounding" in tvar.dims:
                        tvar = tvar.isel(sounding=0)
                    # pick first finite timestamp
                    lt_scalar = _first_finite_dt64_scalar(tvar)
                    if lt_scalar is not None:
                        break
        # 3) final fallback: dataset attribute
        if lt_scalar is None and "launch_time" in ds_new.attrs:
            lt_scalar = xr.DataArray(
                np.datetime64(ds_new.attrs["launch_time"]).astype("datetime64[ns]"),
                dims=()
            )

        if lt_scalar is None:
            raise ValueError(
                "Failed to determine a scalar launch_time (looked for scalar 'launch_time', "
                "or first finite in 'flight_time'/'time', or ds_new.attrs['launch_time'])."
            )

        ds_interp["launch_time"] = lt_scalar

    elif method == "bin":
        # --- bin-mean on alt using numeric labels (no pandas.Interval left) ---
        print("[pysonde] Using vertical interpolation method: bin")
        interpolation_bins = np.arange(
            cfg.level2.setup.interpolation_grid_min - cfg.level2.setup.interpolation_grid_inc / 2,
            cfg.level2.setup.interpolation_grid_max + cfg.level2.setup.interpolation_grid_inc / 2,
            cfg.level2.setup.interpolation_grid_inc,
        )

        # Workaround for xarray#6995 (object dtype time issues)
        if "flight_time" in ds_new:
            ds_new["flight_time"] = ds_new.flight_time.astype(int)

        # Use labels=interpolation_grid so resulting coord is numeric (bin centers)
        ds_interp = ds_new.groupby_bins(
            vertical_interpolation_axis,
            interpolation_bins,
            labels=interpolation_grid,
            restore_coord_dims=True,
        ).mean()

        ds_interp = ds_interp.transpose()
        ds_interp = ds_interp.rename({f"{vertical_interpolation_axis}_bins": vertical_interpolation_axis})

        # Create bounds variable (lower, upper] per CF-ish convention
        ds_interp[f"{vertical_interpolation_axis}_bnds"] = xr.DataArray(
            np.array([interpolation_bins[:-1], interpolation_bins[1:]]).T,
            dims=[vertical_interpolation_axis, "nv"],
            coords={vertical_interpolation_axis: ds_interp[vertical_interpolation_axis].data},
        )

        # carry launch_time forward
        if "launch_time" in ds_new:
            ds_interp["launch_time"] = ds_new["launch_time"]

    elif method == "linear_masked":
        print("[pysonde] Using vertical interpolation method: linear_masked")
        print("[INERPOLATION]: I am going to interpolate over:", vertical_interpolation_axis)
        # --- 1) Do exactly what 'linear' does (preserve flight_time, interp, robust log-p) ---


        # drop NaNs & sort by altitude (same as linear)
        ds_sorted = ds_new.sortby(vertical_interpolation_axis)
        ds_sorted = ds_sorted.where(np.isfinite(ds_sorted[vertical_interpolation_axis]), drop=True)

        # preserve flight_time across interpolation (datetime -> int64 ns -> datetime)
        if "flight_time" in ds_sorted.variables and (vertical_interpolation_axis in ds_sorted["flight_time"].dims):
            ds_lin = ds_sorted.copy()
            ds_lin["flight_time_ns"] = ds_lin["flight_time"].astype("datetime64[ns]").astype("int64")
            ds_lin = ds_lin.drop_vars("flight_time")

            # pre-fill internal NaNs along alt so .interp doesn't propagate gaps
            ds_filled = ds_lin.interpolate_na(dim=vertical_interpolation_axis, method="linear", use_coordinate=True)

            ds_interp = ds_filled.interp(
                {vertical_interpolation_axis: interpolation_grid},
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )
            # convert back to datetime64 and restore the original name
            ds_interp["flight_time"] = ds_interp["flight_time_ns"].astype("datetime64[ns]")
            ds_interp = ds_interp.drop_vars("flight_time_ns")
        else:
            ds_filled = ds_sorted.interpolate_na(dim=vertical_interpolation_axis, method="linear", use_coordinate=True)
            ds_interp = ds_filled.interp(
                {vertical_interpolation_axis: interpolation_grid},
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )
            

        # --- robust log-p interpolation to preserve p(z) profile (aligned with 'linear') ---
        p_name = "pressure" if "pressure" in ds_new else ("p" if "p" in ds_new else None)
        if (p_name is not None) and (vertical_interpolation_axis in ds_new.dims or vertical_interpolation_axis in ds_new.coords):
            p_da = ds_sorted[p_name]
            z_da = ds_sorted[vertical_interpolation_axis]

            # reduce to 1D along alt if needed
            if "sounding" in p_da.dims:
                p_da = p_da.isel(sounding=0)
            if "sounding" in z_da.dims:
                z_da = z_da.isel(sounding=0)

            # numeric arrays with units handled
            p_src = np.asarray(_values_in_unit(p_da, "hPa", default_unit="hPa")).ravel()
            z_src = np.asarray(_values_in_unit(z_da, "m",   default_unit="m")).ravel()

            mfin = np.isfinite(p_src) & np.isfinite(z_src)
            p_src, z_src = p_src[mfin], z_src[mfin]

            if z_src.size >= 2:
                order = np.argsort(z_src)
                z_src, p_src = z_src[order], p_src[order]
                z_out = np.asarray(ds_interp[vertical_interpolation_axis].values).ravel()

                # DEBUG prints (optional)
                # print("\nz_src:", z_src)
                # print("\np_src:", p_src)
                # print("\nz_out:", z_out)
                # print(f"[pysonde] log-p inputs: p_src={p_src.shape}, z_src={z_src.shape}, z_out={z_out.shape}")

                p_out = mh.pressure_interpolation(p_src, z_src, z_out)  # numeric hPa

                # --- minimal endpoint fix for p_out without changing mh.pressure_interpolation ---
                # fill endpoints if they are NaN but lie within the measured span
                zmin, zmax = z_src[0], z_src[-1]  # z_src is sorted above

                def _fill_endpoint(z_target, idx_target):
                    if np.isnan(p_out[idx_target]) and (zmin <= z_target <= zmax):
                        # find bracketing indices in z_src
                        j_hi = np.searchsorted(z_src, z_target, side="left")
                        if 0 < j_hi < len(z_src):
                            j_lo = j_hi - 1
                            z1, z2 = z_src[j_lo], z_src[j_hi]
                            p1, p2 = p_src[j_lo], p_src[j_hi]
                            if np.isfinite(z1) and np.isfinite(z2) and np.isfinite(p1) and np.isfinite(p2) and (z2 > z1):
                                # log-linear interpolation in pressure vs altitude
                                lp = np.interp(z_target, [z1, z2], [np.log(p1), np.log(p2)])
                                p_out[idx_target] = float(np.exp(lp))

                # first bin (often 0 m)
                _fill_endpoint(z_out[0], 0)
                # last bin
                _fill_endpoint(z_out[-1], -1)

                # DEBUG print (optional)
                # print("\np_out (after endpoint fix):", p_out)

                ds_interp[p_name] = xr.DataArray(
                    p_out,
                    dims=(vertical_interpolation_axis,),
                    coords={vertical_interpolation_axis: ds_interp[vertical_interpolation_axis].values},
                    attrs={**ds_new[p_name].attrs, "units": "hPa"},
                )

        # --- ensure pressure has Pint units downstream ---
        p_name = "pressure" if "pressure" in ds_interp else ("p" if "p" in ds_interp else None)
        if p_name is not None:
            ds_interp[p_name] = _safe_to(ds_interp[p_name], "hPa", default_unit="hPa")

        # --- 2) Build per-variable occupancy mask from a temporary bin-mean (to match 'bin' exactly) ---

        interpolation_bins = np.arange(
            cfg.level2.setup.interpolation_grid_min - cfg.level2.setup.interpolation_grid_inc / 2,
            cfg.level2.setup.interpolation_grid_max + cfg.level2.setup.interpolation_grid_inc / 2,
            cfg.level2.setup.interpolation_grid_inc,
        )

        # Use the same source and quirks as the 'bin' branch
        src = ds_new
        if "sounding" in src.dims and src.sizes["sounding"] > 1:
            src = src.isel(sounding=0)

        # Workaround for object dtype times (same as in bin path)
        if "flight_time" in src:
            src = src.copy()
            src["flight_time"] = src["flight_time"].astype(int)

        # Build a temporary bin-mean dataset on the SAME edges/labels as the 'bin' method
        tmp_bin = (
            src.groupby_bins(
                vertical_interpolation_axis,
                interpolation_bins,
                labels=interpolation_grid,
                restore_coord_dims=True,
            )
            .mean()  # skipna=True by default
            .transpose()
            .rename({f"{vertical_interpolation_axis}_bins": vertical_interpolation_axis})
        )

        # Apply per-variable mask: keep values where the tmp_bin mean is non-NaN
        for v in ds_interp.data_vars:
            da = ds_interp[v]
            if (vertical_interpolation_axis in da.dims) and np.issubdtype(da.dtype, np.number) and (v in tmp_bin):
                valid = ~np.isnan(tmp_bin[v])
                ds_interp[v] = da.where(valid)

        # carry launch_time if present
        if "launch_time" in ds_new:
            ds_interp["launch_time"] = ds_new["launch_time"]

    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    return ds_interp
    
'''
#### DEBUGGING Sript that I used with Nina
############################################


def _band_check(ds, axis, vnames, z_lo=17000, z_hi=20000, tag=""):
    z = np.asarray(ds[axis].values, float)
    m = (z >= z_lo) & (z <= z_hi)
    print(f"[BAND {tag}] {z_lo}-{z_hi} m: n_levels={int(m.sum())}")
    for v in vnames:
        if v in ds:
            a = ds[v]
            if "sounding" in a.dims:
                a = a.isel(sounding=0)
            arr = np.asarray(a.values)
            print(f"[BAND {tag}] NaNs {v}: {int(np.isnan(arr[m]).sum())}")


def interpolation(ds_new, method, interpolation_grid, vertical_interpolation_axis, sounding, variables, cfg):
    """
    Interpolate/bin Level-2 dataset along GPS altitude 'alt'.

    Parameters
    ----------
    ds_new : xr.Dataset
        Output of prepare_data_for_interpolation(), with 'alt' as vertical dim.
    method : {"linear","bin"}
        Interpolation strategy.
    interpolation_grid : np.ndarray
        Target altitude grid (centers) in meters.
    sounding : object
        Sounding object (for unitregistry etc).
    variables : iterable[(in_name, out_name)]
        Variable rename mapping for outputs.
    cfg : omegaconf.DictConfig
        Full merged config.

    Returns
    -------
    ds_interp : xr.Dataset
    """
    
    if method == "linear":
        print("[pysonde] Using vertical interpolation method: linear")

        # --- clean & sort only by 'alt' ---
        ds_sorted = ds_new.sortby(vertical_interpolation_axis)
        ds_sorted = ds_sorted.where(np.isfinite(ds_sorted[vertical_interpolation_axis]), drop=True)

        # Interpolate on the target alt grid
        ds_interp = ds_sorted.interp(
            {vertical_interpolation_axis: interpolation_grid},
            method="linear"
            )

        print("pressure before:", ds_sorted.pressure.values)
        # --- robust log-p interpolation to preserve p(z) profile ---
        p_name = "pressure" if "pressure" in ds_new else ("p" if "p" in ds_new else None)
        if (p_name is not None) and (vertical_interpolation_axis in ds_new.dims or vertical_interpolation_axis in ds_new.coords):
            p_da = ds_sorted[p_name]
            z_da = ds_sorted[vertical_interpolation_axis]

            # reduce to 1D along alt if needed
            if "sounding" in p_da.dims:
                p_da = p_da.isel(sounding=0)
            if "sounding" in z_da.dims:
                z_da = z_da.isel(sounding=0)

            # numeric arrays with units handled
            p_src = np.asarray(_values_in_unit(p_da, "hPa", default_unit="hPa")).ravel()
            z_src = np.asarray(_values_in_unit(z_da, "m",   default_unit="m")).ravel()

            mfin = np.isfinite(p_src) & np.isfinite(z_src)
            p_src, z_src = p_src[mfin], z_src[mfin]

            if z_src.size >= 2:
                order = np.argsort(z_src)
                z_src, p_src = z_src[order], p_src[order]
                z_out = np.asarray(ds_interp[vertical_interpolation_axis].values).ravel()

                # print(f"[pysonde] log-p inputs: p_src={p_src.shape}, z_src={z_src.shape}, z_out={z_out.shape}")
                p_out = mh.pressure_interpolation(p_src, z_src, z_out)  # numeric hPa
                # print("\np_out:", p_out)

                # --- minimal endpoint fix for p_out without changing mh.pressure_interpolation ---
                # fill p_out[0] if it's NaN but within the measured span
                zmin, zmax = z_src[0], z_src[-1]
                # NOTE: z_src is sorted above

                def _fill_endpoint(z_target, idx_target):
                    if np.isnan(p_out[idx_target]) and (zmin <= z_target <= zmax):
                        # find bracketing indices in z_src
                        j_hi = np.searchsorted(z_src, z_target, side="left")
                        if 0 < j_hi < len(z_src):
                            j_lo = j_hi - 1
                            z1, z2 = z_src[j_lo], z_src[j_hi]
                            p1, p2 = p_src[j_lo], p_src[j_hi]
                            if np.isfinite(z1) and np.isfinite(z2) and np.isfinite(p1) and np.isfinite(p2) and (z2 > z1):
                                # log-linear in pressure vs altitude
                                lp = np.interp(z_target, [z1, z2], [np.log(p1), np.log(p2)])
                                p_out[idx_target] = float(np.exp(lp))

                # first bin (usually 0 m)
                _fill_endpoint(z_out[0], 0)
                # last bin
                _fill_endpoint(z_out[-1], -1)
                # print("\np_out out:", p_out)

                ds_interp[p_name] = xr.DataArray(
                    p_out,
                    dims=(vertical_interpolation_axis,),
                    coords={vertical_interpolation_axis: ds_interp[vertical_interpolation_axis].values},
                    attrs={**ds_new[p_name].attrs, "units": "hPa"},
                )
        # print("pressure after:", ds_interp.pressure.values)

        # --- ensure pressure has Pint units downstream ---
        p_name = "pressure" if "pressure" in ds_interp else ("p" if "p" in ds_interp else None)
        if p_name is not None:
            ds_interp[p_name] = _safe_to(ds_interp[p_name], "hPa", default_unit="hPa")
        # print("pressure after after:", ds_interp.pressure.values)

        # --- unit harmonization for other variables ---
        for var_in, var_out in variables:
            try:
                ds_interp[var_out] = ds_interp[var_out].pint.quantify(ds_new[var_out].pint.units)
                ds_interp[var_out] = ds_interp[var_out].pint.to(cfg.level2.variables[var_in].attrs.units)
            except (KeyError, ValueError, ConfigAttributeError):
                pass

        # --- guarantee size-1 'sounding' dim (adjust_ds_after_interpolation expects it) ---
        if "sounding" not in ds_interp.dims:
            ds_interp = ds_interp.expand_dims({"sounding": 1})

        # --- ensure scalar launch_time present (0-D datetime64[ns]) ---
        def _first_finite_dt64_scalar(da):
            """Return first finite datetime64[ns] from a time-like DataArray as 0-D."""
            # reduce any extra dims (e.g., sounding/alt) by taking the first finite element
            vals = da.values
            # to ns int for finite mask
            ns = vals.astype("datetime64[ns]").astype("int64")
            m = np.isfinite(ns)
            if m.any():
                return xr.DataArray(vals[m][0].astype("datetime64[ns]"), dims=())
            return None

        lt_scalar = None

        # 1) prefer a scalar 'launch_time' from the input
        if "launch_time" in ds_new and ds_new["launch_time"].ndim == 0:
            lt_scalar = xr.DataArray(
                ds_new["launch_time"].values.astype("datetime64[ns]"), dims=()
            )
        # 2) fallback: derive from per-level time variables in the ORIGINAL input
        elif any(v in ds_new for v in ("flight_time", "time")):
            for cand in ("flight_time", "time"):
                if cand in ds_new:
                    tvar = ds_new[cand]
                    # drop sounding dim if present
                    if "sounding" in tvar.dims:
                        tvar = tvar.isel(sounding=0)
                    # pick first finite timestamp
                    lt_scalar = _first_finite_dt64_scalar(tvar)
                    if lt_scalar is not None:
                        break
        # 3) final fallback: dataset attribute
        if lt_scalar is None and "launch_time" in ds_new.attrs:
            lt_scalar = xr.DataArray(
                np.datetime64(ds_new.attrs["launch_time"]).astype("datetime64[ns]"),
                dims=()
            )

        if lt_scalar is None:
            raise ValueError(
                "Failed to determine a scalar launch_time (looked for scalar 'launch_time', "
                "or first finite in 'flight_time'/'time', or ds_new.attrs['launch_time'])."
            )

        ds_interp["launch_time"] = lt_scalar

    elif method == "bin":
        # --- bin-mean on alt using numeric labels (no pandas.Interval left) ---
        print("[pysonde] Using vertical interpolation method: bin")
        interpolation_bins = np.arange(
            cfg.level2.setup.interpolation_grid_min - cfg.level2.setup.interpolation_grid_inc / 2,
            cfg.level2.setup.interpolation_grid_max + cfg.level2.setup.interpolation_grid_inc / 2,
            cfg.level2.setup.interpolation_grid_inc,
        )

        # Workaround for xarray#6995 (object dtype time issues)
        if "flight_time" in ds_new:
            ds_new["flight_time"] = ds_new.flight_time.astype(int)

        # Use labels=interpolation_grid so resulting coord is numeric (bin centers)
        ds_interp = ds_new.groupby_bins(
            vertical_interpolation_axis,
            interpolation_bins,
            labels=interpolation_grid,
            restore_coord_dims=True,
        ).mean()

        ds_interp = ds_interp.transpose()
        ds_interp = ds_interp.rename({f"{vertical_interpolation_axis}_bins": vertical_interpolation_axis})

        # Create bounds variable (lower, upper] per CF-ish convention
        ds_interp[f"{vertical_interpolation_axis}_bnds"] = xr.DataArray(
            np.array([interpolation_bins[:-1], interpolation_bins[1:]]).T,
            dims=[vertical_interpolation_axis, "nv"],
            coords={vertical_interpolation_axis: ds_interp[vertical_interpolation_axis].data},
        )

        # carry launch_time forward
        if "launch_time" in ds_new:
            ds_interp["launch_time"] = ds_new["launch_time"]

    elif method == "linear_masked":
        print("[pysonde] Using vertical interpolation method: linear_masked")
        print("[INERPOLATION]: I am going to interpolate over:", vertical_interpolation_axis)
        # --- 1) Do exactly what 'linear' does (preserve flight_time, interp, robust log-p) ---

        print("At the top of linear_masked:")
        _band_check(ds_new, vertical_interpolation_axis,
            ["pressure","theta","temperature","specific_humidity"],
            17000, 20000, tag="post-interp")
        
        # drop NaNs & sort by altitude (same as linear)
        ds_sorted = ds_new.sortby(vertical_interpolation_axis)
        ds_sorted = ds_sorted.where(np.isfinite(ds_sorted[vertical_interpolation_axis]), drop=True)

        #plt.plot(ds_new[vertical_interpolation_axis])
        #plt.show()

        # preserve flight_time across interpolation (datetime -> int64 ns -> datetime)
        if "flight_time" in ds_sorted.variables and (vertical_interpolation_axis in ds_sorted["flight_time"].dims):
            print("[flight_time]: datetime -> int64 ns -> datetime")
            ds_lin = ds_sorted.copy()
            ds_lin["flight_time_ns"] = ds_lin["flight_time"].astype("datetime64[ns]").astype("int64")
            ds_lin = ds_lin.drop_vars("flight_time")

            # pre-fill internal NaNs along alt so .interp doesn't propagate gaps
            print(f"{vertical_interpolation_axis}")
            #ds_lin[f"{vertical_interpolation_axis}"].plot()
            #plt.show()

            ds_filled = ds_lin.interpolate_na(dim=vertical_interpolation_axis, method="linear", use_coordinate=True)
            #ds_filled.temperature.plot(y=vertical_interpolation_axis)
            #plt.show()

            ds_interp = ds_filled.interp(
                {vertical_interpolation_axis: interpolation_grid},
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )

            #ds_interp.temperature.plot(y=vertical_interpolation_axis)
            #plt.show()


            # convert back to datetime64 and restore the original name
            ds_interp["flight_time"] = ds_interp["flight_time_ns"].astype("datetime64[ns]")
            ds_interp = ds_interp.drop_vars("flight_time_ns")
        else:
            print("[flight_time: was fine already")
            ds_filled = ds_sorted.interpolate_na(dim=vertical_interpolation_axis, method="linear", use_coordinate=True)
            ds_interp = ds_filled.interp(
                {vertical_interpolation_axis: interpolation_grid},
                method="linear",
                kwargs={"fill_value": "extrapolate"},
            )
            
        print("After interp in linear_masked:")
        _band_check(ds_interp, vertical_interpolation_axis,
            ["pressure","theta","temperature","specific_humidity"],
            17000, 20000, tag="post-interp")
        
        #plt.plot(ds_sorted[vertical_interpolation_axis])
        #plt.show()

        # --- robust log-p interpolation to preserve p(z) profile (aligned with 'linear') ---
        p_name = "pressure" if "pressure" in ds_new else ("p" if "p" in ds_new else None)
        if (p_name is not None) and (vertical_interpolation_axis in ds_new.dims or vertical_interpolation_axis in ds_new.coords):
            print("Popop pipipi")

            #print(ds_sorted)
            p_da = ds_sorted[p_name]
            z_da = ds_sorted[vertical_interpolation_axis]

            #z_da.plot()
            #plt.show()

            # reduce to 1D along alt if needed
            if "sounding" in p_da.dims:
                p_da = p_da.isel(sounding=0)
            if "sounding" in z_da.dims:
                z_da = z_da.isel(sounding=0)

            # numeric arrays with units handled
            p_src = np.asarray(_values_in_unit(p_da, "hPa", default_unit="hPa")).ravel()
            z_src = np.asarray(_values_in_unit(z_da, "m",   default_unit="m")).ravel()

            #(z_src, p_src)
            #plt.show()
            #plt.plot(z_da, marker='o', linestyle=None)
            #plt.show()

            # numeric + positive pressure for log space
            mfin = np.isfinite(p_src) & np.isfinite(z_src) & (p_src > 0)
            p_src, z_src = p_src[mfin], z_src[mfin]

            if z_src.size >= 2:
                order = np.argsort(z_src)
                z_src, p_src = z_src[order], p_src[order]
                z_out = np.asarray(ds_interp[vertical_interpolation_axis].values).ravel()

                # DEBUG prints (optional)
                # print("\nz_src:", z_src)
                # print("\np_src:", p_src)
                # print("\nz_out:", z_out)
                # print(f"[pysonde] log-p inputs: p_src={p_src.shape}, z_src={z_src.shape}, z_out={z_out.shape}")


                p_out = mh.pressure_interpolation(p_src, z_src, z_out)  # numeric hPa

                #plt.plot(z_src, marker='o', linestyle=None)
                #plt.show()

                # --- DIAGNOSTICS for pressure interpolation ---
                # --- DIAGNOSTICS for pressure interpolation ---
                print("[P-DBG] inputs:",
                    f"p_src.shape={p_src.shape}, z_src.shape={z_src.shape}")
                print("[P-DBG] finite counts:", 
                    f"p={np.isfinite(p_src).sum()}/{p_src.size}",
                    f"z={np.isfinite(z_src).sum()}/{z_src.size}")

                # monotonicity / duplicates in z_src (required by many log-p routines)
                dz = np.diff(z_src)
                print("[P-DBG] z_src strictly inc?:", bool(np.all(dz > 0)),
                    " nondecreasing?:", bool(np.all(dz >= 0)),
                    " duplicates:", int(np.sum(dz == 0)))

                # coverage of z_out vs z_src
                zmin_p, zmax_p = float(z_src[0]), float(z_src[-1])
                z_out = np.asarray(ds_interp[vertical_interpolation_axis].values).ravel()
                inside = (z_out >= zmin_p) & (z_out <= zmax_p)
                print("[P-DBG] z_out inside pressure-span:",
                    int(inside.sum()), "/", z_out.size, 
                    f"span=({zmin_p:.1f},{zmax_p:.1f}) out_min={np.nanmin(z_out):.1f} out_max={np.nanmax(z_out):.1f}\n")
                # --- DIAGNOSTICS for pressure interpolation ---
                # --- DIAGNOSTICS for pressure interpolation ---



                # --- minimal endpoint fix for p_out without changing mh.pressure_interpolation ---
                # fill endpoints if they are NaN but lie within the measured span
                zmin, zmax = z_src[0], z_src[-1]  # z_src is sorted above

                def _fill_endpoint(z_target, idx_target):
                    if np.isnan(p_out[idx_target]) and (zmin <= z_target <= zmax):
                        # find bracketing indices in z_src
                        j_hi = np.searchsorted(z_src, z_target, side="left")
                        if 0 < j_hi < len(z_src):
                            j_lo = j_hi - 1
                            z1, z2 = z_src[j_lo], z_src[j_hi]
                            p1, p2 = p_src[j_lo], p_src[j_hi]
                            if np.isfinite(z1) and np.isfinite(z2) and np.isfinite(p1) and np.isfinite(p2) and (z2 > z1):
                                # log-linear interpolation in pressure vs altitude
                                lp = np.interp(z_target, [z1, z2], [np.log(p1), np.log(p2)])
                                p_out[idx_target] = float(np.exp(lp))

                # first bin (often 0 m)
                _fill_endpoint(z_out[0], 0)
                # last bin
                _fill_endpoint(z_out[-1], -1)

                # DEBUG print (optional)
                # print("\np_out (after endpoint fix):", p_out)

                #p_out.plot()
                #plt.show()

                ds_interp[p_name] = xr.DataArray(
                    p_out,
                    dims=(vertical_interpolation_axis,),
                    coords={vertical_interpolation_axis: ds_interp[vertical_interpolation_axis].values},
                    attrs={**ds_new[p_name].attrs, "units": "hPa"},
                )

        #ds_interp[vertical_interpolation_axis].plot()
        #plt.show()

        # --- ensure pressure has Pint units downstream ---
        p_name = "pressure" if "pressure" in ds_interp else ("p" if "p" in ds_interp else None)
        if p_name is not None:
            ds_interp[p_name] = _safe_to(ds_interp[p_name], "hPa", default_unit="hPa")





        print("After log-p interp in linear_masked:")
        _band_check(ds_interp, vertical_interpolation_axis,
            ["pressure","theta","temperature","specific_humidity"],
            17000, 20000, tag="post-interp")
        
        # --- 2) Build per-variable occupancy mask from a temporary bin-mean (to match 'bin' exactly) ---

        interpolation_bins = np.arange(
            cfg.level2.setup.interpolation_grid_min - cfg.level2.setup.interpolation_grid_inc / 2,
            cfg.level2.setup.interpolation_grid_max + cfg.level2.setup.interpolation_grid_inc / 2,
            cfg.level2.setup.interpolation_grid_inc,
        )

        axis = vertical_interpolation_axis

        # Use the same source and quirks as the 'bin' branch
        src = ds_new
        if "sounding" in src.dims and src.sizes["sounding"] > 1:
            src = src.isel(sounding=0)

        # Workaround for object dtype times (same as in bin path)
        if "flight_time" in src:
            src = src.copy()
            src["flight_time"] = src["flight_time"].astype(int)

        # Build a temporary bin-mean dataset on the SAME edges/labels as the 'bin' method
        tmp_bin = (
            src.groupby_bins(
                axis,
                interpolation_bins,
                labels=interpolation_grid,
                restore_coord_dims=True,
            )
            .mean()  # skipna=True by default
            .transpose()
            .rename({f"{axis}_bins": axis})
        )

        # NEW: a pure occupancy (counts) vector by *axis only* (independent of any variable NaNs)
        occ = (
            src[axis]
            .groupby_bins(axis, interpolation_bins, labels=interpolation_grid, restore_coord_dims=False)
            .count()
            .rename({f"{axis}_bins": axis})
        )

        # ---------------------- DIAGNOSTICS ----------------------
        band = slice(17000, 20000)
        occ_band = occ.sel({axis: band})
        zero_occ = (occ_band == 0)
        print("[MASK-DBG] occ zeros in 17–20 km:", int(zero_occ.sum().item()))
        if int(zero_occ.sum().item()) > 0:
            zeros_at = occ_band.where(zero_occ, drop=True)[axis].values
            print("[MASK-DBG] first zero-occ heights in band:", zeros_at[:10])

        for probe in ["pressure", "theta", "temperature", "specific_humidity"]:
            if probe in tmp_bin:
                tb = tmp_bin[probe].sel({axis: band})
                n_nans = int(np.isnan(tb.values).sum())
                print(f"[MASK-DBG] tmp_bin[{probe}] NaN bins in 17–20 km:", n_nans)
        # ---------------------------------------------------------

        # Apply per-variable mask
        for v in list(ds_interp.data_vars):
            da = ds_interp[v]
            if not ((axis in da.dims) and np.issubdtype(da.dtype, np.number)):
                continue

            # choose mask mode
            if v in ("pressure", "p"):
                if v in tmp_bin:
                    valid = ~np.isnan(tmp_bin[v])
                    mode = "pressure-strict(tmp_bin)"
                else:
                    valid = (occ.fillna(0) > 0)
                    mode = "pressure-occupancy"
            else:
                # >>> THIS is the key change: occupancy (axis) only for non-pressure <<<
                valid = (occ.fillna(0) > 0)
                mode = "occupancy-only"

            # restrict to observed span too (optional but nice)
            z = ds_interp[axis]
            zmin = float(np.nanmin(src[axis].values))
            zmax = float(np.nanmax(src[axis].values))
            within_span = (z >= zmin) & (z <= zmax)
            valid = valid & within_span

            # DIAG: alignment & counts
            try:
                vb = valid.sel({axis: slice(17000, 20000)})
                n_drop_band = int((~vb).sum().item())
                n_keep_band = int(vb.sum().item())
            except Exception:
                n_drop_band = n_keep_band = -1

            print(f"[MASK-DBG] mode for {v}: {mode}; "
                f"keep={n_keep_band} drop={n_drop_band} in 17–20 km")

            ds_interp[v] = da.where(valid)


        # carry launch_time if present
        if "launch_time" in ds_new:
            ds_interp["launch_time"] = ds_new["launch_time"]

        print("After masking in linear_masked:")
        _band_check(ds_interp, axis,
                    ["pressure","theta","temperature","specific_humidity"],
                    17000, 20000, tag="post-interp")






    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    return ds_interp


'''

def adjust_ds_after_interpolation(ds_interp, ds, ds_input, vertical_interpolation_axis, variables, cfg):
    """
    Final adjustments to the interpolated sounding dataset.

    This function enhances the interpolated dataset by computing derived variables,
    applying geolocation conversions, reassigning launch time, and restoring physical
    units. It is typically called after vertical interpolation to ensure the dataset
    is complete and consistent.

    Parameters
    ----------
    ds_interp : xarray.Dataset
        The vertically interpolated dataset with variables like wind_u, wind_v, theta, etc.
    ds : xarray.Dataset
        The original input dataset before interpolation (used for auxiliary data).
    ds_input : xarray.Dataset
        A preserved copy of the original dataset before interpolation (e.g., for sorting or coordinate integrity).
    variables : iterable
        Variable mapping (input → output names), typically used for variable-specific unit assignments.
    cfg : omegaconf.DictConfig
        Configuration object containing unit definitions, thresholds, and setup parameters.

    Returns
    -------
    ds_interp : xarray.Dataset
        The updated dataset with:
        - Wind speed and direction
        - Latitude, longitude, and WGS84 altitude (if ECEF input available)
        - Launch time as a single sounding-level variable
        - Recomputed temperature, relative humidity, and dew point
        - Properly assigned physical units for key thermodynamic variables

    Notes
    -----
    - Wind direction and speed are computed from horizontal wind components.
    - If coordinates are provided in ECEF (x, y, z), they are converted to geodetic coordinates.
    - Temperature and RH are recalculated from interpolated theta and specific humidity.
    - Unit handling is performed using `pint` via `xarray` integration.
    """
    dims_2d = ["sounding", vertical_interpolation_axis]
    dims_1d = [vertical_interpolation_axis]
    ureg = ds["lat"].pint.units._REGISTRY
    coords_1d = {vertical_interpolation_axis: ds_interp[vertical_interpolation_axis].pint.quantify("m", unit_registry=ureg)}

    axis_coord = ds_interp[vertical_interpolation_axis]
    if "sounding" in axis_coord.dims:
        axis_coord = axis_coord.isel(sounding=0)
    axis_coord = axis_coord.pint.quantify("m", unit_registry=ureg)
    coords_1d = {vertical_interpolation_axis: axis_coord}

    wind_u = ds_interp.isel({"sounding": 0})["wind_u"]
    wind_v = ds_interp.isel({"sounding": 0})["wind_v"]
    dir, wsp = mh.get_directional_wind(wind_u, wind_v)

    ds_interp["wind_direction"] = xr.DataArray(
        dir.expand_dims({"sounding": 1}).data, dims=dims_2d, coords=coords_1d
    )
    ds_interp["wind_speed"] = xr.DataArray(
        wsp.expand_dims({"sounding": 1}).data, dims=dims_2d, coords=coords_1d
    )

    if "alt_WGS84" in ds.keys():
        print("yes, alt_WGS84 is still here")
        ecef = pyproj.Proj(proj="geocent", ellps="WGS84", datum="WGS84")
        lla = pyproj.Proj(proj="latlong", ellps="WGS84", datum="WGS84")
        lon, lat, alt = pyproj.transform(
            ecef,
            lla,
            ds_interp["x"].values,
            ds_interp["y"].values,
            ds_interp["z"].values,
            radians=False,
        )

        for var, val in {
            "lat": lat,
            "lon": lon,
            "alt_WGS84": alt,
        }.items():
            try:
                ds_interp[var] = xr.DataArray(
                    val, dims=dims_1d, coords=coords_1d
                ).pint.quantify(
                    cfg.level2.variables[var].attrs.units, unit_registry=ureg
                )
            except ConfigKeyError:
                pass
        del ds_interp["x"]
        del ds_interp["y"]
        del ds_interp["z"]
        del ds_interp["alt_WGS84"]
    else:
        print("no, alt_WGS84 is gone")

    ds_input = ds_input.sortby(vertical_interpolation_axis)
    ds_input.alt.load()
    ds_input.p.load()
    ds_input = ds_input.reset_coords()

    # ds_interp['pressure'] = ds_interp['pressure'].pint.to(cfg.level2.variables['p'].attrs.units)
    # ds_interp['pressure'] = ds_interp['pressure'].expand_dims({'sounding':1})

    ds_interp["launch_time"] = xr.DataArray(
        [ds_interp.isel({"sounding": 0}).launch_time.item() / 1e9], dims=["sounding"]
    )

    # Calculations after interpolation
    # Recalculate temperature and relative humidity from theta and q

    temperature = td.calc_T_from_theta(
        _safe_to(ds_interp.isel(sounding=0)["theta"], "K", default_unit="K"),
        _safe_to(ds_interp.isel(sounding=0)["pressure"],    "hPa",  default_unit="hPa"),
    )

    print("theta[:5] =", ds_interp.isel(sounding=0)["theta"].values[:5])
    print("press [:5] =", ds_interp.isel(sounding=0)["pressure"].values[:5])
    print("T_new[:5]  =", temperature.values[:5])

    ds_interp["temperature"] = xr.DataArray(
        temperature.data, dims=dims_1d, coords=coords_1d
    )

    ds_interp["temperature"] = ds_interp["temperature"].expand_dims({"sounding": 1})

    q = ds_interp.isel(sounding=0)["specific_humidity"]
    w = q / (1 - q)

    e_s = td.calc_saturation_pressure(ds_interp.isel(sounding=0)["temperature"], method="wagner_pruss")
    w_s = td.calc_wv_mixing_ratio(ds_interp.isel(sounding=0), e_s)
    #w_s = mpcalc.mixing_ratio(e_s, ds_interp.isel(sounding=0)["pressure"].data)
    relative_humidity = w / w_s * 100

    ds_interp["relative_humidity"] = xr.DataArray(
        relative_humidity.data, dims=dims_1d, coords=coords_1d
    )
    ds_interp["relative_humidity"] = ds_interp["relative_humidity"].expand_dims(
        {"sounding": 1}
    )

    ds_interp["relative_humidity"].data = ds_interp["relative_humidity"].data * units(
        "%"
    )

    dewpoint = td.convert_rh_to_dewpoint(
        ds_interp.isel(sounding=0)["temperature"],
        ds_interp.isel(sounding=0)["relative_humidity"],
    )

    ds_interp["dewpoint"] = xr.DataArray(dewpoint.data, dims=dims_1d, coords=coords_1d)
    ds_interp["dewpoint"] = ds_interp["dewpoint"].expand_dims({"sounding": 1})
    ds_interp["dewpoint"].data = ds_interp["dewpoint"].data * units.K
    ds_interp["mixing_ratio"].data = ds_interp["mixing_ratio"].data * units("g/g")
    ds_interp["specific_humidity"].data = ds_interp["specific_humidity"].data * units(
        "g/g"
    )
    return ds_interp


def count_number_of_measurement_within_bin(ds_interp, ds_new, cfg, interpolation_grid, vertical_interpolation_axis):
    """
    Count the number of original measurements within each vertical interpolation bin 
    and flag how each interpolated value was obtained.

    This function evaluates, for each altitude bin in the interpolated dataset, 
    how many original measurements contributed to the interpolated values. 
    It distinguishes between bins filled via averaging of real data and those filled 
    via interpolation, and stores this information for both PTU (pressure, temperature, 
    humidity) and GPS-related variables.

    Parameters
    ----------
    ds_interp : xarray.Dataset
        The interpolated dataset containing altitude as a dimension.
    ds_new : xarray.Dataset
        The pre-interpolated dataset with original measurements, including 
        pressure and GPS data (latitude).
    cfg : omegaconf.DictConfig
        Configuration object with setup parameters including bin spacing and altitude range.
    interpolation_grid : np.ndarray
        The target vertical grid used for interpolation (usually bin centers).

    Returns
    -------
    ds_interp : xarray.Dataset
        Updated dataset with four new data variables:
        - 'N_ptu': number of PTU measurements per altitude bin
        - 'N_gps': number of GPS measurements per altitude bin
        - 'm_ptu': method flag for PTU values
            * 0: no data
            * 1: value was interpolated
            * 2: value was averaged from real data
        - 'm_gps': same as above, for GPS variables

    Notes
    -----
    - Altitude bins are created based on interpolation grid boundaries.
    - `groupby_bins(...).count()` is used to determine how many measurements fall within each bin.
    - Method flags help identify whether values are directly measured or interpolated.
    - This is useful for quality control and metadata tracking in processed soundings.
    """
    interpolation_bins = np.arange(
        cfg.level2.setup.interpolation_grid_min
        - cfg.level2.setup.interpolation_grid_inc / 2,
        cfg.level2.setup.interpolation_grid_max
        + cfg.level2.setup.interpolation_grid_inc / 2,
        cfg.level2.setup.interpolation_grid_inc,
    )

    # Count number of measurements within each bin
    dims_2d = ["sounding", f"{vertical_interpolation_axis}"]
    coords_1d = {f"{vertical_interpolation_axis}": ds_interp[vertical_interpolation_axis]}

    ds_interp["N_ptu"] = xr.DataArray(
        ds_new.pressure.groupby_bins(
            f"{vertical_interpolation_axis}",
            interpolation_bins,
            labels=interpolation_grid,
            restore_coord_dims=True,
        )
        .count()
        .values,
        dims=dims_2d,
        coords=coords_1d,
    )
    ds_interp["N_gps"] = xr.DataArray(
        ds_new.latitude.groupby_bins(
            f"{vertical_interpolation_axis}",
            interpolation_bins,
            labels=interpolation_grid,
            restore_coord_dims=True,
        )
        .count()
        .values,
        dims=dims_2d,
        coords=coords_1d,
    )

    # Cell method used
    data_exists = np.where(np.isnan(ds_interp.isel(sounding=0).pressure), False, True)
    data_mean = np.where(
        np.isnan(ds_interp.isel(sounding=0).N_ptu), False, True
    )  # no data or interp: nan; mean > 0
    data_method = np.zeros_like(data_exists, dtype="uint")
    data_method[np.logical_and(data_exists, data_mean)] = 2
    data_method[np.logical_and(data_exists, ~data_mean)] = 1
    ds_interp["m_ptu"] = xr.DataArray([data_method], dims=dims_2d, coords=coords_1d)
    ds_interp["N_ptu"].values[0, np.logical_and(~data_mean, data_method > 0)] = 0

    data_exists = np.where(np.isnan(ds_interp.isel(sounding=0).latitude), False, True)
    data_mean = np.where(
        np.isnan(ds_interp.isel(sounding=0).N_gps), False, True
    )  # no data or interp: nan; mean > 0
    data_method = np.zeros_like(data_exists, dtype="uint")
    data_method[np.logical_and(data_exists, data_mean)] = 2
    data_method[np.logical_and(data_exists, ~data_mean)] = 1
    ds_interp["m_gps"] = xr.DataArray([data_method], dims=dims_2d, coords=coords_1d)
    ds_interp["N_gps"].values[0, np.logical_and(~data_mean, data_method > 0)] = 0

    return ds_interp


def finalize_attrs(ds_interp, ds, cfg, file, variables, vertical_interpolation_axis):
    """
    Finalize metadata and attributes of the interpolated sounding dataset.

    This function performs the final adjustments to `ds_interp` before exporting,
    including:
    - Converting launch time from numerical timestamps to datetime objects
    - Assigning ascent/descent flag based on vertical motion
    - Copying and formatting global attributes and variable metadata
    - Transposing variables to ensure correct dimension order
    - Replacing template placeholders in attribute strings

    Parameters
    ----------
    ds_interp : xarray.Dataset
        The interpolated sounding dataset to finalize. Assumes fields like
        `launch_time`, `ascent_rate`, and `sounding` are present.
    ds : xarray.Dataset
        The original dataset before interpolation. Used to inherit attributes.
    cfg : omegaconf.DictConfig
        Configuration object containing global and variable-level metadata.
    file : pathlib.Path or str
        Path to the source file used for generating `ds_interp`.
    variables : iterable of tuples
        List of variable mappings (input_var, output_var) used to assign metadata.

    Returns
    -------
    ds_interp : xarray.Dataset
        The finalized dataset with:
        - Converted launch time (to datetime64)
        - Ascent/descent flag (`ascent_flag`)
        - Fully populated global and variable attributes
        - Correct dimension ordering
        - Source file name stored in `attrs["source"]`

    Notes
    -----
    - `ascent_flag` is set to 0 for descending and 1 for ascending profiles,
      depending on which type of vertical motion dominates.
    - Attributes with string placeholders (e.g., `{platform}`) are replaced using
      the dataset's own attributes. If a placeholder is missing, a warning is printed.
    - Variable attributes are assigned based on the configuration (`cfg.level2.variables`).
    """
    import pandas as pd
    from netCDF4 import num2date

    def convert_num2_date_with_nan(num, format):
        if not np.isnan(num):
            return num2date(
                num,
                format,
                only_use_python_datetimes=True,
                only_use_cftime_datetimes=False,
            )
        else:
            return pd.NaT

    convert_nums2date = np.vectorize(convert_num2_date_with_nan)

    # ds_interp['flight_time'].data = convert_nums2date(ds_interp.flight_time.data, "seconds since 1970-01-01")
    ds_interp["launch_time"].data = convert_nums2date(
        ds_interp.launch_time.data, "seconds since 1970-01-01"
    )

    # direction = get_direction(ds_interp, ds)
    most_common_vertical_movement = np.argmax(
        [
            np.count_nonzero(ds_interp.ascent_rate > 0),
            np.count_nonzero(ds_interp.ascent_rate < 0),
        ]
    )
    ds_interp["ascent_flag"] = xr.DataArray(
        [most_common_vertical_movement], dims=["sounding"]
    )

    # # Copy trajectory id from level1 dataset
    # ds_interp['sounding'] = xr.DataArray([ds['sounding'].values])#, dims=['sounding'])
    ds_interp.sounding.attrs = ds["sounding"].attrs

    # merged_conf = OmegaConf.merge(config.level2, meta_data_cfg)
    # merged_conf._set_parent(OmegaConf.merge(config, meta_data_cfg))
    ds_interp.attrs = ds.attrs

    # Replace placeholders in global attributes
    for key, value in ds_interp.attrs.items():
        if isinstance(value, str):  # Only process string attributes
            try:
                ds_interp.attrs[key] = value.format(**ds_interp.attrs)
            except KeyError as e:
                print(f"Warning: Placeholder {e} in attribute '{key}' could not be replaced.")


    ds_interp = h.replace_global_attributes(ds_interp, cfg)
    ds_interp.attrs["source"] = str(file).split("/")[-1]

    ds_interp = h.set_additional_var_attributes(
        ds_interp, cfg.level2.variables, variables
    )

    # Transpose dataset if necessary
    for variable in ds_interp.data_vars:
        if variable == f"{vertical_interpolation_axis}_bnds":
            continue
        dims = ds_interp[variable].dims
        if (len(dims) == 2) and (dims[0] != "sounding"):
            ds_interp[variable] = ds_interp[variable].T

    # time_dt = pd.Timestamp(np.datetime64(ds_interp.isel({'sounding': 0}).launch_time.data.astype("<M8[ns]")))

    return ds_interp


def export(output_fmt, ds_interp, cfg):
    """Saves sounding to disk"""

    if ds_interp.ascent_flag.values[0] == 0:
        direction = "AscentProfile"
    elif ds_interp.ascent_flag.values[0] == 1:
        direction = "DescentProfile"

    # time_fmt = time_dt.strftime('%Y%m%dT%H%M')
    outfile = output_fmt.format(
        platform=cfg.main.platform,
        campaign=cfg.main.campaign,
        campaign_id=cfg.main.campaign_id,
        direction=direction,
        version=cfg.main.data_version,
        level="2",
    )
    launch_time = pd.to_datetime(ds_interp.launch_time.item(0))
    outfile = launch_time.strftime(outfile)
    directory = os.path.dirname(outfile)
    Path(directory).mkdir(parents=True, exist_ok=True)

    logging.info("Write output to {}".format(outfile))
    h.write_dataset(ds_interp, outfile)
