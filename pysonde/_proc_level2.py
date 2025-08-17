import logging
import os
from pathlib import Path

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


def prepare_data_for_interpolation(ds, uni, variables, reader=pysondeL1):
    # Wind components from direction + speed
    u, v = mh.get_wind_components(ds.wdir, ds.wspd)
    ds["u"] = xr.DataArray(u.data, dims=["level"])
    ds["v"] = xr.DataArray(v.data, dims=["level"])

    # Optional: Cartesian coordinates if WGS84 altitude is present
    if "alt_WGS84" in ds.keys():
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
        if "alt" in ds and "alt" not in ds.coords:
            ds = ds.set_coords("alt")

        # preferred path: swap_dims when 'alt' is a 1-D coord over 'level'
        if "alt" in ds.coords and ds["alt"].dims == ("level",):
            ds = ds.swap_dims({"level": "alt"})
        else:
            # fallback: if an 'alt' data_var conflicts, rename it temporarily, then rename the dim
            if "alt" in ds.variables and "alt" not in ds.dims:
                ds = ds.rename({"alt": "alt_var"})
            ds = ds.rename({"level": "alt"})
    # ----------------------------------------------------------------

    # Thermodynamics on the dataset now using 'alt' as the vertical dim
    td.metpy.units.units = uni
    theta = td.calc_theta_from_T(ds["ta"], ds["p"])

    # Saturation vapor pressure (Wagner–Pruss), mixing ratio and specific humidity
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
    if "height_ptu" in ds:
        ds_new["height_ptu"] = ds["height_ptu"]

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

def interpolation(ds_new, method, interpolation_grid, sounding, variables, cfg):
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
    import numpy as np
    import xarray as xr
    from omegaconf.errors import ConfigAttributeError

    if method == "linear":
        # --- linear interpolation on numeric 'alt' ---
        # Ensure we drop NaNs for a clean 1D interpolation
        ds_lin = ds_new.dropna(dim="alt", how="any")
        ds_interp = ds_lin.interp(alt=interpolation_grid)

        # Logarithmic pressure interpolation to preserve p(z) profile
        pres_int_p = ds_new.pressure.pint.to("hPa").values[0]
        # 'alt' may be a coordinate without pint; handle both cases
        try:
            pres_int_a = ds_new["alt"].pint.to("m").values[0]
        except Exception:
            pres_int_a = ds_new["alt"].values[0] * sounding.unitregistry("m")

        dims_1d = ["alt"]
        coords_1d = {"alt": ds_interp.alt.data}
        alt_out = ds_interp.alt.values
        interp_pres = mh.pressure_interpolation(
            pres_int_p, pres_int_a, alt_out
        ) * sounding.unitregistry("hPa")
        ds_interp["pressure"] = xr.DataArray(interp_pres, dims=dims_1d, coords=coords_1d)
        ds_interp["pressure"] = ds_interp["pressure"].expand_dims({"sounding": 1})

        # Unit harmonization for other variables
        for var_in, var_out in variables:
            try:
                ds_interp[var_out] = ds_interp[var_out].pint.quantify(ds_new[var_out].pint.units)
                ds_interp[var_out] = ds_interp[var_out].pint.to(cfg.level2.variables[var_in].attrs.units)
            except (KeyError, ValueError, ConfigAttributeError):
                # some vars may be unitless or absent
                pass

    elif method == "bin":
        # --- bin-mean on alt using numeric labels (no pandas.Interval left) ---
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
            "alt",
            interpolation_bins,
            labels=interpolation_grid,
            restore_coord_dims=True,
        ).mean()

        # xarray returns 'alt_bins' -> rename to 'alt' (numeric coord already)
        ds_interp = ds_interp.transpose()
        ds_interp = ds_interp.rename({"alt_bins": "alt"})

        # Create bounds variable (lower, upper] per CF-ish convention
        ds_interp["alt_bnds"] = xr.DataArray(
            np.array([interpolation_bins[:-1], interpolation_bins[1:]]).T,
            dims=["alt", "nv"],
            coords={"alt": ds_interp.alt.data},
        )

        # carry launch_time forward
        if "launch_time" in ds_new:
            ds_interp["launch_time"] = ds_new["launch_time"]

    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    return ds_interp



def adjust_ds_after_interpolation(ds_interp, ds, ds_input, variables, cfg):
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
    dims_2d = ["sounding", "alt"]
    dims_1d = ["alt"]
    ureg = ds["lat"].pint.units._REGISTRY
    coords_1d = {"alt": ds_interp.alt.pint.quantify("m", unit_registry=ureg)}

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

    ds_input = ds_input.sortby("alt")
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
        ds_interp.isel(sounding=0)["theta"].pint.to("K"),
        ds_interp.isel(sounding=0)["pressure"].pint.to("hPa"),
    )

    ds_interp["temperature"] = xr.DataArray(
        temperature.data, dims=dims_1d, coords=coords_1d
    )
    ds_interp["temperature"] = ds_interp["temperature"].expand_dims({"sounding": 1})

    w = (ds_interp.isel(sounding=0)["specific_humidity"]) / (
        1 - ds_interp.isel(sounding=0)["specific_humidity"]
    )
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
    # ds_interp = ds_interp.drop('dew_point')

    # ds_interp = ds_interp.drop('altitude')

    ds_interp["mixing_ratio"].data = ds_interp["mixing_ratio"].data * units("g/g")
    ds_interp["specific_humidity"].data = ds_interp["specific_humidity"].data * units(
        "g/g"
    )

    return ds_interp


def count_number_of_measurement_within_bin(ds_interp, ds_new, cfg, interpolation_grid):
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
    dims_2d = ["sounding", "alt"]
    coords_1d = {"alt": ds_interp.alt}

    ds_interp["N_ptu"] = xr.DataArray(
        ds_new.pressure.groupby_bins(
            "alt",
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
            "alt",
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


def finalize_attrs(ds_interp, ds, cfg, file, variables):
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
        if variable == "alt_bnds":
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
