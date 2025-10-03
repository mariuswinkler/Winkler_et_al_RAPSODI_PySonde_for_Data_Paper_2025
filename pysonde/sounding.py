"""Sounding class"""

import copy
import logging
import os
from pathlib import Path

from . import _dataset_creator as dc
import _helpers as h
import numpy as np
import pandas as pd
import pint_pandas
import pint_xarray
import thermodynamics as td
import xarray as xr
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError
from moist_thermodynamics import constants as mtc

logging.debug(f"Pint_xarray version:{pint_xarray.__version__}")


class SondeTypeNotImplemented(Exception):
    pass


class Sounding:
    """Sounding class with processing functions"""

    def __init__(self, profile=None, meta_data={}, config=None, ureg=None):
        if profile is None:
            self.profile = None
        else:
            self.profile = profile.copy(deep=True)
        self.level0_reader = None
        self.meta_data = meta_data
        self.config = config
        self.unitregistry = ureg

    def split_by_direction(self, method="maxHeight"):
        """Split sounding into ascending and descending branch"""
        # Simple approach
        sounding_ascent = copy.deepcopy(self)
        sounding_descent = copy.deepcopy(self)
        sounding_ascent.profile = self.profile.loc[self.profile.Dropping == 0]
        sounding_descent.profile = self.profile.loc[self.profile.Dropping == 1]

        # Bugfix 17
        if method == "maxHeight":
            for s, (sounding, func) in enumerate(
                zip(
                    (sounding_ascent.profile, sounding_descent.profile),
                    (np.greater_equal, np.less_equal),
                )
            ):
                #if len(sounding) < 2:
                #    continue
                window_size = 5
                if len(sounding) < window_size:
                    logging.warning(f"Skipping profile {s} due to insufficient height data (len={len(sounding)})")
                    continue

                smoothed_heights = np.convolve(
                    sounding.height, np.ones((window_size,)) / window_size, mode="valid"
                )

                if len(smoothed_heights) < 2:
                    logging.warning(f"Skipping gradient computation due to insufficient smoothed heights (len={len(smoothed_heights)})")
                    continue

                smoothed_heights = np.convolve(
                    sounding.height, np.ones((window_size,)) / window_size, mode="valid"
                )
                if not np.all(func(np.gradient(smoothed_heights), 0)):
                    total = len(sounding.height)
                    nb_diff = total - np.sum(func(np.gradient(sounding.height), 0))
                    logging.warning(
                        "Of {} observations, {} observations have an inconsistent "
                        "sounding direction".format(total, nb_diff)
                    )
                    # Find split time for ascending and descending sounding by maximum height
                    # instead of relying on Dropping variable
                    logging.warning(
                        "Calculate bursting of balloon from maximum geopotential height"
                    )
                    idx_max_hgt = np.argmax(self.profile.height)

                    sounding_ascent.profile = self.profile.iloc[0 : idx_max_hgt + 1]
                    sounding_descent.profile = self.profile.iloc[idx_max_hgt + 1 :]
            
        sounding_ascent.meta_data["sounding_direction"] = "ascent"
        sounding_descent.meta_data["sounding_direction"] = "descent"

        return sounding_ascent, sounding_descent

    def convert_sounding_df2ds(self):
        unit_dict = {}
        profile_copy = self.profile.copy()

        for var in profile_copy.columns:
            if isinstance(profile_copy[var].dtype, pint_pandas.pint_array.PintType):
                unit_dict[var] = profile_copy[var].pint.units
                profile_copy[var] = profile_copy[var].pint.magnitude

        # Convert to xarray Dataset
        self.profile = xr.Dataset.from_dataframe(profile_copy)

        if self.unitregistry is not None:
            self.unitregistry.force_ndarray_like = True

        for var, unit in unit_dict.items():
            self.profile[var].attrs["units"] = unit.__str__()
        self.profile = self.profile.pint.quantify(unit_registry=self.unitregistry)

    def calc_ascent_rate(self):
        """
        Calculate ascent rate

        negative if sonde is falling
        """
        time_delta = np.diff(self.profile.flight_time) / np.timedelta64(1, "s")
        height_delta = np.diff(self.profile.height)
        ascent_rate = height_delta / time_delta
        ascent_rate_ = np.concatenate(([0], ascent_rate))  # 0 at first measurement
        return ascent_rate_

    def calc_temporal_resolution(self):
        """
        Calculate temporal resolution of sounding

        Returns the most common temporal resolution
        by calculating the temporal differences
        and returning the most common difference.

        Input
        -----
        sounding : obj
            sounding class containing flight time
            information

        Return
        ------
        temporal_resolution : float
            temporal resolution
        """
        time_differences = np.abs(
            np.diff(np.ma.compressed(self.profile.flight_time))
        ) / np.timedelta64(1, "s")
        time_differences_counts = np.bincount(time_differences.astype(int))
        most_common_diff = np.argmax(time_differences_counts)
        temporal_resolution = most_common_diff
        self.meta_data["temporal_resolution"] = temporal_resolution

    def generate_location_coord(self):
        """Generate unique id of sounding"""
        lat = self.profile.latitude.values[0]
        if lat > 0:
            lat = "{:04.1f}".format(lat)
        else:
            lat = "{:05.1f}".format(lat)

        lon = self.profile.longitude.values[0]
        if lon > 0:
            lon = "{:04.1f}".format(lon)
        else:
            lon = "{:05.1f}".format(lon)

        loc = str(lat) + "N" + str(lon) + "E"
        self.meta_data["location_coord"] = loc

    def generate_sounding_id(self, config):
        """Generate unique id of sounding"""
        id = config.level1.variables.sounding.format.format(
            direction=self.meta_data["sounding_direction"],
            lat=self.profile.latitude.values[0],
            lon=self.profile.longitude.values[0],
        )
        id = self.meta_data["launch_time_dt"].strftime(id)
        self.meta_data["sounding"] = id

    def get_sonde_type(self):
        """Get sonde type"""
        if self.level0_reader == "MW41":
            # Check if "SondeTypeName" exists in meta_data
            if "SondeTypeName" in self.meta_data:
                if self.meta_data["SondeTypeName"] == "RS41-SGP":
                    self.meta_data["sonde_type"] = "123"
                else:
                    raise SondeTypeNotImplemented(
                        "SondeTypeName {} is not implemented".format(
                            self.meta_data["SondeTypeName"]
                        )
                    )
        elif self.level0_reader == "METEOMODEM":
            logging.warning("Sonde type for METEOMODEM is assumed to be 163")
            self.meta_data["sonde_type"] = "163"

    def get_sonde_serial_number(self):
        """Get sonde serial number"""

        if self.level0_reader == "MW41":
            # Check if "SerialNbr" exists in meta_data
            if "SerialNbr" in self.meta_data:
                self.meta_data["sonde_serial_number"] = self.meta_data["SerialNbr"]
            else:
                logging.warning("Serial number missing in meta_data for MW41.")
                self.meta_data["sonde_serial_number"] = np.nan
        elif self.level0_reader == "METEOMODEM":
            logging.warning("METEOMODEM does not have a serial number. Assigning NaN.")
            self.meta_data["sonde_serial_number"] = np.nan

    def calculate_additional_variables(self, config):
        """Calculation of additional variables"""
        # Ascent rate
        ascent_rate = self.calc_ascent_rate()
        if "ascent_rate" in self.profile:
            logging.warning(
                "Values for ascent rate already exist in input file. To ensure consistency, they will be recalculated."
            )
            diff = np.mean(self.profile.ascent_rate - ascent_rate)
            logging.info(
                f"Mean difference between calculated and existing ascent rate: {diff:.2f}"
            )
            self.profile.ascent_rate = ascent_rate
        else:
            self.profile.insert(10, "ascent_rate", ascent_rate)



        '''
        # Dew point temperature
        dewpoint = td.convert_rh_to_dewpoint(
            self.profile.temperature.values, self.profile.humidity.values
        )
        if "dew_point" in self.profile:
            logging.warning(
                "Values for dew point already exist in input file. To ensure consistency, they will be recalculated."
            )
            diff = np.mean(self.profile.dew_point - dewpoint)
            assert (
                np.abs(diff).magnitude < 50
            ), "The difference seems to be large. Are the input units in the config correct?"
            logging.info(
                f"Mean difference between calculated and existing dew point: {diff:.2f}"
            )
            self.profile.dew_point = dewpoint
        else:
            self.profile.insert(10, "dew_point", dewpoint)
        '''
        # Dew point temperature (normalize to plain float °C; NA-safe)
        dewpoint = td.convert_rh_to_dewpoint(
            self.profile.temperature.values, self.profile.humidity.values
        )

        def _to_float_degC(x):
            import numpy as np
            # If it's a pint Quantity
            try:
                import pint
                if isinstance(x, pint.Quantity):
                    try:
                        return x.to("degC").magnitude.astype("float64")
                    except Exception:
                        return (x.to("kelvin").magnitude.astype("float64") - 273.15)
            except Exception:
                pass
            # If it's an array-like (possibly pandas / numpy)
            arr = getattr(x, "values", x)
            arr = np.asarray(arr, dtype="float64")
            # Heuristic: if looks like Kelvin, convert to °C
            with np.errstate(invalid="ignore"):
                if np.nanmax(arr) > 200.0:
                    arr = arr - 273.15
            return arr

        dp_new = _to_float_degC(dewpoint)

        if "dew_point" in self.profile:
            logging.warning(
                "Values for dew point already exist in input file. To ensure consistency, they will be recalculated."
            )

            dp_old = _to_float_degC(self.profile["dew_point"])
            m = np.isfinite(dp_old) & np.isfinite(dp_new)
            if m.any():
                diff = float(np.nanmean(dp_old[m] - dp_new[m]))  # °C
                if not (abs(diff) < 50):
                    raise AssertionError("The difference seems to be large. Are the input units in the config correct?")
                logging.info(f"Mean difference between calculated and existing dew point: {diff:.2f} °C")
            else:
                logging.warning("No overlapping finite values to compare existing vs recalculated dew point.")

            # Overwrite safely (avoid SettingWithCopy)
            self.profile.loc[:, "dew_point"] = dp_new
        else:
            # First-time write in °C
            self.profile.insert(10, "dew_point", dp_new)











        # Mixing ratio
        e_s = td.calc_saturation_pressure(self.profile.temperature.values, method="wagner_pruss")
        if "pint" in e_s.dtype.__str__():
            mixing_ratio = (
                td.calc_wv_mixing_ratio(self.profile, e_s)
                * self.profile.humidity.values
            )
        else:
            mixing_ratio = (
                td.calc_wv_mixing_ratio(self.profile, e_s)
                * self.profile.humidity.values
                / 100.0
            )
        self.profile.insert(10, "mixing_ratio", mixing_ratio)
        self.meta_data["launch_time_dt"] = self.profile.flight_time.iloc[0]
        # Resolution
        self.calc_temporal_resolution()
        # Location
        self.generate_location_coord()
        # Sounding ID
        self.generate_sounding_id(config)
        self.get_sonde_type()
        self.get_sonde_serial_number()

    def collect_config(self, config, vertical_interpolation_axis: str, level):
        level_dims = {1: "flight_time", 2: f"{vertical_interpolation_axis}"}
        runtime_cfg = OmegaConf.create(
            {
                "runtime": {
                    "sounding_dim": 1,
                    "level_dim": len(self.profile[level_dims[level]]),
                }
            }
        )
        meta_data_cfg = OmegaConf.create(
            {"meta_level0": h.remove_nontype_keys(self.meta_data, type("str"))}
        )

        merged_conf = OmegaConf.merge(
            config[f"level{level}"], meta_data_cfg, runtime_cfg
        )
        merged_conf._set_parent(OmegaConf.merge(config, meta_data_cfg, runtime_cfg))
        return merged_conf

    def isquantity(self, ds):
        return ds.pint.units is not None

    def set_unset_items(self, ds, unset_vars, config, level):
        for var_out, var_int in unset_vars.items():
            if var_int == "launch_time":
                ds[var_out].data = [self.meta_data["launch_time_dt"]]
            elif var_int == "sounding":
                try:
                    ds[var_out].data = [self.meta_data["sounding"]]
                except ValueError:
                    ds = ds.assign_coords({var_out: [self.meta_data["sounding"]]})
            elif var_int == "platform":
                ds[var_out].data = [config.main.platform_number]
            ds[var_out].attrs = (
                config[f"level{level}"].variables[var_out].get("attrs", {})
            )
        return ds

    def set_coordinate_data(self, ds, coords, config):
        unset_coords = {}
        for k in ds.coords.keys():
            try:
                int_var = config.coordinates[k].internal_varname
            except ConfigAttributeError:
                logging.debug(f"{k} does not seem to have an internal varname")
                continue
            if int_var not in self.profile:
                logging.warning(f"No data for output variable {k} found in input.")
                unset_coords[k] = int_var
                pass
            elif self.isquantity(
                self.profile[int_var]
            ):  # convert values to output unit
                ds = ds.assign_coords(
                    {
                        k: self.profile[int_var]
                        .pint.to(ds[k].attrs["units"])
                        .pint.magnitude
                    }
                )
            else:
                ds = ds.assign_coords({k: self.profile[int_var].pint.dequantify()})
            coord_dtype = config.coordinates[k].get("encodings")
            if coord_dtype is not None:
                coord_dtype = coord_dtype.get("dtype")
            if coord_dtype is not None:
                ds[k].encoding["dtype"] = coord_dtype
        return ds, unset_coords


    def create_dataset(self, config, vertical_interpolation_axis: str, level=1):
        merged_conf = self.collect_config(config, vertical_interpolation_axis, level)

        # --- inject runtime dims right before creating the dataset ---
        from omegaconf import OmegaConf

        axis = vertical_interpolation_axis #"height" if (("height" in self.profile.dims) or ("height" in self.profile.coords)) else "alt"
        height_coord = np.asarray(self.profile[axis].values).tolist()  # use your actual grid

        n_soundings = int(self.profile.sizes.get("sounding", 1))

        runtime_conf = OmegaConf.create({
            "runtime": {
                "sounding_dim": n_soundings,
                "height_coord": height_coord,   # <— swapped in YAML above
            }
        })

        merged_conf = OmegaConf.merge(merged_conf, runtime_conf)
        ds = dc.create_dataset(merged_conf)

        # -------------------------------------------------------------

        # Fill dataset with data
        unset_vars = {}

        # Ensure every profile var has a 'sounding' dim for consistent broadcasting
        for var in self.profile.data_vars:
            if var == "alt_bnds":
                continue
            if "sounding" not in self.profile[var].dims:
                self.profile[var] = self.profile[var].expand_dims({"sounding": 1})

        for k in ds.data_vars.keys():
            try:
                int_var = config[f"level{level}"].variables[k].internal_varname
            except ConfigAttributeError:
                logging.debug(f"{k} does not seem to have an internal varname")
                continue
            dims = ds[k].dims

            if k == "launch_time":
                try:
                    ds[k].data = self.profile[int_var].values
                except KeyError:
                    unset_vars[k] = int_var
                continue
            elif k == "platform":
                continue  # will be set at later stage

            # If source missing: write NaNs with exact target shape
            if int_var not in self.profile.keys():
                shape = tuple(ds.sizes[d] for d in ds[k].dims)
                ds[k].data = np.full(shape, np.nan, dtype=float)
                continue

            # Units-aware / shape-aware assignment
            if self.isquantity(self.profile[int_var]):
                data = self.profile[int_var].pint.to(ds[k].attrs.get("units", "")).pint.magnitude
                if len(dims) > 1 and "sounding" == dims[1]:
                    ds[k].data = np.array(data).T
                else:
                    ds[k].data = data
            else:
                if len(dims) > 1 and "sounding" == dims[1]:
                    ds[k].data = np.array(self.profile[int_var].values).T
                else:
                    ds[k].data = self.profile[int_var].values

        ds, unset_coords = self.set_coordinate_data(ds, ds.coords, config[f"level{level}"])
        unset_items = {**unset_vars, **unset_coords}
        ds = self.set_unset_items(ds, unset_items, config, level)
        merged_conf = h.replace_placeholders_cfg(self, merged_conf)

        logging.debug("Add global attributes")
        if "global_attrs" in merged_conf.keys():
            _cfg = h.remove_missing_cfg(merged_conf["global_attrs"])
            ds.attrs = _cfg




        # --- Overwrite ds['height'] with PTU result if available in the profile (DF or XR.Dataset) ---
        def _profile_has(name):
            if isinstance(self.profile, pd.DataFrame):
                return name in self.profile.columns
            if isinstance(self.profile, xr.Dataset):
                return (name in self.profile.variables) or (name in self.profile.coords)
            return False

        def _profile_values(name, dtype="float32"):
            if isinstance(self.profile, pd.DataFrame):
                return np.asarray(self.profile[name].values, dtype=dtype)
            elif isinstance(self.profile, xr.Dataset):
                return np.asarray(self.profile[name].values, dtype=dtype)
            else:
                raise TypeError("self.profile must be a pandas.DataFrame or xarray.Dataset")

        def _ensure_1d(v: np.ndarray) -> np.ndarray:
            # (1, L) -> (L,), (L,) -> (L,)
            if v.ndim == 2 and v.shape[0] == 1:
                return v[0]
            return v.squeeze()

        def _set_height_with_dims(ds: xr.Dataset, vals_1d: np.ndarray, n_snd: int, n_lvl: int):
            """Replace ds['height'] respecting its current dims (if present)."""
            if "height" in ds:
                dims_h = ds["height"].dims
            else:
                # if not present, prefer ('level',)
                dims_h = ("level",)

            if dims_h == ("level",):
                if vals_1d.shape[0] != n_lvl:
                    raise ValueError(f"'height' length {vals_1d.shape[0]} != level dim {n_lvl}")
                ds["height"] = (("level",), vals_1d)

            elif dims_h == ("sounding", "level"):
                if vals_1d.shape[0] != n_lvl:
                    raise ValueError(f"'height' length {vals_1d.shape[0]} != level dim {n_lvl}")
                arr = vals_1d[None, :] if n_snd == 1 else np.broadcast_to(vals_1d[None, :], (n_snd, n_lvl))
                ds["height"] = (("sounding", "level"), arr)

            elif dims_h == ("level", "sounding"):
                if vals_1d.shape[0] != n_lvl:
                    raise ValueError(f"'height' length {vals_1d.shape[0]} != level dim {n_lvl}")
                arr = vals_1d[:, None] if n_snd == 1 else np.broadcast_to(vals_1d[:, None], (n_lvl, n_snd))
                ds["height"] = (("level", "sounding"), arr)

            else:
                # last resort: attach to last dim if lengths match
                last_dim = ds["height"].dims[-1] if "height" in ds else list(ds.dims)[-1]
                if ds.dims[last_dim] == vals_1d.shape[-1]:
                    ds["height"] = ((last_dim,), vals_1d)
                else:
                    raise ValueError(f"Cannot align 'height' to ds dims: {ds.dims} with vals {vals_1d.shape}")

        if _profile_has("height_ptu"):
            vals = _profile_values("height_ptu", dtype="float32")
            vals = np.asarray(vals, dtype="float32")

            n_snd = int(ds.dims.get("sounding", 1))
            n_lvl = int(ds.dims.get("level", vals.shape[-1]))

            vals_1d = _ensure_1d(vals)
            if ("level" in ds.dims) and (vals_1d.shape[0] != n_lvl):
                raise ValueError(f"'height_ptu' length {vals_1d.shape[0]} != level dim {n_lvl}")

            _set_height_with_dims(ds, vals_1d, n_snd, n_lvl)

            # Metadata for final 'height'
            ds["height"].attrs.update({
                "standard_name": "geopotential_height",
                "long_name":     "geopotential height from PTU",
                "units":         "m",
                "source":        "PTU via hypsometric equation",
            })

            # Remove any leftover 'height_ptu'
            if "height_ptu" in ds.variables or "height_ptu" in ds.coords:
                ds = ds.drop_vars("height_ptu")

        # --- Promote extra variables to coordinates if present ---
        for coord_var in ["alt", "height"]:
            if coord_var in ds.data_vars and coord_var not in ds.coords:
                ds = ds.set_coords(coord_var)

        self.dataset = ds






    def get_direction(self):
        if self.profile.ascent_flag.values[0] == 0:
            direction = "ascent"
        elif self.profile.ascent_flag.values[0] == 1:
            direction = "descent"
        self.meta_data["sounding_direction"] = direction

    def set_launchtime(self):
        first_idx_w_time = np.argwhere(
            ~np.isnan(self.profile.squeeze().flight_time.values)
        )[0][0]
        self.meta_data["launch_time_dt"] = pd.to_datetime(
            self.profile.squeeze().flight_time.values[first_idx_w_time]
        )

    def export(self, output_fmt, cfg):
        """
        Save sounding to disk.

        - Uses platform from `cfg.main.get("platform")`
        - Changes platform name to snake_format.
        """
        platform_name = cfg.main.get("platform").replace(" ", "_")

        output = output_fmt.format(
            platform=platform_name,
            campaign=cfg.main.get("campaign"),
            campaign_id=cfg.main.get("campaign_id"),
            direction=self.meta_data["sounding_direction"],
            version=cfg.main.get("data_version"),
        )
        output = self.meta_data["launch_time_dt"].strftime(output)
        directory = os.path.dirname(output)
        Path(directory).mkdir(parents=True, exist_ok=True)

        platform_data = [platform_name] * self.dataset.dims["sounding"]
        self.dataset["platform"] = xr.DataArray(platform_data, dims=["sounding"])
        self.dataset["platform"].attrs["long_name"] = "Launching platform"
        self.dataset.encoding["unlimited_dims"] = ["sounding"]
        self.dataset.to_netcdf(output)
        logging.info(f"Sounding written to {output}")


    def ensure_ptu_height_pre_split(self, level=1):
        """
        Compute PTU-derived geopotential height on the full profile *before* splitting.
        Writes plain-float columns only:
        p [Pa], ta [K], alt [m] (if missing), and height_ptu [m].
        """
        import numpy as np
        import logging

        is_level1 = (level == 1)
        reader_type = str(self.meta_data.get("reader_type", self.meta_data.get("source", ""))).lower()
        if not (is_level1 and ".cor" in reader_type):
            return  # only act on Meteomodem .cor Level-1

        prof = self.profile

        # --- resolve standardized column names that your reader already maps to ---
        p_name   = "pressure"     if "pressure"     in prof.columns else ("p"    if "p"    in prof.columns else None)
        t_name   = "temperature"  if "temperature"  in prof.columns else ("ta"   if "ta"   in prof.columns else None)
        alt_name = "height"       if "height"       in prof.columns else ("alt"  if "alt"  in prof.columns else None)

        if p_name is None or t_name is None:
            logging.warning("[pysonde.sounding.py]: PTU height skipped — missing pressure (%s) or temperature (%s).",
                            p_name, t_name)
            return

        # --- pull as plain floats, no pint ---
        def _to_float(col):
            a = prof[col]
            # pint-pandas column -> magnitude
            try:
                import pint_pandas as pp
                if isinstance(a.dtype, pp.pint_array.PintType):
                    return a.pint.magnitude.astype("float64")
            except Exception:
                pass
            # xarray not expected here; DataFrame .values gives numpy
            return getattr(a, "values", a).astype("float64")

        p_raw = _to_float(p_name)   # likely hPa from .cor reader, but your reader already standardized → check below
        T_raw = _to_float(t_name)   # likely °C standardized to temperature; we’ll normalize safely

        # --- unit heuristics to plain floats ---
        # pressure: if max < 2000 → hPa, convert to Pa; else assume already Pa
        p_pa = (p_raw * 100.0) if (np.nanmax(p_raw) < 2000.0) else p_raw
        # temperature: if max < 200 → °C, convert to K; else assume already K
        ta_K = (T_raw + 273.15) if (np.nanmax(T_raw) < 200.0) else T_raw

        # persist normalized columns as plain floats
        self.profile.loc[:, "p"]  = p_pa
        self.profile.loc[:, "ta"] = ta_K

        # optional alt anchor (plain floats)
        z0 = None
        if alt_name is not None:
            alt_val = _to_float(alt_name)
            if "alt" not in self.profile.columns:
                self.profile.loc[:, "alt"] = alt_val.astype("float32")
            finite_idx = np.where(np.isfinite(alt_val))[0]
            if finite_idx.size > 0:
                z0 = float(alt_val[finite_idx[0]])

        # optional mixing ratio (plain floats if present; not required)
        mr = None
        if "mr" in self.profile.columns:
            mr = _to_float("mr")

        # compute PTU height (returns numpy floats)
        z_ptu = self._ptu_geopotential_height(self.profile["p"].values, self.profile["ta"].values, mr=mr, z0=z0)
        self.profile.loc[:, "height_ptu"] = z_ptu.astype("float32")

        # --- DEBUG: verify plain-float columns (no pint registries) ---
        for col in ("p", "ta", "alt", "height_ptu"):
            if col in self.profile.columns:
                s = self.profile[col]
                is_pint = False
                try:
                    import pint_pandas as pp
                    is_pint = isinstance(s.dtype, pp.pint_array.PintType)
                except Exception:
                    pass
                print(f"[PTU DEBUG] {col}: dtype={getattr(s, 'dtype', None)} pint_col={is_pint}")


        logging.info("[pysonde.sounding.py]: PTU height computed into 'height_ptu'. (p, ta, alt written as plain floats)")


    def _ptu_geopotential_height(self, p, T, mr=None, z0=None):
        """
        Hypsometric integration along the full time order.
        Works for both ascent (p decreasing) and descent (p increasing).
        Returns a z array aligned to the original indexing; invalid samples are NaN.
        """
        import numpy as np

        p  = np.asarray(p,  dtype="float64")
        T  = np.asarray(T,  dtype="float64")
        mr = None if mr is None else np.asarray(mr, dtype="float64")

        n = p.size
        z = np.full(n, np.nan, dtype="float64")
        if n == 0:
            return z

        # Reference height
        if z0 is None:
            z0 = 0.0
            try:
                a0 = np.asarray(self.profile["alt"]).astype("float64")
                i0a = np.where(np.isfinite(a0))[0]
                if i0a.size:
                    z0 = float(a0[i0a[0]])
            except Exception:
                pass

        # Basic QC masks per-sample
        m = np.isfinite(p) & np.isfinite(T) & (p > 100.0) & (T > 150.0) & (T < 330.0)
        if mr is not None:
            m &= np.isfinite(mr) & (mr >= 0.0) & (mr < 0.05)

        if not np.any(m):
            return z

        # Virtual temperature
        if mr is not None:
            q = mr / (1.0 + mr)
            Tv = T * (1.0 + 0.61 * q)
        else:
            Tv = T

        # Pairwise mask: both ends valid
        vpair = m[:-1] & m[1:]

        # Pairwise hypsometric steps (sign handles both directions)
        Rd = 287.05
        g  = 9.80665
        Tv_bar = 0.5 * (Tv[:-1] + Tv[1:])
        with np.errstate(invalid="ignore", divide="ignore"):
            ln_ratio = np.log(p[:-1] / p[1:])

        dz = (Rd / g) * Tv_bar * ln_ratio
        dz[~vpair] = 0.0  # don’t advance across invalid gaps

        # Cumulative height relative to first valid sample
        cum = np.concatenate(([0.0], np.cumsum(dz)))
        i0 = int(np.where(m)[0][0])
        z = z0 + (cum - cum[i0])

        # Invalidate samples where single-point QC failed
        z[~m] = np.nan
        return z
