"""Helper to create dataset

Create an empty dataset based on a template.yaml file
in the format of

    global_attrs:
        title: "Awesome dataset"
        description: "It is just awesome"
    coordinates:
        time:
            attrs:
                units: "seconds since 1970-01-01 00:00:00"
                calendar: "standard"
                axis: "T"
                standard_name: "time"
            dimension: ${time_dimension}
        range:
            attrs:
                units: "m"
            dimension: ${range_dimension}
    variables:
        label:
            attrs:
                description: "really great variable"
            coordinates:
                - time
                - range

The dimensions need to be passed during runtime. This can be easily
dome by creating an additional OmegaConf instance and merge it with
the one above, e.g.
    runtime_conf = OmegaConf.create({'time_dimension':100, 'range_dimension':1000})
    cfg = OmegaConf.merge(template_conf, runtime_conf)
    ds = create_dataset(cfg)

"""

import logging

import _helpers as h
import xarray as xr
import numpy as np


def create_dataset(cfg):
    """
    Create dataset based on template

    cfg : OmegaConf
        Config containing dataset template
    """

    ds = xr.Dataset()
    ds = set_global_attrs(cfg, ds)
    ds = set_coords(cfg, ds)
    ds = set_variables(cfg, ds)
    return ds


def set_global_attrs(cfg, ds):
    logging.debug("Add global attributes")
    if "global_attrs" in cfg.keys():
        _cfg = h.remove_missing_cfg(cfg["global_attrs"])
        ds.attrs = _cfg
    return ds

def set_coords(cfg, ds):
    if "coordinates" in cfg.keys():
        for coord, params in cfg.coordinates.items():
            if type(params["dimension"]) is int:
                ds = ds.assign_coords(
                    {coord: range(params["dimension"])}
                )  # write temporary values to coord
            else:
                ds = ds.assign_coords({coord: params["dimension"]})
            if "attrs" in params.keys():
                ds[coord].attrs = params["attrs"]
            if "encodings" in params.keys():
                ds[coord].encoding = params["encodings"]
    return ds

def _resolve_axis_name(ds, name: str) -> str:
    """
    Return a coord/dim name that actually exists in ds.
    For the vertical axis and its bounds, accept either 'alt' or 'height'
    (and 'alt_bnds' or 'height_bnds') and pick whichever is present.
    """
    # already present? keep it
    if name in ds.dims or name in ds.coords or name in ds.data_vars:
        return name

    # vertical axis aliasing
    if name in ("alt", "height"):
        if "height" in ds.dims or "height" in ds.coords or "height" in ds.data_vars:
            return "height"
        if "alt" in ds.dims or "alt" in ds.coords or "alt" in ds.data_vars:
            return "alt"

    # bounds aliasing
    if name in ("alt_bnds", "height_bnds"):
        if "height_bnds" in ds:
            return "height_bnds"
        if "alt_bnds" in ds:
            return "alt_bnds"

    return name


def set_variables(cfg, ds):
    logging.debug("Add variables to dataset")
    if "variables" in cfg.keys():
        for var, params in cfg.variables.items():
            if var == "level" or var == "sounding":
                if "encodings" in params.keys():
                    ds[var].encoding = params["encodings"]
            else:
                coord_dict = {coord: ds[coord] for coord in params.coordinates}
                ds[var] = xr.DataArray(None, coords=coord_dict, dims=params.coordinates)
                if "attrs" in params.keys():
                    ds[var].attrs = params["attrs"]
                if "encodings" in params.keys():
                    ds[var].encoding = params["encodings"]
    return ds

'''
def set_variables(cfg, ds):
    logging.debug("Add variables to dataset")

    if "variables" not in cfg.keys():
        return ds

    for var, params in cfg.variables.items():
        # Pass-through for dims that already exist
        if var in ("level", "sounding"):
            if "encodings" in params.keys():
                ds[var].encoding = params["encodings"]
            continue

        # ---- 1) Resolve coordinate names (alt/height, bounds, etc.) ----
        coord_names = list(getattr(params, "coordinates", []))
        coords_resolved = [_resolve_axis_name(ds, c) for c in coord_names]

        # Build a lookup of coords from the dataset (don’t merge them yet)
        coord_lookup = {}
        for c in coords_resolved:
            if c in ds:
                coord_lookup[c] = ds[c]
            elif c in ds.coords:
                coord_lookup[c] = ds.coords[c]
            else:
                raise KeyError(
                    f"Coordinate {c!r} not found. "
                    f"Available coords: {list(ds.coords)}, data_vars: {list(ds.data_vars)}"
                )

        # ---- 2) Resolve variable dimensions (MUST match the dataset) ----
        dims_cfg = list(getattr(params, "dimensions", []))
        if dims_cfg:
            dims_resolved = [_resolve_axis_name(ds, d) for d in dims_cfg]
        else:
            # DEFAULT: use resolved coordinates as dims (don’t filter by ds.dims!)
            dims_resolved = list(coords_resolved)

        # ---- 3) Compute the target shape from dataset sizes ----
        shape = []
        for d in dims_resolved:
            if d in ds.dims:
                shape.append(ds.sizes[d])
            elif d in ds.coords and ds.coords[d].sizes.get(d, None) is not None:
                # just in case coord exists but ds.dims not populated yet
                shape.append(int(ds.coords[d].sizes[d]))
            else:
                raise ValueError(
                    f"Dimension {d!r} is not defined in the dataset. "
                    f"Current dims: {dict(ds.dims)}. "
                    f"Make sure it’s declared in cfg.coordinates before variables."
                )

        # ---- 4) Create the variable with proper shape and dims (no coords yet) ----
        # Use NaNs as placeholder (most fields are numeric). If you need non-numeric,
        # you can special-case here by checking params.attrs.get('dtype', ...)
        data_placeholder = np.full(tuple(shape), np.nan, dtype="float64")
        ds[var] = xr.DataArray(
            data_placeholder,
            dims=dims_resolved,
        )

        # ---- 5) (Optional) attach ONLY auxiliary (non-dim) coords to the variable ----
        #     Attaching dim-coords (e.g., 'alt'/'height') here can cause merge ambiguity.
        aux_coords = {}
        for name, cda in coord_lookup.items():
            if name not in dims_resolved:
                aux_coords[name] = cda
        if aux_coords:
            ds[var] = ds[var].assign_coords(aux_coords)

        # ---- 6) Attributes / encodings ----
        if "attrs" in params.keys():
            ds[var].attrs = params["attrs"]
        if "encodings" in params.keys():
            ds[var].encoding = params["encodings"]

    return ds
'''