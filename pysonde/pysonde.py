#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to convert sounding files from different sources
to netCDF
"""

import matplotlib.pyplot as plt
import argparse
import logging
import sys

import numpy as np
import tqdm
from omegaconf import OmegaConf

from . import _helpers as h
from . import _proc_level2 as p2
from . import readers


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "-i",
        "--inputfile",
        metavar="INPUT_FILE",
        help="Single sonde file or file format\n" "including wildcards",
        default=None,
        required=False,
        nargs="+",
        type=h.unixpath,
    )

    parser.add_argument(
        "-o",
        "--output",
        metavar="/some/example/path/",
        help="Output folder for converted files (netCDF). You can\n"
        " although not recommended also define an output file\n"
        "(format). However, please share only those with the\n"
        " the default filename.\n"
        " The following formats can be used:\n"
        "\t {platform}\t platform name\n"
        "\t {location}\t platform location\n"
        "\t {direction}\t sounding direction\n"
        "\t\t date format with\n"
        "\t\t %%Y%%m%%d %%H%%M and so on\n"
        "\t\t and others to format the output folder dynamically.",
        default=None,
        required=False,
    )

    parser.add_argument(
        "-c",
        "--config",
        metavar="MAIN_CONFIG.YML",
        help="Main config file with references\n" "to specific config files",
        default="../config/main.yaml",
        required=False,
        type=h.unixpath,
    )

    parser.add_argument(
        "-m",
        "--method",
        metavar="METHOD",
        help="Interpolation method ('bin', 'linear', 'linear_masked')",
        default="linear_masked",
        required=False,
    )

    parser.add_argument(
        "-ia",
        "--interp_axis",
        metavar="INTERPOLATION_AXIS",
        help="Interpolation along which axis ('alt', 'height')",
        default="height",
        required=False,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        metavar="DEBUG",
        help="Set the level of verbosity [DEBUG, INFO," " WARNING, ERROR]",
        required=False,
        default="INFO",
    )

    parsed_args = vars(parser.parse_args())

    if parsed_args["inputfile"] is None:
        parser.error(
            "--inputfile must be defined. For several files"
            "enter the inputfile format with wildcards."
        )

    return parsed_args


def load_reader(filename):
    """
    Infer appropriate reader from filename
    """
    ending = filename.suffix
    if ending == ".mwx":
        from .readers.readers import MW41

        reader = MW41
    elif ending == ".cor":
        from .readers.readers import METEOMODEM

        reader = METEOMODEM
    elif ending == ".nc":
        from .readers.readers import pysondeL1

        reader = pysondeL1
    else:
        raise h.ReaderNotImplemented(f"Reader for filetype {ending} not implemented")
    return reader


def main(args=None):
    if args is None:
        args = {}
        try:
            args = get_args()
        except ValueError:
            sys.exit()
    else:
        pass

    h.setup_logging(args["verbose"])

    # Combine all configurations
    main_cfg = OmegaConf.load(args["config"])
    cfg = h.combine_configs(main_cfg.configs)

    input_files = h.find_files(args["inputfile"])
    logging.info("Files to process {}".format([file.name for file in input_files]))

    logging.debug("Load reader. All files need to be of same type!")
    # Load correct reader class
    reader_class = load_reader(input_files[0])
    # Configure reader according to config file
    reader = reader_class(cfg)

    for ifile, file in enumerate(tqdm.tqdm(input_files)):
        logging.debug("Reading file number {}".format(ifile))
    
        sounding = reader.read(file)

        if isinstance(reader, readers.readers.MW41) or isinstance(
            reader, readers.readers.METEOMODEM
        ):
            # NEW — computes 'height' on full profile
            sounding.ensure_ptu_height_pre_split(level=1)         
            # Split sounding into ascending and descending branch
            sounding_asc, sounding_dsc = sounding.split_by_direction()
            for snd in [sounding_asc, sounding_dsc]:
                if len(snd.profile) < 2:
                    logging.warning(
                        "Sounding ({}) does not contain data. "
                        "Skip sounding-direction of {}".format(
                            snd.meta_data["sounding_direction"], file
                        )
                    )
                    continue
                snd.calculate_additional_variables(cfg)
                snd.convert_sounding_df2ds()
                snd.create_dataset(cfg, vertical_interpolation_axis=args["interp_axis"])
                snd.export(args["output"], cfg)

        elif isinstance(reader, readers.readers.pysondeL1):
            cfg = h.replace_placeholders_cfg_level2(cfg)

            if len(sounding.profile.sounding) != 1:
                raise NotImplementedError(
                    "Level 1 files with more than one sounding are currently not supported"
                )

            ds = sounding.profile.isel({"sounding": 0})
            ds_input = ds.copy()

            # Check monotonic ascent/descent
            if np.all(np.diff(ds.isel(level=slice(20, -1))[args["interp_axis"]].values) > 0) or np.all(
                np.diff(ds.isel(level=slice(20, -1))[args["interp_axis"]].values) < 0
            ):
                logging.debug("Sounding is monotonic ascending/descending")
            else:
                logging.warning(
                    "Sounding is not monotonic ascending/descending. The ascent rate will be artificial"
                )

            # Geopotential height issue
            # the geopotential height is not a measured coordinate and
            # the same height can occur at different pressure levels
            # here the first occurrence is used
            _, uniq_altitude_idx = np.unique(ds[args["interp_axis"]].values, return_index=True)
            ds = ds.isel({"level": uniq_altitude_idx})

            # Consistent platform test
            if ifile == 0:
                platform = ds.platform
            else:
                assert (
                    ds.platform == platform
                ), "The platform seems to change from {} to {}".format(
                    platform, ds.platform
                )

            # Unique levels test
            if len(ds[args["interp_axis"]]) != len(np.unique(ds[args["interp_axis"]])):
                logging.error("Altitude levels are not unique of {}".format(file))
                break
            

            # Prepare some data that cannot be linearly interpolated
            ds, ds_new = p2.prepare_data_for_interpolation(
                ds, 
                sounding.unitregistry, 
                args["interp_axis"], 
                reader.variable_name_mapping_output.items(),
            )



            axis = args["interp_axis"]
            if axis not in {"alt", "height"}:
                raise ValueError(f"--interpolation must be 'alt' or 'height', got {axis!r}")
            if axis not in ds_new:
                raise ValueError(f"Chosen axis {axis!r} not found in dataset.")
            # Make sure it is a coordinate (not just a data_var)
            if axis in ds_new and axis not in ds_new.coords:
                ds_new = ds_new.set_coords(axis)

            # Interpolation
            interpolation_grid = np.arange(
                cfg.level2.setup.interpolation_grid_min,
                cfg.level2.setup.interpolation_grid_max,
                cfg.level2.setup.interpolation_grid_inc,
            )
            ds_interp = p2.interpolation(
                ds_new,
                args["method"],
                interpolation_grid,
                args["interp_axis"],
                sounding,
                reader.variable_name_mapping_output.items(),
                cfg,
            )

            ds_interp = p2.adjust_ds_after_interpolation(
                ds_interp,
                ds,
                ds_input,
                args["interp_axis"],
                reader.variable_name_mapping_output.items(),
                cfg,
            )

            if args["method"] in ("bin"):
                ds_interp = p2.count_number_of_measurement_within_bin(
                    ds_interp, 
                    ds_new, 
                    cfg, 
                    interpolation_grid,
                    args["interp_axis"],
                )

            ds_interp = p2.finalize_attrs(
                ds_interp, 
                ds, 
                cfg, 
                file, 
                reader.variable_name_mapping_output.items(), 
                args["interp_axis"],
            )

            sounding.profile = ds_interp
            sounding.create_dataset(cfg, vertical_interpolation_axis=args["interp_axis"], level=2)
            sounding.get_direction()
            sounding.set_launchtime()
            sounding.export(args["output"], cfg)


if __name__ == "__main__":
    main()
