"""
Thermodynamic functions
"""

import metpy
import metpy.calc as mpcalc
import numpy as np
import pint_pandas as pp
import xarray as xr
import pint
from pint import UnitRegistry
from moist_thermodynamics import saturation_vapor_pressures as mtsvp

# ureg = pint.UnitRegistry()
# ureg.define("fraction = [] = frac")
# ureg.define("percent = 1e-2 frac = pct")
# pp.PintType.ureg = ureg


def convert_rh_to_dewpoint(temperature, rh):
    """
    Convert T and RH to dewpoint exactly
    following the formula used by the Vaisala
    M41 sounding system

    Input
    -----
    temperature : array
        temperature in Kevlin or supported quantity
    """
    if isinstance(temperature, pp.pint_array.PintArray):
        ureg = temperature.units._REGISTRY
        temperature_K = temperature.quantity.to(
            "K"
        ).magnitude  # would be better to stay unit aware
    else:
        temperature_K = temperature.data.magnitude
    if isinstance(rh, pp.pint_array.PintArray):
        rh = rh.quantity.to("percent").magnitude
    else:
        rh = rh.data.magnitude
    assert np.any(temperature_K > 100), "Temperature seems to be not given in Kelvin"
    kelvin = 15 * np.log(100 / rh) - 2 * (temperature_K - 273.15) + 2711.5
    t_dew = temperature_K * 2 * kelvin / (temperature_K * np.log(100 / rh) + 2 * kelvin)
    if isinstance(temperature, pp.pint_array.PintArray):
        t_dew = t_dew * ureg("K")
    return t_dew


def calc_saturation_pressure(temperature, method="hardy1998"):
    """
    Calculate saturation water vapor pressure.

    Parameters
    ----------
    temperature : array, pint.Quantity, or xarray.DataArray
        Temperature in Kelvin or dew point temperature for actual vapor pressure.
    method : str, optional
        Method for calculating saturation pressure:
            'hardy1998' : ITS-90 formulation (default).
            'wagner_pruss' : IAPWS-95 formulation from Wagner & Pruss (2002).

    Returns
    -------
    e_sw : array, pint.Quantity, or xarray.DataArray
        Saturation vapor pressure in Pascals.
    """
    # Ensure temperature is in Kelvin
    if isinstance(temperature, pp.pint_array.PintArray):
        ureg = temperature.units._REGISTRY
        temperature_K = temperature.quantity.to("K").magnitude
    elif isinstance(temperature, xr.core.dataarray.DataArray) and hasattr(temperature.data, "_units"):
        temperature_K = temperature.pint.to("K").metpy.magnitude
    else:
        temperature_K = temperature
    if method == "hardy1998":
        g = np.empty(8)
        g[0] = -2.8365744 * 10**3
        g[1] = -6.028076559 * 10**3
        g[2] = 1.954263612 * 10**1
        g[3] = -2.737830188 * 10 ** (-2)
        g[4] = 1.6261698 * 10 ** (-5)
        g[5] = 7.0229056 * 10 ** (-10)
        g[6] = -1.8680009 * 10 ** (-13)
        g[7] = 2.7150305

        e_sw = np.exp(np.sum([g[i] * temperature_K ** (i - 2) for i in range(7)], axis=0) + g[7] * np.log(temperature_K))

    elif method == "wagner_pruss":
        #print("I use wagner_pruss for e_s!!!")
        e_sw = mtsvp.liq_wagner_pruss(temperature_K)

    else:
        raise ValueError(f"Unknown method '{method}'. Available methods: 'hardy1998', 'wagner_pruss'.")

    # Convert result back to PintArray or xarray DataArray
    if isinstance(temperature, pp.pint_array.PintArray):
        pp.PintType.ureg = ureg
        e_sw = pp.PintArray(e_sw, dtype="Pa")
    elif isinstance(temperature, xr.DataArray) and hasattr(temperature.data, "_units"):
        e_sw = xr.DataArray(e_sw, dims=temperature.dims, coords=temperature.coords) * metpy.units.units("Pa")

    return e_sw

'''
def calc_wv_mixing_ratio(sounding, vapor_pressure):
    """
    Calculate water vapor mixing ratio
    """

    ureg = pint.get_application_registry()  # Ensure unit consistency

    # Convert vapor_pressure to Pascal if necessary
    if isinstance(vapor_pressure, pp.pint_array.PintArray):
        vapor_pressure = vapor_pressure.quantity
    elif isinstance(vapor_pressure, xr.DataArray):
        try:
            vapor_pressure = vapor_pressure.pint.to("Pa")
        except AttributeError:
            vapor_pressure = vapor_pressure.pint.quantify({"vapor_pressure": "Pa"}).pint.to("Pa")
    elif isinstance(vapor_pressure, pint.Quantity):
        vapor_pressure = vapor_pressure.to(ureg.Pa)
    else:
        vapor_pressure = vapor_pressure * ureg.Pa  

    # Identify and convert total pressure
    pressure_var = "p" if "p" in sounding else "pressure" if "pressure" in sounding else None
    if pressure_var is None:
        raise KeyError("No valid pressure variable found in the dataset! Expected 'p' or 'pressure'.")

    total_pressure = sounding[pressure_var]
    if hasattr(total_pressure, "pint"):
        total_pressure = total_pressure.pint.to("Pa")
    elif isinstance(total_pressure, pint.Quantity):
        total_pressure = total_pressure.to(ureg.Pa)
    else:
        total_pressure = pint.Quantity(total_pressure.values, ureg.Pa)

    # Compute mixing ratio
    wv_mix_ratio = (0.622 * vapor_pressure) / (total_pressure - vapor_pressure)

    # Ensure correct output units
    try:
        wv_mix_ratio = wv_mix_ratio.pint.to("kg/kg")
    except AttributeError:
        pass  # Return as-is if units are missing

    return wv_mix_ratio
'''

import numpy as np
import pandas as pd
import xarray as xr
import pint
import pint_pandas as pp

def calc_wv_mixing_ratio(sounding, vapor_pressure):
    """
    Calculate water vapor mixing ratio w = 0.622 e / (p - e)
    Returns a pandas.Series if 'sounding' is a DataFrame,
    else returns an xarray.DataArray aligned to the pressure variable.
    """

    def _pa_mag(x):
        # pint-xarray DataArray
        if isinstance(x, xr.DataArray) and hasattr(x, "pint"):
            return x.pint.to("Pa").pint.magnitude
        # pint.Quantity
        if isinstance(x, pint.Quantity):
            return x.to("Pa").magnitude
        # pandas PintArray
        if isinstance(x, pp.pint_array.PintArray):
            return x.quantity.to("Pa").magnitude
        # plain array / scalar
        return getattr(x, "values", x)

    # Pick pressure column from sounding (DataFrame or Dataset)
    if isinstance(sounding, pd.DataFrame):
        if "p" in sounding:
            p_raw = sounding["p"]
        elif "pressure" in sounding:
            p_raw = sounding["pressure"]
        else:
            raise KeyError("No valid pressure variable found in the dataset! Expected 'p' or 'pressure'.")
        p_pa = _pa_mag(p_raw).astype(float)
        e_pa = _pa_mag(vapor_pressure).astype(float)

        with np.errstate(invalid="ignore", divide="ignore"):
            w = 0.622 * e_pa / (p_pa - e_pa)

        return pd.Series(w, index=sounding.index, name="mr")

    elif isinstance(sounding, xr.Dataset):
        if "p" in sounding:
            pv = sounding["p"]
        elif "pressure" in sounding:
            pv = sounding["pressure"]
        else:
            raise KeyError("No valid pressure variable found in the dataset! Expected 'p' or 'pressure'.")

        p_pa = _pa_mag(pv).astype(float)
        e_pa = _pa_mag(vapor_pressure).astype(float)

        with np.errstate(invalid="ignore", divide="ignore"):
            w = 0.622 * e_pa / (p_pa - e_pa)

        da = xr.DataArray(w, dims=pv.dims, coords=pv.coords, name="mr")
        da.attrs["units"] = "kg kg-1"
        return da

    else:
        raise TypeError("sounding must be a pandas DataFrame or xarray Dataset")


def calc_theta_from_T(T, p):
    """
    Input :
        T : temperature
        p : pressure
    Output :
        theta : Potential temperature values
    Function to estimate potential temperature from the
    temperature and pressure in the given dataset. This function uses MetPy's
    functions to get theta:
    (i) mpcalc.potential_temperature()

    """
    theta = mpcalc.potential_temperature(p.metpy.quantify(), T.metpy.quantify())

    return theta


def calc_T_from_theta(theta, p):
    """
    Input :
        theta : potential temperature (K)
        p : pressure (hPa)
    Output :
        T : Temperature values
    Function to estimate temperature from potential temperature and pressure,
    in the given dataset. This function uses MetPy's
    functions to get T:
    (i) mpcalc.temperature_from_potential_temperature()

    """
    T = mpcalc.temperature_from_potential_temperature(
        p,
        theta,
    )

    return T
