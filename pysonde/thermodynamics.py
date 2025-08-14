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

'''
def calc_saturation_pressure(temperature, method="hardy1998"):
    """
    Calculate saturation water vapor pressure

    Input
    -----
    temperature : array
        array of temperature in Kevlin or dew point temperature for actual vapor pressure
    method : str
        Formula used for calculating the saturation pressure
            'hardy1998' : ITS-90 Formulations for Vapor Pressure, Frostpoint Temperature,
                Dewpoint Temperature, and Enhancement Factors in the Range –100 to +100 C,
                Bob Hardy, Proceedings of the Third International Symposium on Humidity and Moisture,
                1998 (same as used in Aspen software after May 2018)

    Return
    ------
    e_sw : array
        saturation pressure (Pa)

    Examples
    --------
    >>> calc_saturation_pressure([273.15])
    array([ 611.2129107])

    >>> calc_saturation_pressure([273.15, 293.15, 253.15])
    array([  611.2129107 ,  2339.26239586,   125.58350529])
    """
    if isinstance(temperature, pp.pint_array.PintArray):
        ureg = temperature.units._REGISTRY
        temperature_K = temperature.quantity.to(
            "K"
        ).magnitude  # would be better to stay unit aware
    elif isinstance(temperature, xr.core.dataarray.DataArray) and hasattr(
        temperature.data, "_units"
    ):
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

        e_sw = np.zeros_like(temperature_K)

        for t, temp in enumerate(temperature_K):
            ln_e_sw = np.sum([g[i] * temp ** (i - 2) for i in range(0, 7)]) + g[
                7
            ] * np.log(temp)
            e_sw[t] = np.exp(ln_e_sw)
        if isinstance(temperature, pp.pint_array.PintArray):
            pp.PintType.ureg = ureg
            e_sw = pp.PintArray(e_sw, dtype="Pa")
        elif isinstance(temperature, xr.core.dataarray.DataArray) and hasattr(
            temperature.data, "_units"
        ):
            e_sw = xr.DataArray(
                e_sw, dims=temperature.dims, coords=temperature.coords
            ) * metpy.units.units("Pa")
        return e_sw
    '''


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
        temperature_K = temperature  # Assume already in Kelvin

    if method == "hardy1998":
        #print("I use hardy1998 for e_s!!!")
        g = np.array([-2.8365744e3, -6.028076559e3, 1.954263612e1, -2.737830188e-2,
                      1.6261698e-5, 7.0229056e-10, -1.8680009e-13, 2.7150305])
        
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
    PySonde Original
    
    Calculate water vapor mixing ratio
    """
    if isinstance(vapor_pressure, pp.pint_array.PintArray):
        ureg = vapor_pressure.units._REGISTRY
        vapor_pressure_Pa = vapor_pressure.quantity.to("Pa").magnitude
    else:
        vapor_pressure_Pa = vapor_pressure
    if "pint" in sounding.pressure.dtype.__str__():
        total_pressure = sounding.pressure.values.quantity.to("hPa").magnitude
    else:
        total_pressure = sounding.pressure.values
    wv_mix_ratio = 1000.0 * (
        (0.622 * vapor_pressure_Pa) / (100.0 * total_pressure - vapor_pressure_Pa)
    )
    if isinstance(vapor_pressure, pp.pint_array.PintArray):
        return wv_mix_ratio * ureg("g") / ureg("kg")
    else:
        return wv_mix_ratio

'''




'''

ureg = pint.UnitRegistry()  # Global unit registry

def calc_wv_mixing_ratio(sounding, vapor_pressure):
    """
    Works for LEVEL-1 
    Calculate water vapor mixing ratio in kg/kg.

    Parameters:
    ----------
    sounding : xarray.Dataset
        Dataset containing pressure values.
    vapor_pressure : xarray.DataArray or pint.Quantity
        Vapor pressure values (must have units of Pascal).

    Returns:
    -------
    xarray.DataArray
        Water vapor mixing ratio with units of kg/kg.
    """

    ureg = pint.get_application_registry()  # Ensure we use the same Pint registry

    # 🔍 Debug: Check vapor_pressure
    print("\n--- Debugging: vapor_pressure ---")
    print("Type of vapor_pressure:", type(vapor_pressure))
    if hasattr(vapor_pressure, "pint"):
        print("vapor_pressure units (pint_xarray):", vapor_pressure.pint.units)
    elif isinstance(vapor_pressure, pint.Quantity):
        print("vapor_pressure units (pint.Quantity):", vapor_pressure.units)
    else:
        print("vapor_pressure has no Pint units!")

    # Convert `PintArray` to `pint.Quantity`
    if isinstance(vapor_pressure, pp.pint_array.PintArray):
        vapor_pressure = vapor_pressure.quantity  # Extract as `pint.Quantity`

    # Ensure vapor pressure has correct units
    if isinstance(vapor_pressure, pint.Quantity):
        vapor_pressure = vapor_pressure.to(ureg.Pa)  # Ensure units are in Pascal
    else:
        vapor_pressure = vapor_pressure * ureg.Pa  # Add missing units

    # 🔧 Check total pressure in dataset
    pressure_var = "p" if "p" in sounding else "pressure" if "pressure" in sounding else None
    if pressure_var is None:
        raise KeyError("No valid pressure variable found in the dataset! Expected 'p' or 'pressure'.")

    total_pressure = sounding[pressure_var]
    if hasattr(total_pressure, "pint"):
        total_pressure = total_pressure.pint.to("Pa")  # Already a Pint array
    elif isinstance(total_pressure, pint.Quantity):
        total_pressure = total_pressure.to(ureg.Pa)  # Ensure same registry
    else:
        total_pressure = pint.Quantity(total_pressure.values, ureg.Pa)  # Convert raw values

    # Compute mixing ratio
    wv_mix_ratio = (0.622 * vapor_pressure) / (total_pressure - vapor_pressure)

    # 🔧 Ensure correct output units using `pint_xarray`
    wv_mix_ratio = wv_mix_ratio.pint.to("kg/kg")  # Convert to kg/kg

    print("\n--- Debugging: Final Mixing Ratio ---")
    print("Final wv_mix_ratio sample before returning:", wv_mix_ratio[:10])
    print("Final wv_mix_ratio units before returning:", wv_mix_ratio.pint.units)

    return wv_mix_ratio

'''
'''

def calc_wv_mixing_ratio(sounding, vapor_pressure):
    """
    Works for LEVEL-2 
    Calculate water vapor mixing ratio in kg/kg.

    Parameters:
    ----------
    sounding : xarray.Dataset
        Dataset containing pressure values.
    vapor_pressure : xarray.DataArray or pint.Quantity
        Vapor pressure values (must have units of Pascal).

    Returns:
    -------
    xarray.DataArray
        Water vapor mixing ratio with units of kg/kg.
    """

    ureg = pint.UnitRegistry()  # Initialize Pint unit registry

    # 🔧 Ensure dataset variables have explicit units
    sounding = sounding.pint.quantify({"p": "Pa"})  # Ensure 'p' is in Pascals

    # 🔧 Ensure vapor pressure has correct units
    if isinstance(vapor_pressure, xr.DataArray):
        vapor_pressure = vapor_pressure.pint.quantify({"vapor_pressure": "Pa"}).pint.to("Pa")
    elif isinstance(vapor_pressure, pint.Quantity):
        vapor_pressure = vapor_pressure.to("Pa")
    else:
        vapor_pressure = vapor_pressure * ureg.Pa  # Add missing units

    # 🔧 Convert total pressure to Pascals
    total_pressure = sounding["p"].pint.to("Pa")

    # Compute mixing ratio
    wv_mix_ratio = (0.622 * vapor_pressure) / (total_pressure - vapor_pressure)

    # 🔧 Ensure correct output units using pint_xarray
    wv_mix_ratio = wv_mix_ratio.pint.to("kg/kg")  # Convert to kg/kg

    print("\n--- Debugging: Final Mixing Ratio ---")
    print("Final wv_mix_ratio sample before returning:", wv_mix_ratio[:10])
    print("Final wv_mix_ratio units before returning:", wv_mix_ratio.pint.units)

    return wv_mix_ratio
'''
'''
def calc_wv_mixing_ratio(sounding, vapor_pressure):
    """
    Works for both LEVEL-1 and LEVEL-2.
    Calculate water vapor mixing ratio in kg/kg.

    Parameters:
    ----------
    sounding : xarray.Dataset
        Dataset containing pressure values.
    vapor_pressure : xarray.DataArray or pint.Quantity
        Vapor pressure values (must have units of Pascal).

    Returns:
    -------
    xarray.DataArray
        Water vapor mixing ratio with units of kg/kg.
    """

    ureg = pint.get_application_registry()  # Ensure we use the same Pint registry

    # 🔍 Debug: Check vapor_pressure
    print("\n--- Debugging: vapor_pressure ---")
    print("Type of vapor_pressure:", type(vapor_pressure))
    if hasattr(vapor_pressure, "pint"):
        print("vapor_pressure units (pint_xarray):", vapor_pressure.pint.units)
    elif isinstance(vapor_pressure, pint.Quantity):
        print("vapor_pressure units (pint.Quantity):", vapor_pressure.units)
    else:
        print("vapor_pressure has no Pint units!")

    # Convert `PintArray` to `pint.Quantity` if necessary
    if isinstance(vapor_pressure, pp.pint_array.PintArray):
        vapor_pressure = vapor_pressure.quantity  # Extract as `pint.Quantity`

    # Ensure vapor pressure has correct units
    if isinstance(vapor_pressure, xr.DataArray):
        try:
            vapor_pressure = vapor_pressure.pint.to("Pa")  # Try converting directly
        except AttributeError:
            vapor_pressure = vapor_pressure.pint.quantify({"vapor_pressure": "Pa"}).pint.to("Pa")
    elif isinstance(vapor_pressure, pint.Quantity):
        vapor_pressure = vapor_pressure.to(ureg.Pa)  # Ensure units are in Pascal
    else:
        vapor_pressure = vapor_pressure * ureg.Pa  # Add missing units

    # 🔧 Check total pressure in dataset
    pressure_var = "p" if "p" in sounding else "pressure" if "pressure" in sounding else None
    if pressure_var is None:
        raise KeyError("No valid pressure variable found in the dataset! Expected 'p' or 'pressure'.")

    total_pressure = sounding[pressure_var]

    # Handle different unit representations
    if hasattr(total_pressure, "pint"):
        total_pressure = total_pressure.pint.to("Pa")  # Already a Pint array
    elif isinstance(total_pressure, pint.Quantity):
        total_pressure = total_pressure.to(ureg.Pa)  # Ensure same registry
    else:
        total_pressure = pint.Quantity(total_pressure.values, ureg.Pa)  # Convert raw values

    # Compute mixing ratio
    wv_mix_ratio = (0.622 * vapor_pressure) / (total_pressure - vapor_pressure)

    # 🔧 Ensure correct output units using `pint_xarray`
    try:
        wv_mix_ratio = wv_mix_ratio.pint.to("kg/kg")  # Convert to kg/kg
    except AttributeError:
        print("Warning: Output does not have Pint units, returning as-is.")

    print("\n--- Debugging: Final Mixing Ratio ---")
    print("Final wv_mix_ratio sample before returning:", wv_mix_ratio[:10])
    if hasattr(wv_mix_ratio, "pint"):
        print("Final wv_mix_ratio units before returning:", wv_mix_ratio.pint.units)

    return wv_mix_ratio
'''

def calc_wv_mixing_ratio(sounding, vapor_pressure):
    """
    Works for both LEVEL-1 and LEVEL-2.
    
    Key improvements:
    - Handles both raw numeric values and Pint-quantified inputs.
    - Dynamically detects pressure variable ("p" or "pressure").
    - Ensures unit consistency across different dataset formats.
    - Includes error handling for missing or incompatible units.

    Parameters:
    ----------
    sounding : xarray.Dataset
        Dataset containing pressure values.
    vapor_pressure : xarray.DataArray or pint.Quantity
        Vapor pressure values (must have units of Pascal).

    Returns:
    -------
    xarray.DataArray
        Water vapor mixing ratio with units of kg/kg.
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
