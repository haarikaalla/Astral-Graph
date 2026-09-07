"""MCP server exposing **deterministic astronomy calculators**.

This server exists to satisfy the guardrail *"numeric calculations are verified in
code, not by the LLM"*. Agents must route every quantitative claim through these
tools; the Critic rejects numbers that were not produced here or by an upstream
data server.

Every function is pure Python with no network access.

Run standalone:  ``python -m mcp_servers.astro_compute_server``
"""

from __future__ import annotations

import math
from typing import Any

from mcp_servers._common import err, ok
from mcp_servers._compat import MCPServer

SOURCE = "AstralGraph deterministic calculator"
URL = "local://astro_compute"

# Physical constants (SI unless noted)
G = 6.67430e-11
SIGMA_SB = 5.670374419e-8
AU_M = 1.495978707e11
PC_LY = 3.26156378
EARTH_RADIUS_M = 6.371e6
EARTH_MASS_KG = 5.9722e24
SOLAR_RADIUS_M = 6.957e8
SOLAR_MASS_KG = 1.98892e30
SOLAR_LUMINOSITY_W = 3.828e26
TNT_JOULES_PER_MEGATON = 4.184e15
EARTH_INSOLATION_W_M2 = 1361.0

mcp = MCPServer("astral-compute")


@mcp.tool()
def impact_energy(
    diameter_m: float, velocity_km_s: float, density_kg_m3: float = 3000.0
) -> dict:
    """Kinetic energy of an asteroid impact, in joules, megatons TNT and Hiroshimas.

    Uses E = 1/2 m v^2 with m = rho * (pi/6) * d^3.

    Args:
        diameter_m: Object diameter in metres.
        velocity_km_s: Impact velocity in km/s.
        density_kg_m3: Bulk density; 3000 for stony, 8000 for iron, 1500 for cometary.
    """
    if diameter_m <= 0 or velocity_km_s <= 0 or density_kg_m3 <= 0:
        return err("diameter, velocity and density must be positive", source=SOURCE, url=URL)
    mass = density_kg_m3 * (math.pi / 6.0) * diameter_m**3
    v = velocity_km_s * 1000.0
    energy = 0.5 * mass * v**2
    megatons = energy / TNT_JOULES_PER_MEGATON
    # Collins et al. (2005) scaling for a simple crater in sedimentary rock.
    crater_km = 1.161 * (density_kg_m3 / 2500.0) ** (1 / 3) * (diameter_m / 1000.0) ** 0.78 * (
        velocity_km_s
    ) ** 0.44 * 9.81 ** -0.22 * math.sin(math.radians(45)) ** (1 / 3)
    return ok(
        {
            "mass_kg": mass,
            "energy_joules": energy,
            "energy_megatons_tnt": megatons,
            "hiroshima_equivalents": energy / (15 * 4.184e12),
            "approx_crater_diameter_km": max(0.0, crater_km),
            "formula": "E = 0.5 * rho * (pi/6) * d^3 * v^2",
        },
        source=SOURCE,
        url=URL,
    )


@mcp.tool()
def torino_scale_band(energy_megatons: float, impact_probability: float) -> dict:
    """Approximate Torino-scale band for a hypothetical impactor.

    Args:
        energy_megatons: Kinetic energy in megatons of TNT.
        impact_probability: Probability of impact, 0..1.
    """
    if energy_megatons < 0 or not 0.0 <= impact_probability <= 1.0:
        return err("energy must be >= 0 and probability in [0,1]", source=SOURCE, url=URL)
    if impact_probability < 1e-8 or energy_megatons < 1:
        level, label = 0, "No hazard"
    elif impact_probability < 1e-2 and energy_megatons < 100:
        level, label = 1, "Normal — routine discovery, collision very unlikely"
    elif impact_probability < 1e-2 and energy_megatons < 1e5:
        level, label = 2, "Meriting attention by astronomers"
    elif impact_probability >= 1e-2 and energy_megatons < 1e3:
        level, label = 4, "Threatening — close encounter, regional devastation possible"
    elif impact_probability >= 1e-2 and energy_megatons < 1e5:
        level, label = 7, "Threatening — unprecedented global threat"
    else:
        level, label = 8, "Certain collision — localised to global consequences"
    return ok(
        {
            "torino_level": level,
            "label": label,
            "note": "Approximation of the published Torino scale for explanation only; "
            "authoritative values come from JPL/CNEOS Sentry.",
        },
        source=SOURCE,
        url=URL,
    )


@mcp.tool()
def orbital_period(semi_major_axis_au: float, star_mass_solar: float = 1.0) -> dict:
    """Kepler's third law: orbital period in days for a given semi-major axis.

    Args:
        semi_major_axis_au: Semi-major axis in AU.
        star_mass_solar: Host-star mass in solar masses.
    """
    if semi_major_axis_au <= 0 or star_mass_solar <= 0:
        return err("inputs must be positive", source=SOURCE, url=URL)
    a = semi_major_axis_au * AU_M
    m = star_mass_solar * SOLAR_MASS_KG
    seconds = 2 * math.pi * math.sqrt(a**3 / (G * m))
    return ok(
        {
            "period_seconds": seconds,
            "period_days": seconds / 86400.0,
            "period_years": seconds / 3.15576e7,
            "formula": "T = 2*pi*sqrt(a^3 / (G*M))",
        },
        source=SOURCE,
        url=URL,
    )


@mcp.tool()
def semi_major_axis(period_days: float, star_mass_solar: float = 1.0) -> dict:
    """Inverse Kepler: semi-major axis in AU from an orbital period.

    Args:
        period_days: Orbital period in days.
        star_mass_solar: Host-star mass in solar masses.
    """
    if period_days <= 0 or star_mass_solar <= 0:
        return err("inputs must be positive", source=SOURCE, url=URL)
    t = period_days * 86400.0
    a = (G * star_mass_solar * SOLAR_MASS_KG * t**2 / (4 * math.pi**2)) ** (1 / 3)
    return ok({"semi_major_axis_au": a / AU_M, "semi_major_axis_m": a}, source=SOURCE, url=URL)


@mcp.tool()
def equilibrium_temperature(
    star_teff_k: float,
    star_radius_solar: float,
    semi_major_axis_au: float,
    albedo: float = 0.3,
) -> dict:
    """Planetary equilibrium temperature (K).

    Args:
        star_teff_k: Stellar effective temperature in kelvin.
        star_radius_solar: Stellar radius in solar radii.
        semi_major_axis_au: Orbital distance in AU.
        albedo: Bond albedo, 0..1 (Earth ~0.3).
    """
    if min(star_teff_k, star_radius_solar, semi_major_axis_au) <= 0 or not 0 <= albedo < 1:
        return err("invalid inputs", source=SOURCE, url=URL)
    r_star = star_radius_solar * SOLAR_RADIUS_M
    a = semi_major_axis_au * AU_M
    teq = star_teff_k * math.sqrt(r_star / (2 * a)) * (1 - albedo) ** 0.25
    luminosity = 4 * math.pi * r_star**2 * SIGMA_SB * star_teff_k**4
    flux = luminosity / (4 * math.pi * a**2)
    return ok(
        {
            "equilibrium_temperature_k": teq,
            "equilibrium_temperature_c": teq - 273.15,
            "stellar_luminosity_solar": luminosity / SOLAR_LUMINOSITY_W,
            "insolation_w_m2": flux,
            "insolation_earth_units": flux / EARTH_INSOLATION_W_M2,
            "formula": "Teq = Teff * sqrt(R*/2a) * (1-A)^0.25",
        },
        source=SOURCE,
        url=URL,
    )


@mcp.tool()
def habitable_zone(star_teff_k: float, star_luminosity_solar: float) -> dict:
    """Conservative and optimistic habitable-zone bounds (Kopparapu et al. 2013).

    Args:
        star_teff_k: Stellar effective temperature in kelvin (2600-7200 valid).
        star_luminosity_solar: Stellar luminosity in solar units.
    """
    if star_luminosity_solar <= 0 or star_teff_k <= 0:
        return err("inputs must be positive", source=SOURCE, url=URL)
    ts = star_teff_k - 5780.0
    coeffs = {
        "recent_venus": (1.7753, 1.4316e-4, 2.9875e-9, -7.5702e-12, -1.1635e-15),
        "runaway_greenhouse": (1.0512, 1.3242e-4, 1.5418e-8, -7.9895e-12, -1.8328e-15),
        "maximum_greenhouse": (0.3438, 5.8942e-5, 1.6558e-9, -3.0045e-12, -5.2983e-16),
        "early_mars": (0.3179, 5.4513e-5, 1.5313e-9, -2.7786e-12, -4.8997e-16),
    }
    bounds: dict[str, float] = {}
    for name, (s_eff_sun, a, b, c, d) in coeffs.items():
        s_eff = s_eff_sun + a * ts + b * ts**2 + c * ts**3 + d * ts**4
        bounds[name] = math.sqrt(star_luminosity_solar / s_eff)
    return ok(
        {
            "conservative_hz_au": [bounds["runaway_greenhouse"], bounds["maximum_greenhouse"]],
            "optimistic_hz_au": [bounds["recent_venus"], bounds["early_mars"]],
            "boundaries_au": bounds,
            "valid_teff_range_k": [2600, 7200],
            "reference": "Kopparapu et al. 2013, ApJ 765, 131",
        },
        source=SOURCE,
        url=URL,
    )


@mcp.tool()
def transit_depth(planet_radius_earth: float, star_radius_solar: float) -> dict:
    """Fractional transit depth (Rp/R*)^2 in ppm.

    Args:
        planet_radius_earth: Planet radius in Earth radii.
        star_radius_solar: Star radius in solar radii.
    """
    if planet_radius_earth <= 0 or star_radius_solar <= 0:
        return err("inputs must be positive", source=SOURCE, url=URL)
    ratio = (planet_radius_earth * EARTH_RADIUS_M) / (star_radius_solar * SOLAR_RADIUS_M)
    depth = ratio**2
    return ok(
        {
            "radius_ratio": ratio,
            "depth_fraction": depth,
            "depth_ppm": depth * 1e6,
            "depth_percent": depth * 100,
        },
        source=SOURCE,
        url=URL,
    )


@mcp.tool()
def planet_bulk_properties(mass_earth: float, radius_earth: float) -> dict:
    """Density, surface gravity and escape velocity, with a rough composition class.

    Args:
        mass_earth: Planet mass in Earth masses.
        radius_earth: Planet radius in Earth radii.
    """
    if mass_earth <= 0 or radius_earth <= 0:
        return err("inputs must be positive", source=SOURCE, url=URL)
    m = mass_earth * EARTH_MASS_KG
    r = radius_earth * EARTH_RADIUS_M
    volume = (4 / 3) * math.pi * r**3
    density = m / volume
    gravity = G * m / r**2
    v_esc = math.sqrt(2 * G * m / r)
    if radius_earth < 1.6 and density > 3000:
        composition = "likely rocky / terrestrial"
    elif radius_earth < 4.0:
        composition = "likely sub-Neptune with volatile envelope"
    else:
        composition = "likely gas giant"
    return ok(
        {
            "density_kg_m3": density,
            "density_relative_to_earth": density / 5514.0,
            "surface_gravity_m_s2": gravity,
            "surface_gravity_g": gravity / 9.80665,
            "escape_velocity_km_s": v_esc / 1000.0,
            "composition_class": composition,
        },
        source=SOURCE,
        url=URL,
    )


@mcp.tool()
def convert_distance(value: float, from_unit: str, to_unit: str) -> dict:
    """Convert between astronomical distance units.

    Args:
        value: Numeric value to convert.
        from_unit: One of ``au``, ``km``, ``pc``, ``ly``, ``lunar_distance``, ``m``.
        to_unit: Same set of units.
    """
    to_m = {
        "m": 1.0,
        "km": 1e3,
        "au": AU_M,
        "ly": 9.4607304725808e15,
        "pc": 9.4607304725808e15 * PC_LY,
        "lunar_distance": 3.844e8,
    }
    f, t = from_unit.lower().strip(), to_unit.lower().strip()
    if f not in to_m or t not in to_m:
        return err(f"units must be one of {sorted(to_m)}", source=SOURCE, url=URL)
    result = value * to_m[f] / to_m[t]
    return ok(
        {"input": value, "from": f, "to": t, "result": result},
        source=SOURCE,
        url=URL,
    )


@mcp.tool()
def light_travel_time(distance: float, unit: str = "pc") -> dict:
    """How long light takes to cross a distance.

    Args:
        distance: Numeric distance.
        unit: ``pc``, ``ly``, ``au`` or ``km``.
    """
    converted = convert_distance(distance, unit, "ly")
    if not converted.get("ok"):
        return converted
    years = converted["data"]["result"]
    return ok(
        {"light_travel_years": years, "light_travel_days": years * 365.25},
        source=SOURCE,
        url=URL,
    )


@mcp.tool()
def compare_values(actual: float, expected: float, tolerance_percent: float = 5.0) -> dict:
    """Check whether a stated number matches a reference within tolerance.

    Used by the Critic agent to verify LLM-stated numbers against tool data.

    Args:
        actual: The number as stated.
        expected: The number from an authoritative tool.
        tolerance_percent: Allowed relative difference.
    """
    if expected == 0:
        within = abs(actual) <= 1e-12
        rel = 0.0 if within else float("inf")
    else:
        rel = abs(actual - expected) / abs(expected) * 100.0
        within = rel <= tolerance_percent
    return ok(
        {
            "actual": actual,
            "expected": expected,
            "absolute_difference": abs(actual - expected),
            "relative_difference_percent": rel,
            "within_tolerance": within,
            "tolerance_percent": tolerance_percent,
        },
        source=SOURCE,
        url=URL,
    )


@mcp.tool()
def summarise_statistics(values: list[float], label: str = "values") -> dict:
    """Deterministic descriptive statistics for a list of numbers.

    Args:
        values: Numbers to summarise.
        label: Human-readable label for the series.
    """
    nums: list[float] = [float(v) for v in values if isinstance(v, (int, float))]
    if not nums:
        return err("no numeric values supplied", source=SOURCE, url=URL)
    nums_sorted = sorted(nums)
    n = len(nums_sorted)
    mean = sum(nums_sorted) / n
    variance = sum((x - mean) ** 2 for x in nums_sorted) / n
    median = (
        nums_sorted[n // 2]
        if n % 2
        else (nums_sorted[n // 2 - 1] + nums_sorted[n // 2]) / 2
    )
    stats: dict[str, Any] = {
        "label": label,
        "count": n,
        "min": nums_sorted[0],
        "max": nums_sorted[-1],
        "mean": mean,
        "median": median,
        "stdev": math.sqrt(variance),
        "sum": sum(nums_sorted),
    }
    return ok(stats, source=SOURCE, url=URL)


if __name__ == "__main__":
    mcp.run(transport="stdio")
