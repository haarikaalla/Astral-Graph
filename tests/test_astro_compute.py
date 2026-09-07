"""The numbers must be right.

Every quantitative claim in AstralGraph comes from the ``astro_compute`` MCP
server, never from the model. These tests pin its formulas to values that can be
checked by hand.
"""

from __future__ import annotations

import math

import pytest

from mcp_servers.astro_compute_server import (
    convert_distance,
    equilibrium_temperature,
    habitable_zone,
    impact_energy,
    light_travel_time,
    orbital_period,
    planet_bulk_properties,
    semi_major_axis,
    torino_scale_band,
    transit_depth,
)


def data(envelope: dict) -> dict:
    assert envelope["ok"], envelope
    assert envelope["source"]["name"]
    return envelope["data"]


def test_impact_energy_matches_hand_calculation():
    """m = rho*(pi/6)*d^3 = 1.5708e9 kg; E = 0.5*m*v^2 = 3.1416e17 J = 75.1 Mt."""
    result = data(impact_energy(diameter_m=100, velocity_km_s=20, density_kg_m3=3000))
    assert result["mass_kg"] == pytest.approx(1.5708e9, rel=1e-3)
    assert result["energy_joules"] == pytest.approx(3.1416e17, rel=1e-3)
    assert result["energy_megatons_tnt"] == pytest.approx(75.1, rel=0.02)


def test_impact_energy_scales_with_cube_of_diameter():
    small = data(impact_energy(diameter_m=100, velocity_km_s=20))["energy_joules"]
    large = data(impact_energy(diameter_m=200, velocity_km_s=20))["energy_joules"]
    assert large / small == pytest.approx(8.0, rel=1e-6)


def test_impact_energy_scales_with_square_of_velocity():
    slow = data(impact_energy(diameter_m=100, velocity_km_s=10))["energy_joules"]
    fast = data(impact_energy(diameter_m=100, velocity_km_s=20))["energy_joules"]
    assert fast / slow == pytest.approx(4.0, rel=1e-6)


def test_impact_energy_rejects_nonsense_input():
    assert impact_energy(diameter_m=-5, velocity_km_s=20)["ok"] is False


def test_torino_scale_zero_when_no_impact_probability():
    result = data(torino_scale_band(energy_megatons=75.1, impact_probability=0.0))
    assert result["torino_level"] == 0


def test_parsec_to_light_year():
    result = data(convert_distance(value=1, from_unit="pc", to_unit="ly"))
    assert result["result"] == pytest.approx(3.26156, rel=1e-4)


def test_au_to_km():
    result = data(convert_distance(value=1, from_unit="au", to_unit="km"))
    assert result["result"] == pytest.approx(1.495978707e8, rel=1e-6)


def test_distance_conversion_round_trips():
    out = data(convert_distance(value=4.24, from_unit="ly", to_unit="pc"))["result"]
    back = data(convert_distance(value=out, from_unit="pc", to_unit="ly"))["result"]
    assert back == pytest.approx(4.24, rel=1e-9)


def test_light_travel_time_one_light_year():
    result = data(light_travel_time(distance=1, unit="ly"))
    assert result["light_travel_years"] == pytest.approx(1.0, rel=1e-6)


def test_keplers_third_law_earth():
    """1 AU around 1 solar mass must give 1 year (365.25 d)."""
    result = data(orbital_period(semi_major_axis_au=1.0, star_mass_solar=1.0))
    assert result["period_days"] == pytest.approx(365.25, rel=0.01)


def test_semi_major_axis_is_inverse_of_period():
    result = data(semi_major_axis(period_days=365.25, star_mass_solar=1.0))
    assert result["semi_major_axis_au"] == pytest.approx(1.0, rel=0.01)


def test_equilibrium_temperature_of_earth():
    """Earth with albedo 0.3 gives ~255 K."""
    result = data(equilibrium_temperature(
        star_teff_k=5772, star_radius_solar=1.0,
        semi_major_axis_au=1.0, albedo=0.3,
    ))
    assert result["equilibrium_temperature_k"] == pytest.approx(255, abs=6)


def test_habitable_zone_of_the_sun_matches_kopparapu():
    result = data(habitable_zone(star_teff_k=5772, star_luminosity_solar=1.0))
    inner, outer = result["conservative_hz_au"]
    assert inner == pytest.approx(0.99, abs=0.05)
    assert outer == pytest.approx(1.70, abs=0.10)
    assert "Kopparapu" in result["reference"]


def test_transit_depth_of_earth_across_the_sun():
    """(Rearth/Rsun)^2 = 84 ppm."""
    result = data(transit_depth(planet_radius_earth=1.0, star_radius_solar=1.0))
    assert result["depth_ppm"] == pytest.approx(84, abs=3)


def test_transit_depth_is_the_square_of_the_radius_ratio():
    one = data(transit_depth(planet_radius_earth=1.0, star_radius_solar=1.0))["depth_ppm"]
    two = data(transit_depth(planet_radius_earth=2.0, star_radius_solar=1.0))["depth_ppm"]
    assert two / one == pytest.approx(4.0, rel=1e-6)


def test_planet_bulk_properties_of_earth():
    result = data(planet_bulk_properties(radius_earth=1.0, mass_earth=1.0))
    assert result["density_kg_m3"] == pytest.approx(5514, rel=0.02)
    assert result["surface_gravity_m_s2"] == pytest.approx(9.8, rel=0.02)
    assert result["escape_velocity_km_s"] == pytest.approx(11.2, rel=0.02)
    assert result["composition_class"] == "likely rocky / terrestrial"


def test_every_result_carries_provenance():
    for envelope in (
        impact_energy(diameter_m=50, velocity_km_s=15),
        convert_distance(value=2, from_unit="au", to_unit="km"),
        transit_depth(planet_radius_earth=1.0, star_radius_solar=1.0),
    ):
        assert envelope["source"]["url"].startswith("local://")
        assert envelope["source"]["retrieved_at"]
