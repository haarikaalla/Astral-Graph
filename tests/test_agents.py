"""Agent planning tests — the deterministic layer that works without an LLM key."""

from __future__ import annotations

import pytest

from agents.base import rank_facts
from agents.events_agent import CITY_COORDS
from agents.neo_agent import explicit_impact_params
from tests.conftest import make_fact


# --------------------------------------------------------------------------- #
# Impactor parameter extraction (question numbers must beat catalogue numbers)
# --------------------------------------------------------------------------- #
def test_extracts_diameter_velocity_and_density():
    params = explicit_impact_params(
        "What is the impact energy in megatons of TNT of a 100 metre stony asteroid "
        "with a density of 3000 kg/m3 striking at 20 km/s?"
    )
    assert params == {"diameter_m": 100.0, "velocity_km_s": 20.0, "density_kg_m3": 3000.0}


def test_defaults_density_when_unstated():
    params = explicit_impact_params("A 50 m asteroid hitting at 17 km/s")
    assert params is not None
    assert params["density_kg_m3"] == 3000.0


def test_accepts_kilometre_sized_impactors():
    params = explicit_impact_params("a 1.5 km wide asteroid at 25 km/s")
    assert params is not None
    assert params["diameter_m"] == 1500.0


@pytest.mark.parametrize(
    "question",
    [
        "Which asteroids pass close to Earth today?",
        "How dangerous is Apophis?",
        "What is 20 km/s in miles per hour?",  # velocity but no impactor size
    ],
)
def test_returns_none_without_a_full_parameter_set(question):
    assert explicit_impact_params(question) is None


# --------------------------------------------------------------------------- #
# Fact ranking (drives the no-LLM answer)
# --------------------------------------------------------------------------- #
def test_computed_facts_outrank_incidental_catalogue_rows():
    facts = [
        make_fact("(2019 LV)", "close_approach.miss_distance_km", 35743376.7, "km"),
        make_fact("(2014 DV110)", "close_approach.relative_velocity_km_s", 14.37, "km/s"),
        make_fact("computation:impact_energy", "energy_megatons_tnt", 75.1, "Mt TNT",
                  server="astro_compute", source="AstralGraph deterministic calculator"),
    ]
    ranked = rank_facts(facts, "What is the impact energy in megatons of a 100 m asteroid?")
    assert ranked[0].subject == "computation:impact_energy"


def test_ranking_follows_question_vocabulary():
    facts = [
        make_fact("Apophis", "diameter_max_m", 375.0, "m"),
        make_fact("Bennu", "diameter_max_m", 565.0, "m"),
    ]
    ranked = rank_facts(facts, "How big is Bennu?")
    assert ranked[0].subject == "Bennu"


def test_ranking_preserves_every_fact():
    facts = [make_fact(f"obj{i}", "value", float(i)) for i in range(20)]
    assert len(rank_facts(facts, "anything")) == 20


# --------------------------------------------------------------------------- #
# Events agent geography
# --------------------------------------------------------------------------- #
def test_london_coordinates_are_correct():
    lat, lon = CITY_COORDS["london"]
    assert lat == pytest.approx(51.5074, abs=0.01)
    assert lon == pytest.approx(-0.1278, abs=0.01)


def test_city_coordinates_are_physically_valid():
    for city, (lat, lon) in CITY_COORDS.items():
        assert -90 <= lat <= 90, city
        assert -180 <= lon <= 180, city
