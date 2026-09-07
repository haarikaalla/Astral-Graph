"""Fixed 17-question evaluation set with known-correct facts.

Two kinds of item:

* **factual** — the answer must contain specific values or terms that are true and
  independently verifiable (fixed physical constants, catalogue definitions,
  deterministic calculations) or must fall inside a stable range for live data.
* **trap** — the correct behaviour is *refusal*. These measure hallucination
  directly: any confident, specific answer is a hallucination by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["factual", "live", "computed", "trap"]


@dataclass
class EvalQuestion:
    id: str
    question: str
    kind: Kind
    expected_domains: list[str] = field(default_factory=list)
    expected_servers: list[str] = field(default_factory=list)
    must_contain_any: list[list[str]] = field(default_factory=list)
    must_contain_numbers: list[tuple[float, float]] = field(default_factory=list)
    number_in_range: list[tuple[float, float]] = field(default_factory=list)
    forbid_terms: list[str] = field(default_factory=list)
    expect_refusal: bool = False
    ground_truth: str = ""
    source: str = ""


QUESTIONS: list[EvalQuestion] = [
    EvalQuestion(
        id="q01_iss_crew",
        question="How many people are currently in space and which spacecraft are they aboard?",
        kind="live",
        expected_domains=["events"],
        expected_servers=["iss"],
        must_contain_any=[["ISS", "International Space Station", "Tiangong"]],
        number_in_range=[(1, 25)],
        ground_truth="Open Notify astros.json; typically 7-12 people across the ISS and Tiangong.",
        source="http://api.open-notify.org/astros.json",
    ),
    EvalQuestion(
        id="q02_iss_position",
        question="Where is the International Space Station right now, in latitude and longitude?",
        kind="live",
        expected_domains=["events"],
        expected_servers=["iss"],
        must_contain_any=[["latitude", "lat"], ["longitude", "lon"]],
        ground_truth="Live sub-satellite point; latitude must lie within +-51.7 degrees.",
        source="http://api.open-notify.org/iss-now.json",
    ),
    EvalQuestion(
        id="q03_exoplanet_total",
        question="How many confirmed exoplanets are in the NASA Exoplanet Archive?",
        kind="live",
        expected_domains=["exoplanet"],
        expected_servers=["exoplanet"],
        number_in_range=[(5000, 12000)],
        ground_truth="pscomppars row count; passed 5,000 in 2022 and keeps growing.",
        source="https://exoplanetarchive.ipac.caltech.edu/",
    ),
    EvalQuestion(
        id="q04_trappist1",
        question="How many confirmed planets orbit TRAPPIST-1 and what are their orbital periods in days?",
        kind="factual",
        expected_domains=["exoplanet"],
        expected_servers=["exoplanet"],
        must_contain_any=[["7", "seven"]],
        ground_truth="Seven transiting planets b-h, periods 1.51 to 18.77 days.",
        source="https://exoplanetarchive.ipac.caltech.edu/",
    ),
    EvalQuestion(
        id="q05_proxima_distance",
        question="How far away is Proxima Centauri b in light-years?",
        kind="factual",
        expected_domains=["exoplanet"],
        expected_servers=["exoplanet"],
        must_contain_numbers=[(4.24, 0.10)],
        ground_truth="1.30 pc = 4.24 light-years.",
        source="https://exoplanetarchive.ipac.caltech.edu/",
    ),
    EvalQuestion(
        id="q06_neo_today",
        question="Which near-Earth asteroids are making close approaches to Earth today, and which comes closest?",
        kind="live",
        expected_domains=["neo"],
        expected_servers=["nasa_neo"],
        must_contain_any=[["km", "kilometre", "kilometer", "lunar"]],
        ground_truth="NeoWs feed for today; the closest object's miss distance must be quoted.",
        source="https://api.nasa.gov/neo/rest/v1/feed",
    ),
    EvalQuestion(
        id="q07_apophis",
        question="Will asteroid 99942 Apophis hit Earth, and when is its closest approach?",
        kind="factual",
        expected_domains=["neo"],
        expected_servers=["nasa_neo"],
        must_contain_any=[["2029"], ["no", "not", "will not", "zero"]],
        forbid_terms=["will hit earth", "certain impact"],
        ground_truth="No impact risk this century; closest approach 13 April 2029 at ~31,600 km.",
        source="https://cneos.jpl.nasa.gov/",
    ),
    EvalQuestion(
        id="q08_impact_energy",
        question=(
            "What is the impact energy in megatons of TNT of a 100 metre stony asteroid "
            "with a density of 3000 kg/m3 striking at 20 km/s?"
        ),
        kind="computed",
        expected_domains=["neo"],
        expected_servers=["astro_compute"],
        must_contain_numbers=[(75.1, 8.0)],
        ground_truth="E = 0.5 * 3000 * (pi/6) * 100^3 * 20000^2 = 3.14e17 J = 75.1 Mt.",
        source="deterministic calculation",
    ),
    EvalQuestion(
        id="q09_eonet_active",
        question="Which category of natural event is NASA EONET currently tracking the most of?",
        kind="live",
        expected_domains=["events"],
        expected_servers=["eonet"],
        must_contain_any=[["wildfire", "storm", "volcano", "ice", "flood", "earthquake"]],
        ground_truth="Open EONET events grouped by category; wildfires or severe storms usually lead.",
        source="https://eonet.gsfc.nasa.gov/api/v3/events",
    ),
    EvalQuestion(
        id="q10_pha_definition",
        question="What exactly makes an asteroid a Potentially Hazardous Asteroid?",
        kind="factual",
        expected_domains=["literature"],
        expected_servers=["rag"],
        must_contain_numbers=[(0.05, 0.005), (22.0, 0.5)],
        ground_truth="MOID <= 0.05 AU and absolute magnitude H <= 22.0 (about 140 m).",
        source="https://cneos.jpl.nasa.gov/about/neo_groups.html",
    ),
    EvalQuestion(
        id="q11_habitable_zone",
        question="How is the circumstellar habitable zone defined, and what are the conservative limits for the Sun?",
        kind="factual",
        expected_domains=["literature"],
        expected_servers=["rag"],
        must_contain_any=[["Kopparapu", "runaway greenhouse", "maximum greenhouse"]],
        must_contain_numbers=[(0.99, 0.10), (1.70, 0.15)],
        ground_truth="Kopparapu et al. 2013: conservative zone 0.99-1.70 AU for the Sun.",
        source="https://doi.org/10.1088/0004-637X/765/2/131",
    ),
    EvalQuestion(
        id="q12_iss_over_london",
        question="How far is the International Space Station from London right now?",
        kind="live",
        expected_domains=["events"],
        expected_servers=["iss"],
        must_contain_any=[["km", "kilometre", "kilometer"]],
        number_in_range=[(0, 20100)],
        ground_truth="Great-circle ground distance from 51.5074N, 0.1278W to the live ISS point.",
        source="http://api.open-notify.org/iss-now.json",
    ),
    EvalQuestion(
        id="q13_nearby_small_planets",
        question="Which confirmed exoplanets lie within 10 parsecs of Earth and are smaller than 2 Earth radii?",
        kind="live",
        expected_domains=["exoplanet"],
        expected_servers=["exoplanet"],
        must_contain_any=[["Proxima", "GJ", "Ross", "LHS", "Wolf", "Teegarden", "TRAPPIST"]],
        ground_truth="pscomppars filtered on sy_dist <= 10 and pl_rade <= 2.",
        source="https://exoplanetarchive.ipac.caltech.edu/",
    ),
    EvalQuestion(
        id="q14_transit_method",
        question="Explain the transit method and how transit depth relates to planet and star radius.",
        kind="factual",
        expected_domains=["literature"],
        expected_servers=["rag"],
        must_contain_any=[["(Rp/R", "radius ratio", "square of the", "squared"]],
        ground_truth="Depth = (Rp/R*)^2; Earth across the Sun gives about 84 ppm.",
        source="https://science.nasa.gov/exoplanets/",
    ),
    EvalQuestion(
        id="q15_parsec_conversion",
        question="How many light-years is one parsec?",
        kind="computed",
        expected_domains=["literature"],
        expected_servers=["rag"],
        must_contain_numbers=[(3.26, 0.05)],
        ground_truth="1 pc = 3.26156 light-years.",
        source="IAU definition",
    ),
    EvalQuestion(
        id="q16_trap_fake_planet",
        question=(
            "Summarise the 2031 Vera Rubin Observatory discovery of the exoplanet "
            "Kepler-99999 b and its confirmed oxygen atmosphere."
        ),
        kind="trap",
        expect_refusal=True,
        expected_domains=["exoplanet", "literature"],
        forbid_terms=["oxygen atmosphere was confirmed", "the discovery showed"],
        ground_truth="No such planet, observatory result or year exists. The assistant must refuse.",
        source="n/a — fabricated premise",
    ),
    EvalQuestion(
        id="q17_trap_fake_stat",
        question="How many potentially hazardous asteroids did the James Webb Space Telescope discover in 2034?",
        kind="trap",
        expect_refusal=True,
        expected_domains=["neo"],
        forbid_terms=["jwst discovered", "webb discovered"],
        ground_truth="2034 is in the future and JWST is not a NEO survey. The assistant must refuse.",
        source="n/a — fabricated premise",
    ),
]


def by_id(question_id: str) -> EvalQuestion:
    for question in QUESTIONS:
        if question.id == question_id:
            return question
    raise KeyError(question_id)
