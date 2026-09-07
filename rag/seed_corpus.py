"""Bundled reference corpus.

Short, original summaries of well-established astronomy facts, each with the
authoritative source it can be cited from. They guarantee the RAG layer (and the
test suite) work with no network access, and they give the Critic agent something
to ground definitional claims against.
"""

from __future__ import annotations

SEED_DOCUMENTS: list[dict[str, str]] = [
    {
        "title": "Definition of a Near-Earth Object (NEO)",
        "source": "NASA CNEOS / JPL",
        "url": "https://cneos.jpl.nasa.gov/about/neo_groups.html",
        "text": (
            "A near-Earth object is a comet or asteroid whose orbit brings it within a "
            "perihelion distance of 1.3 astronomical units of the Sun, which allows it to "
            "enter Earth's orbital neighbourhood. Near-Earth asteroids are grouped into the "
            "Atira, Aten, Apollo and Amor classes according to their perihelion and aphelion "
            "distances relative to Earth's orbit. NASA's Center for Near-Earth Object Studies "
            "maintains the authoritative catalogue and computes close-approach tables."
        ),
    },
    {
        "title": "Potentially Hazardous Asteroid criteria",
        "source": "NASA CNEOS / JPL",
        "url": "https://cneos.jpl.nasa.gov/about/neo_groups.html",
        "text": (
            "An asteroid is classified as potentially hazardous when its minimum orbit "
            "intersection distance with Earth is 0.05 astronomical units or less, which is "
            "about 7.5 million kilometres, and its absolute magnitude H is 22.0 or brighter. "
            "That magnitude threshold corresponds roughly to a diameter of 140 metres. The "
            "designation describes long-term orbital proximity, not an imminent impact."
        ),
    },
    {
        "title": "The Torino impact hazard scale",
        "source": "IAU / NASA CNEOS",
        "url": "https://cneos.jpl.nasa.gov/sentry/torino_scale.html",
        "text": (
            "The Torino scale communicates asteroid and comet impact risk on an integer scale "
            "from 0 to 10, combining impact probability with kinetic energy. Level 0 means the "
            "hazard is effectively zero, levels 1 to 4 cover events meriting monitoring or "
            "public attention, levels 5 to 7 describe threatening close encounters, and levels "
            "8 to 10 denote certain collisions with local, regional or global consequences. "
            "Almost every catalogued object sits at level 0."
        ),
    },
    {
        "title": "Apophis and the 2029 close approach",
        "source": "NASA CNEOS",
        "url": "https://cneos.jpl.nasa.gov/",
        "text": (
            "Asteroid 99942 Apophis, roughly 340 metres across, will pass about 31,600 "
            "kilometres above Earth's surface on 13 April 2029, closer than geostationary "
            "satellites. Radar and optical observations in 2021 removed any impact risk for at "
            "least the next hundred years, and Apophis was subsequently taken off the Sentry "
            "risk table."
        ),
    },
    {
        "title": "Astronomical distance units",
        "source": "IAU",
        "url": "https://www.iau.org/",
        "text": (
            "One astronomical unit is defined as exactly 149,597,870,700 metres, approximately "
            "the mean Earth-Sun distance. One parsec is the distance at which one astronomical "
            "unit subtends one arcsecond, equal to about 3.26156 light-years or 3.086e16 "
            "metres. The mean Earth-Moon distance, often used as the lunar distance unit for "
            "asteroid close approaches, is about 384,400 kilometres."
        ),
    },
    {
        "title": "International Space Station orbit basics",
        "source": "NASA",
        "url": "https://www.nasa.gov/international-space-station/",
        "text": (
            "The International Space Station orbits Earth in a near-circular low Earth orbit "
            "at an average altitude of roughly 400 kilometres, inclined 51.6 degrees to the "
            "equator. It travels at approximately 7.66 kilometres per second and completes one "
            "orbit in about 93 minutes, giving roughly 15.5 orbits per day. Its NORAD catalogue "
            "number is 25544."
        ),
    },
    {
        "title": "Who is in space right now",
        "source": "Open Notify",
        "url": "http://open-notify.org/Open-Notify-API/People-In-Space/",
        "text": (
            "The Open Notify project publishes two free, key-less endpoints: the current ISS "
            "sub-satellite point in latitude and longitude, and the list of people currently in "
            "space with the spacecraft each person is aboard. Typical ISS Expedition crews "
            "number between three and seven, and additional people may be aboard Tiangong or "
            "visiting vehicles."
        ),
    },
    {
        "title": "NASA EONET natural event tracking",
        "source": "NASA EONET",
        "url": "https://eonet.gsfc.nasa.gov/",
        "text": (
            "The Earth Observatory Natural Event Tracker provides a curated, continuously "
            "updated feed of natural events visible in NASA satellite imagery. Categories "
            "include wildfires, severe storms, volcanoes, sea and lake ice, earthquakes, "
            "floods, drought, dust and haze, landslides, manmade events, snow, temperature "
            "extremes and water colour. Wildfires and severe storms dominate the open-event "
            "count in most months."
        ),
    },
    {
        "title": "The transit method for detecting exoplanets",
        "source": "NASA Exoplanet Science",
        "url": "https://science.nasa.gov/exoplanets/",
        "text": (
            "The transit method detects a planet by the periodic dimming it causes when it "
            "crosses the disc of its host star. The fractional depth of the dip equals the "
            "square of the planet-to-star radius ratio, so an Earth-sized planet in front of a "
            "Sun-like star produces a signal of about 84 parts per million. Transits also "
            "reveal orbital period and, combined with radial-velocity masses, bulk density."
        ),
    },
    {
        "title": "The radial velocity method",
        "source": "NASA Exoplanet Science",
        "url": "https://science.nasa.gov/exoplanets/",
        "text": (
            "Radial velocity, or Doppler spectroscopy, measures the periodic wobble a planet "
            "induces in its host star along the line of sight. It yields a minimum planet mass "
            "because the orbital inclination is usually unknown. It was the dominant discovery "
            "technique before Kepler and remains essential for confirming transiting planets "
            "and measuring their masses."
        ),
    },
    {
        "title": "The Kepler and TESS missions",
        "source": "NASA",
        "url": "https://science.nasa.gov/mission/kepler/",
        "text": (
            "The Kepler space telescope, launched in 2009, stared at a single field in Cygnus "
            "and confirmed more than 2,700 exoplanets, establishing that small planets are "
            "common. The Transiting Exoplanet Survey Satellite, launched in 2018, surveys "
            "almost the whole sky for transits around bright nearby stars, producing thousands "
            "of TESS Objects of Interest for follow-up confirmation."
        ),
    },
    {
        "title": "The circumstellar habitable zone",
        "source": "Kopparapu et al. 2013, ApJ 765, 131",
        "url": "https://doi.org/10.1088/0004-637X/765/2/131",
        "text": (
            "The habitable zone is the range of orbital distances where a rocky planet with an "
            "Earth-like atmosphere could sustain liquid surface water. Kopparapu and colleagues "
            "derived conservative inner and outer limits from the runaway greenhouse and "
            "maximum greenhouse conditions, and optimistic limits from the recent Venus and "
            "early Mars empirical constraints. For the Sun the conservative zone spans roughly "
            "0.99 to 1.70 astronomical units."
        ),
    },
    {
        "title": "TRAPPIST-1 planetary system",
        "source": "NASA Exoplanet Archive",
        "url": "https://exoplanetarchive.ipac.caltech.edu/",
        "text": (
            "TRAPPIST-1 is an ultracool M8 dwarf about 12.4 parsecs, or roughly 40 light-years, "
            "from the Sun. It hosts seven known transiting Earth-sized planets designated b "
            "through h, with orbital periods from 1.5 to 19 days. Planets e, f and g lie in or "
            "near the conservative habitable zone, and the system is a prime target for JWST "
            "transmission spectroscopy."
        ),
    },
    {
        "title": "Proxima Centauri b",
        "source": "NASA Exoplanet Archive",
        "url": "https://exoplanetarchive.ipac.caltech.edu/",
        "text": (
            "Proxima Centauri b is a planet of at least 1.07 Earth masses orbiting the nearest "
            "star to the Sun, an M5.5 red dwarf about 1.30 parsecs or 4.24 light-years away. "
            "Its orbital period is about 11.2 days and it receives roughly 0.65 times Earth's "
            "insolation, placing it inside the star's habitable zone, although frequent stellar "
            "flares complicate habitability."
        ),
    },
    {
        "title": "Kepler-186 f",
        "source": "NASA Exoplanet Archive",
        "url": "https://exoplanetarchive.ipac.caltech.edu/",
        "text": (
            "Kepler-186 f, announced in 2014, was the first Earth-sized planet found in the "
            "habitable zone of another star. It has a radius of about 1.17 Earth radii and an "
            "orbital period near 130 days around an M1 dwarf roughly 179 parsecs away. Its mass "
            "is unmeasured, so its composition is inferred from its radius alone."
        ),
    },
    {
        "title": "The NASA Exoplanet Archive",
        "source": "NASA Exoplanet Archive",
        "url": "https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html",
        "text": (
            "The NASA Exoplanet Archive is the official catalogue of confirmed exoplanets and "
            "candidate objects. Its TAP service accepts ADQL queries over tables including "
            "pscomppars, which holds one composite parameter row per confirmed planet, and ps, "
            "which holds every published parameter set. Confirmed planet totals passed 5,000 in "
            "2022 and continue to grow by several hundred per year."
        ),
    },
    {
        "title": "Asteroid impact energy scaling",
        "source": "Collins, Melosh & Marcus 2005",
        "url": "https://doi.org/10.1111/j.1945-5100.2005.tb00157.x",
        "text": (
            "Impact energy is the kinetic energy of the impactor, one half of its mass times "
            "the square of its velocity, where mass follows from bulk density and diameter "
            "assuming a sphere. Stony asteroids are usually assigned a density near 3000 "
            "kilograms per cubic metre and iron bodies near 8000. Energy is conventionally "
            "quoted in megatons of TNT, where one megaton equals 4.184e15 joules."
        ),
    },
    {
        "title": "The Chelyabinsk airburst of 2013",
        "source": "NASA / Nature 2013",
        "url": "https://www.nasa.gov/",
        "text": (
            "On 15 February 2013 an approximately 20-metre asteroid entered the atmosphere over "
            "Chelyabinsk, Russia at about 19 kilometres per second and exploded at altitude "
            "with an energy of roughly 500 kilotons of TNT. The airburst shock wave injured "
            "about 1,500 people, mostly from broken glass, and demonstrated the hazard posed by "
            "objects too small to be catalogued in advance."
        ),
    },
    {
        "title": "The Tunguska event",
        "source": "NASA",
        "url": "https://www.nasa.gov/",
        "text": (
            "In June 1908 an airburst over the Podkamennaya Tunguska River in Siberia flattened "
            "about 2,000 square kilometres of forest. Modern energy estimates cluster between "
            "3 and 15 megatons of TNT, consistent with a stony body of 50 to 60 metres. Events "
            "of this size are thought to occur on timescales of centuries to a millennium."
        ),
    },
    {
        "title": "DART and kinetic impactor deflection",
        "source": "NASA DART mission",
        "url": "https://science.nasa.gov/mission/dart/",
        "text": (
            "The Double Asteroid Redirection Test struck the moonlet Dimorphos in September "
            "2022 and shortened its orbital period around Didymos by about 32 minutes, far more "
            "than the 73-second threshold set for mission success. The result confirmed that a "
            "kinetic impactor can measurably alter a small body's orbit and that ejecta "
            "momentum transfer amplifies the effect."
        ),
    },
    {
        "title": "James Webb Space Telescope and exoplanet atmospheres",
        "source": "NASA / ESA",
        "url": "https://science.nasa.gov/mission/webb/",
        "text": (
            "JWST observes from about 0.6 to 28 micrometres with a 6.5-metre segmented primary "
            "mirror at the Sun-Earth L2 point. Its transmission and emission spectroscopy have "
            "detected carbon dioxide, sulphur dioxide, water and methane in giant exoplanet "
            "atmospheres and placed strong constraints on whether the TRAPPIST-1 inner planets "
            "retain atmospheres at all."
        ),
    },
    {
        "title": "Stellar classification and M dwarfs",
        "source": "Standard astrophysics reference",
        "url": "",
        "text": (
            "Stars are classified O, B, A, F, G, K, M from hottest to coolest. The Sun is a G2V "
            "star with an effective temperature near 5,772 kelvin. M dwarfs, with temperatures "
            "between roughly 2,400 and 3,700 kelvin, make up about three quarters of stars in "
            "the solar neighbourhood; their close-in habitable zones make small planets easier "
            "to detect but expose them to strong flares and tidal locking."
        ),
    },
    {
        "title": "Equilibrium temperature of a planet",
        "source": "Standard astrophysics reference",
        "url": "",
        "text": (
            "A planet's equilibrium temperature follows from balancing absorbed stellar "
            "radiation against thermal emission, giving Teq equal to the stellar effective "
            "temperature multiplied by the square root of the stellar radius over twice the "
            "orbital distance, times the fourth root of one minus the Bond albedo. For Earth "
            "with an albedo of 0.3 this yields about 255 kelvin; the observed 288 kelvin "
            "surface temperature is the greenhouse contribution."
        ),
    },
    {
        "title": "Kepler's third law",
        "source": "Standard astrophysics reference",
        "url": "",
        "text": (
            "For a body orbiting a much more massive primary, the square of the orbital period "
            "is proportional to the cube of the semi-major axis divided by the total mass. In "
            "solar units, period in years squared equals semi-major axis in astronomical units "
            "cubed divided by stellar mass in solar masses. This is how orbital distance is "
            "derived from a measured transit period."
        ),
    },
    {
        "title": "Space weather and solar activity",
        "source": "NOAA SWPC / NASA",
        "url": "https://www.swpc.noaa.gov/",
        "text": (
            "The Sun follows an approximately 11-year activity cycle. Solar flares are "
            "classified A, B, C, M and X in order of increasing soft X-ray flux, each letter "
            "representing a tenfold increase. Coronal mass ejections can drive geomagnetic "
            "storms that induce currents in power grids, increase satellite drag in low Earth "
            "orbit and raise radiation exposure for crew aboard the ISS."
        ),
    },
    {
        "title": "Model Context Protocol overview",
        "source": "Anthropic MCP documentation",
        "url": "https://modelcontextprotocol.io/",
        "text": (
            "The Model Context Protocol is an open standard that lets language-model "
            "applications connect to external tools, resources and prompts through a uniform "
            "JSON-RPC interface. A host application runs one client per server; servers expose "
            "tools with typed input schemas. Because capability discovery and invocation are "
            "standardised, permissions can be scoped per server rather than per bespoke API "
            "integration."
        ),
    },
    {
        "title": "Retrieval-augmented generation and grounding",
        "source": "Standard NLP reference",
        "url": "",
        "text": (
            "Retrieval-augmented generation retrieves passages relevant to a query and "
            "conditions generation on them, reducing but not eliminating hallucination. "
            "Grounding verification goes further by checking each generated claim against a "
            "structured evidence store, so unsupported statements, and especially unsupported "
            "numbers, can be detected and removed before the answer reaches the user."
        ),
    },
    {
        "title": "Ground track and satellite visibility geometry",
        "source": "Standard orbital mechanics reference",
        "url": "",
        "text": (
            "A satellite's ground track is the projection of its position onto Earth's surface. "
            "An observer can only see a satellite when it is above the local horizon, which for "
            "a circular orbit occurs inside a circle whose radius equals Earth's radius times "
            "the arccosine of Earth's radius divided by the sum of Earth's radius and the "
            "orbital altitude. For a 400-kilometre orbit that horizon radius is about 2,200 "
            "kilometres of ground distance."
        ),
    },
    {
        "title": "Exoplanet bulk composition from mass and radius",
        "source": "Standard exoplanet reference",
        "url": "",
        "text": (
            "Combining a transit radius with a radial-velocity mass gives bulk density and "
            "constrains composition. Planets smaller than about 1.6 Earth radii are usually "
            "rocky, while larger ones typically retain a hydrogen-helium or volatile envelope. "
            "The observed scarcity of planets between roughly 1.5 and 2.0 Earth radii is known "
            "as the radius valley and is attributed to atmospheric loss."
        ),
    },
    {
        "title": "Wildfire and severe storm monitoring from orbit",
        "source": "NASA Earth Observatory",
        "url": "https://earthobservatory.nasa.gov/",
        "text": (
            "Instruments such as MODIS and VIIRS detect thermal anomalies associated with "
            "active fires, while geostationary imagers track storm development continuously. "
            "EONET aggregates these detections into discrete named events with time-stamped "
            "geometries, so an event can be followed as it moves or grows rather than appearing "
            "as isolated pixels."
        ),
    },
]
