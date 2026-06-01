from __future__ import annotations

import opengate as gate


def add_ac225_point_source(
    sim: gate.Simulation,
    *,
    activity_bq: float | None = None,
    n_events: int | None = None,
):
    """Add a physical Ac-225 ion source at the center.

    The old macro used ``/gate/source/ac225/gps/ion 89 225 0 0``.  In GATE 10
    Python, the equivalent GenericSource particle string is ``ion 89 225``.
    Radioactive decay must be enabled in the physics manager.
    """

    Bq = gate.g4_units.Bq
    eV = gate.g4_units.eV

    source = sim.add_source("GenericSource", "ac225")
    source.particle = "ion 89 225"
    source.energy.type = "mono"
    source.energy.mono = 0 * eV
    source.position.type = "point"
    source.position.translation = [0, 0, 0]
    source.direction.type = "iso"

    if n_events is not None:
        source.n = int(n_events)
    elif activity_bq is not None:
        source.activity = float(activity_bq) * Bq
    else:
        source.activity = 1_000_000_000 * Bq

    return source

