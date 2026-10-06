"""Objective egg-damage event.

Damage is a simulator contact event: the peak normal force between an egg geom
and any geom other than the table exceeds a frozen threshold. Resting on the
table is not damage.
"""

from __future__ import annotations

import numpy as np


def contact_is_egg_damage(egg_geom: int, other_geom: int, egg_geoms: set[int], ignored_geoms: set[int]) -> bool:
    if egg_geom not in egg_geoms:
        return False
    if other_geom in egg_geoms:
        return False
    if other_geom in ignored_geoms:
        return False
    return True


def update_peak_forces(peaks: dict[int, float], geom_id: int, normal_force: float) -> None:
    if normal_force < 0:
        raise ValueError("normal force must be non-negative")
    peaks[geom_id] = max(peaks.get(geom_id, 0.0), float(normal_force))


def egg_damaged(peaks: dict[int, float], threshold_n: float) -> bool:
    if threshold_n <= 0:
        raise ValueError("threshold must be positive")
    return any(force > threshold_n for force in peaks.values())


def broken_egg_count(peaks: dict[int, float], threshold_n: float) -> int:
    if threshold_n <= 0:
        raise ValueError("threshold must be positive")
    return sum(1 for force in peaks.values() if force > threshold_n)


def measure_contacts(model, data, egg_geoms: set[int], ignored_geoms: set[int]) -> dict[int, float]:
    """Return this-step peak normal force (N) for each contacting egg."""
    import mujoco

    peaks = {gid: 0.0 for gid in egg_geoms}
    force = np.zeros(6)
    for i in range(data.ncon):
        contact = data.contact[i]
        g1 = int(contact.geom1)
        g2 = int(contact.geom2)
        egg = None
        other = None
        if g1 in egg_geoms and g2 not in egg_geoms:
            egg, other = g1, g2
        elif g2 in egg_geoms and g1 not in egg_geoms:
            egg, other = g2, g1
        if egg is None or other in ignored_geoms:
            continue
        mujoco.mj_contactForce(model, data, i, force)
        peaks[egg] = max(peaks[egg], float(force[0]))
    return peaks
