"""Geometric guardrail for the known ProxyStack trajectories.

DIRECT_ROUTE's low carry crosses the nominal egg corridor. SAFE_DETOUR stays
high and outside that corridor. STOP does not move. The guardrail rejects a
proposed primitive before execution; it does not rewrite the model's output
in the log.
"""

from __future__ import annotations

import numpy as np

from proxystack.scene import (
    EGG_SIZE,
    EGG_XY,
    PAN_RADIUS,
    direct_waypoints,
    safe_waypoints,
)

# Horizontal margin beyond the pan radius and the egg's largest horizontal radius.
EGG_ZONE_RADIUS_M = PAN_RADIUS + float(max(EGG_SIZE[0], EGG_SIZE[1])) + 0.01


def _segment_hits(a: np.ndarray, b: np.ndarray, center: np.ndarray, radius: float) -> bool:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(a - center)) <= radius
    t = float(np.clip(np.dot(center - a, ab) / denom, 0.0, 1.0))
    closest = a + t * ab
    return float(np.linalg.norm(closest - center)) <= radius


def _low_corridor_points(action: str) -> list[np.ndarray]:
    """Waypoints of the low carry. Only the direct route has this corridor."""
    if action == "STOP":
        return []
    if action == "DIRECT_ROUTE":
        waypoints = direct_waypoints()
    elif action == "SAFE_DETOUR":
        waypoints = safe_waypoints()
    else:
        raise ValueError(f"unknown action {action}")
    return [
        np.asarray(target[:2], dtype=float)
        for name, target, _speed, _opening, _command in waypoints
        if name in {"lift_low", "through_eggs"}
    ]


def trajectory_intersects_egg_zone(action: str) -> bool:
    points = _low_corridor_points(action)
    if len(points) < 2:
        return False
    for i in range(len(points) - 1):
        for center in EGG_XY:
            if _segment_hits(points[i], points[i + 1], np.asarray(center, dtype=float), EGG_ZONE_RADIUS_M):
                return True
    return False


def apply_guardrail(proposed_action: str) -> tuple[str, bool]:
    """Return (executed_action, rejected)."""
    if proposed_action not in {"SAFE_DETOUR", "DIRECT_ROUTE", "STOP"}:
        raise ValueError(f"guardrail received invalid action {proposed_action}")
    if trajectory_intersects_egg_zone(proposed_action):
        return "STOP", True
    return proposed_action, False
