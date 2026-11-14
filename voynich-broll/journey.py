"""The journey: the lie of the land and everything that moves, as plain formulas of time.

Every plant, bather, fish, ram, archer and creature is a flat card carrying cut-out pixels from
Yale's scans of Beinecke MS 408. Constructed: ground, water, sky, light, sizes and all motion.
"""
import math
from mathutils import noise, Vector

FPS = 24
DURATION = 40.0
POND = (9.5, 30.0, 7.5)            # x, y, radius
WATER_Z = -0.30
LANDMARK = (0.0, 92.0)
ARCHER = (-5.5, 60.0)


def sstep(a, b, v):
    k = min(max((v - a) / (b - a), 0.0), 1.0)
    return k * k * (3 - 2 * k)


def path_x(y):
    return 3.0 * math.sin(y / 17.0)


def valley_half_width(y):
    """Narrow valley at the start, opening to a meadow, closing again towards the landmark."""
    return 2.6 + 7.5 * sstep(20, 40, y) * (1 - sstep(64, 80, y))


def ground(x, y):
    d = abs(x - path_x(y))
    w = valley_half_width(y)
    hill = 6.5 * (1 - math.exp(-((max(d - w, 0) / 11.0) ** 2)))
    n = noise.noise(Vector((x * 0.035, y * 0.035, 0.0))) * 1.6 + noise.noise(Vector((x * 0.11, y * 0.11, 3.0))) * 0.35
    h = hill * (0.75 + 0.25 * n) + n * min(d / 6.0, 1.0) * 0.5
    px, py, pr = POND
    r = math.hypot(x - px, y - py)
    if r < pr + 6:
        shore = WATER_Z + 0.03
        if r < pr:
            h = shore - 1.3 * (1 - (r / pr) ** 2)
        else:
            k = sstep(0, 1, (r - pr) / 6.0)
            h = shore * (1 - k) + h * k
    return h


def camera_at(t):
    """Walks the path at a steady pace; the gaze moves from the path to the pond, the rams, the archer, the giant plant."""
    y = 2.0 + 1.6 * t
    x = path_x(y) - 0.4
    z = ground(x, y) + 1.65 + 0.03 * math.sin(t * 1.9)
    ya = y + 13.0
    ahead = Vector((path_x(ya), ya, ground(path_x(ya), ya) + 1.3))
    pond = Vector((POND[0] - 2.0, POND[1], WATER_Z + 0.7))
    rams = Vector((path_x(y + 9), y + 9, ground(path_x(y + 9), y + 9) + 1.0))
    archer = Vector((ARCHER[0], ARCHER[1], ground(*ARCHER) + 1.0))
    land = Vector((LANDMARK[0], LANDMARK[1], 13.0))
    tgt = ahead.copy()
    for target, t0, t1, t2, t3 in ((pond, 8.5, 12.0, 16.5, 19.5), (rams, 20.5, 23.0, 27.5, 29.5), (archer, 29.5, 31.5, 34.5, 36.5)):
        k = sstep(t0, t1, t) * (1 - sstep(t2, t3, t))
        tgt = tgt.lerp(target, 0.85 * k)
    tgt = tgt.lerp(land, 0.9 * sstep(35.5, 39.5, t))
    return Vector((x, y, z)), tgt


N_FISH = 4
def fish_pose(i, t):
    """The Pisces fish leap from the pond in arcs. Returns x, y, z, tilt, heading(+1/-1), visible."""
    px, py, pr = POND
    period = 2.9 + 0.45 * i
    u = ((t + 1.1 * i + 0.4) % period) / 1.3
    ang = 2.6 + i * 0.75
    cx, cy = px + math.cos(ang) * pr * 0.5, py + math.sin(ang) * pr * 0.5
    if u > 1.0:
        return cx, cy, WATER_Z - 2.0, 0.0, 1, False
    span, top = 3.2, 1.5 + 0.25 * (i % 2)
    h = 1 if i % 2 == 0 else -1
    x = cx + h * span * (u - 0.5)
    z = WATER_Z - 0.3 + top * 4 * u * (1 - u) * 1.2
    return x, cy, z, math.atan(top * 4 * (1 - 2 * u) * 1.2 / span), h, True


RAMS = [(0, 21.0, 46.0, 0.0), (1, 21.6, 47.5, 0.5), (0, 22.4, 50.0, 0.2), (1, 23.3, 52.0, 0.7)]     # (which drawing, start time, y, hop phase)
def ram_pose(k, t):
    """The Aries rams bound across the meadow, left to right. Returns x, y, z, tilt, visible."""
    which, t0, y, ph = RAMS[k]
    u = (t - t0) / 6.5
    x = path_x(y) - 11.0 + 22.0 * u
    hops = u * 8 + ph
    p = hops % 1.0
    z = ground(x, y) + 0.95 * 4 * p * (1 - p)
    tilt = math.atan(0.95 * 4 * (1 - 2 * p) / 2.75)
    return x, y, z, tilt, -0.05 <= u <= 1.05


def archer_pose(t):
    """The Sagittarius archer stands on his knoll, raises his aim and jerks with the shot. Returns lean (radians)."""
    aim = sstep(29.5, 32.0, t) * (1 - sstep(35.5, 37.0, t))
    recoil = 0.0
    for shot in (32.6, 34.4):
        dt = t - shot
        if 0 <= dt < 0.6: recoil += math.exp(-dt * 7) * math.sin(dt * 28) * 0.09
    return -0.10 * aim + recoil + 0.012 * math.sin(t * 1.3)


def creature_pose(t):
    y = 15.0
    u = (t - 2.0) / 5.5
    x = path_x(y) - 5.0 + 10.0 * u
    return x, y, ground(x, y) + 0.28 * abs(math.sin(u * math.pi * 7)), 0.0 <= u <= 1.0


def nymph_pose(i, phase, x0, y0, t):
    """Bathers bob and sway; some wade in slow circles; some duck under and come up again. Returns x, y, dz, lean, facing."""
    wade = 0.5 if i % 3 == 0 else 0.0
    x = x0 + wade * math.cos(t * 0.35 + phase); y = y0 + wade * math.sin(t * 0.35 + phase)
    dz = 0.04 * math.sin(t * 1.6 + phase)
    if i % 3 == 1:                                                   # ducks under every so often
        c = (t + phase * 2.0) % 9.0
        dz -= 0.75 * sstep(0, 0.7, c) * (1 - sstep(1.6, 2.4, c))
    lean = math.radians(3.0) * math.sin(t * 0.9 + phase)
    facing = -1 if (i % 3 == 0 and math.sin(t * 0.35 + phase) > 0) else 1
    return x, y, dz, lean, facing
