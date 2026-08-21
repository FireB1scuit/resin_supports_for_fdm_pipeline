"""The resin scaffold — the only support structure this project builds.

Four things are asserted here, in order of how much they matter:

1. **A shaft never passes through the model.** This is a guarantee, not a
   preference.
2. **The plate wins whenever it is reachable.** A shaft resting on the model is
   only ever allowed when there is genuinely no way down.
3. **It stays a scaffold.** Shafts stay thin and separate, arms fan off them to
   share, and cross-links tie the whole thing together. Nothing fuses into a
   thickening trunk — that would be an organic tree, which is the wrong
   structure for this project.
4. **The output is deterministic and stays on the right side of the plate.**
"""

from __future__ import annotations

import collections
import itertools
import math

import numpy as np
import pytest
import trimesh

from rsupport import mesh_io, presets, sampling
from rsupport.avoidance import AvoidanceField
from rsupport.raycast import DownRay
from rsupport.resin import (
    _LINKS_PER_SHAFT,
    _choose_links,
    _link_band,
    _link_shafts,
    _link_storeys,
    _neighbour_candidates,
    _plan_shafts,
    _rung_heights,
    _rung_spacing,
    _rungs,
    build_resin,
    link_angle,
)
from test_supports import (
    bracket,
    down_point,
    ledge_grid,
    printability_report,
    solid_box,
    table,
)

PARAMS = presets.from_nozzle(0.2)


def plan(model, points, params=None):
    """Run the planning half and hand back everything it produced."""
    params = params or PARAMS
    ray = DownRay(model)
    top = max(float(p.position[2]) for p in points)
    field = AvoidanceField(model, params, top_z=top)
    shafts, dropped, warnings = _plan_shafts(field, ray, points, params)
    return shafts, dropped, warnings, ray, field


def bar_points(z=10.0):
    """A row of contacts under the table's ledge."""
    return [down_point(float(x), 0.0, z) for x in np.arange(-7.0, 7.1, 1.5) if abs(x) > 2.0]


def hollow_box(outer=20.0, inner=12.0) -> trimesh.Trimesh:
    """A sealed box with a void inside — the plate is unreachable from in there."""
    shell = trimesh.creation.box(extents=[outer, outer, outer])
    cavity = trimesh.creation.box(extents=[inner, inner, inner])
    cavity.invert()
    return mesh_io.drop_to_bed(mesh_io.concat(shell, cavity))


# --------------------------------------------------------------------------- #
# 1. the hard constraint
# --------------------------------------------------------------------------- #


def _pierces(shafts, ray, samples=12):
    """Count sampled points along the shafts that are inside the model.

    Sampled along each shaft's actual axis, which is a polyline once the shaft
    has had to route around something. Asking `xy_at` rather than assuming a
    single XY is the whole difference between checking the strut that gets
    built and checking a straight line between its two ends.

    Skips the bottom of a model landing, which is deliberately sunk
    `tip_penetration` into the surface so its base is not a floating layer.
    """
    hits = 0
    for s in shafts:
        lo = s.land_z + PARAMS.tip_length if s.on_model else s.land_z
        if s.top_z - lo <= 1e-9:
            continue
        # At least a couple of samples per path vertex, so a detour cannot slip
        # between two of them.
        n = max(samples, len(s.path) * 2)
        z = np.linspace(lo, s.top_z, n)
        pts = np.array([[*s.xy_at(zz), zz] for zz in z])
        hits += int(ray.inside(pts).sum())
    return hits


def _arms_pierce(shafts, ray):
    """Arms and tips that cross the model on their way to a contact."""
    bad = 0
    for s in shafts:
        for a in s.arms:
            bad += int(ray.segment_blocked([a.attach], [a.elbow])[0])
            bad += int(ray.segment_blocked([a.elbow], [a.contact])[0])
    return bad


def test_no_shaft_passes_through_the_model():
    model = mesh_io.drop_to_bed(mesh_io.load("samples/synthetic_mini.stl"))
    points = sampling.place_points(model, PARAMS)
    shafts, _, _, ray, _ = plan(model, points)
    assert shafts
    assert _pierces(shafts, ray) == 0


def test_no_arm_or_tip_passes_through_the_model():
    """The shaft was never the only thing that could cross the sculpt. An arm
    reaching up to a contact is a strut like any other, and one that leaves a
    shaft standing on the wrong side of the model is a spear through the middle
    of it."""
    model = mesh_io.drop_to_bed(mesh_io.load("samples/synthetic_mini.stl"), lift=5.0)
    params = PARAMS.with_(lift_height=5.0)
    points = sampling.place_points(model, params)
    shafts, _, _, ray, _ = plan(model, points, params)
    assert shafts
    assert _arms_pierce(shafts, ray) == 0


def test_no_shaft_passes_through_an_awkward_shape():
    """The C-bracket: everything below a contact is model, and the only way out
    is sideways."""
    model = mesh_io.drop_to_bed(bracket())
    points = [down_point(-1.0, 0.0, 12.0), down_point(-3.0, 1.0, 12.0)]
    shafts, _, _, ray, _ = plan(model, points)
    assert shafts
    assert _pierces(shafts, ray) == 0
    assert _arms_pierce(shafts, ray) == 0


# --------------------------------------------------------------------------- #
# 1b. routing around, rather than stopping at
# --------------------------------------------------------------------------- #


def test_a_blocked_shaft_leans_around_the_obstruction_and_reaches_the_plate():
    """The C-bracket again, and the point of the whole exercise. Everything
    directly below the contact is model; the plate is reachable only by leaning
    out from under the arm and past the slab. Stopping on the slab is what the
    generator used to do."""
    model = mesh_io.drop_to_bed(bracket())
    shafts, dropped, _, ray, _ = plan(model, [down_point(-1.0, 0.0, 12.0)])

    assert shafts and not dropped
    shaft = shafts[0]
    assert not shaft.on_model, "the plate is reachable from there"
    assert shaft.land_z == pytest.approx(0.0, abs=1e-9)
    assert shaft.detours > 0, "a straight drop from there would be inside the slab"
    # It came down somewhere else entirely from where it topped out.
    assert float(np.linalg.norm(shaft.base_xy - shaft.xy)) > 1.0
    assert _pierces(shafts, ray) == 0


def test_a_shaft_with_a_clear_column_still_drops_dead_straight():
    """Routing must not cost anything when there is nothing to route around: an
    SLA shaft is a straight vertical stick and the common case has to stay
    one."""
    model = mesh_io.drop_to_bed(table(bar_z=12.0))
    shafts, *_ = plan(model, bar_points(z=12.0))
    assert shafts
    assert all(s.detours == 0 for s in shafts)
    assert all(len(s.path) == 2 for s in shafts), "no wasted rings on a straight shaft"


def test_geometry_never_enters_the_model():
    """The same guarantee, checked on the meshed result rather than the plan.

    Two bands are excluded, both deliberate rather than convenient: the tips at
    the top, which are sunk ``tip_penetration`` into the surface on purpose, and
    the plate itself, where the flared adhesion foot and the model's own
    footprint are both printed flat on glass and may overlap.
    """
    model = mesh_io.drop_to_bed(table(bar_z=12.0))
    build = build_resin(model, bar_points(z=12.0), PARAMS)
    ray = DownRay(model)
    v = build.mesh.vertices
    clear = v[(v[:, 2] > PARAMS.foot_height) & (v[:, 2] < 12.0 - PARAMS.tip_penetration * 2)]
    assert len(clear)
    assert not ray.inside(clear).any()


# --------------------------------------------------------------------------- #
# 2. the plate wins when it can
# --------------------------------------------------------------------------- #


def test_nothing_rests_on_the_model_when_the_plate_is_clear():
    model = mesh_io.drop_to_bed(table(bar_z=10.0))
    shafts, _, _, _, _ = plan(model, bar_points())
    assert shafts
    assert sum(s.on_model for s in shafts) == 0


def test_a_sealed_cavity_falls_back_to_resting_on_the_model():
    """The fallback still has to work: inside a sealed void there is no way
    down, so the shaft rests on the floor of the cavity and says so.

    Only reachable with `plate_only` off, which is not the shipped default — see
    the test below."""
    model = hollow_box()
    ceiling = model.bounds[1][2] - 4.0
    params = PARAMS.with_(plate_only=False)
    build = build_resin(model, [down_point(0.0, 0.0, ceiling)], params)

    assert build.mesh.bounds[0][2] == pytest.approx(4.0, abs=0.6), "should sit on the cavity floor"
    assert any("stand on the model" in w for w in build.warnings)


def test_plate_only_leaves_a_sealed_cavity_unsupported_rather_than_propped():
    """The shipped default. Nothing inside a sealed void can reach the plate, so
    the contact is refused and reported rather than propped off the sculpt."""
    assert PARAMS.plate_only, "plate-only is the default"
    model = hollow_box()
    ceiling = model.bounds[1][2] - 4.0
    build = build_resin(model, [down_point(0.0, 0.0, ceiling)], PARAMS)

    assert len(build.dropped) == 1
    assert build.n_points == 0
    assert any("unsupported" in w for w in build.warnings)
    assert any("not implemented yet" in w for w in build.warnings)


# --------------------------------------------------------------------------- #
# 3. it is a scaffold, not a tree
# --------------------------------------------------------------------------- #


def test_parenting_controls_how_many_shafts_stand_on_the_plate():
    model = mesh_io.drop_to_bed(mesh_io.load("samples/synthetic_mini.stl"))

    counts = []
    for strength in (0.0, 0.5, 1.0):
        params = PARAMS.with_(parenting=strength)
        points = sampling.place_points(model, params)
        shafts, *_ = plan(model, points, params)
        counts.append(len(shafts))

    assert counts[0] > counts[-1], f"shafts should fall as parenting rises: {counts}"


def test_arms_share_shafts_rather_than_one_shaft_per_contact():
    model = mesh_io.drop_to_bed(table(bar_z=14.0))
    points = bar_points(z=14.0)
    shafts, *_ = plan(model, points)
    assert sum(len(s.arms) for s in shafts) == len(points)
    assert len(shafts) < len(points), "no contact was parented onto a shared shaft"


def test_a_shaft_never_thickens_into_a_trunk():
    """The whole point of the style: shafts stay the same slim taper however
    many arms they carry. If this starts failing, the generator is drifting
    into organic-tree territory."""
    model = mesh_io.drop_to_bed(table(bar_z=14.0))
    build = build_resin(model, bar_points(z=14.0), PARAMS)

    v = build.mesh.vertices
    # Ignore the flared feet at the plate and the arms fanning out up top.
    band = v[(v[:, 2] > PARAMS.foot_height * 2) & (v[:, 2] < 14.0 - PARAMS.tip_length * 3)]
    assert len(band)
    # Every ring in that band belongs to a shaft, so no ring may be wider than
    # the shaft's own bottom diameter.
    from scipy.spatial import cKDTree

    axes = np.array([s.xy for s in plan(model, bar_points(z=14.0))[0]])
    dist, _ = cKDTree(axes).query(band[:, :2])
    assert dist.max() <= PARAMS.shaft_lower_diameter * 0.5 + 1e-6


def test_shafts_are_cross_linked_into_a_lattice():
    model = mesh_io.drop_to_bed(table(bar_z=28.0))
    build = build_resin(model, bar_points(z=28.0), PARAMS)
    assert build.n_braces > 0, "tall neighbouring shafts must brace each other"


def test_cross_links_can_be_switched_off():
    model = mesh_io.drop_to_bed(table(bar_z=28.0))
    build = build_resin(model, bar_points(z=28.0), PARAMS.with_(brace_enabled=False))
    assert build.n_braces == 0


# --------------------------------------------------------------------------- #
# 3b. and it is a tidy one: which shaft braces which, and at what height
# --------------------------------------------------------------------------- #
#
# A lattice can hold a model up and still be a mess to look at and worse to cut
# away. These pin the arrangement itself: neighbours are the shafts actually
# next to each other, the bracing is spread evenly instead of going to whoever
# asked first, and the rungs line up into storeys.


def shaft_field(n: int = 6, step: float = 3.0, jitter: float = 0.35) -> np.ndarray:
    """A patch of shaft positions — a grid, nudged so nothing is exactly
    cocircular and the triangulation has to make real choices."""
    rng = np.random.default_rng(7)
    grid = np.array([(x * step, y * step) for x in range(n) for y in range(n)], dtype=float)
    return grid + rng.uniform(-jitter, jitter, grid.shape)


def link_graph(pos: np.ndarray, params=None) -> list[tuple[int, int]]:
    """Which shaft the scaffold would tie to which, over these positions."""
    params = params or PARAMS
    chosen, _ = _choose_links(_neighbour_candidates(pos, params), len(pos))
    return chosen


def _crosses(a, b, c, d) -> bool:
    def side(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])

    return (side(c, d, a) > 0) != (side(c, d, b) > 0) and (side(a, b, c) > 0) != (
        side(a, b, d) > 0
    )


def test_no_two_cross_links_cross():
    """The graph is planar in plan view. Struts that cross each other in mid-air
    are the single loudest way a scaffold looks thrown together."""
    pos = shaft_field()
    edges = link_graph(pos)
    for (a, b), (c, d) in itertools.combinations(edges, 2):
        if len({a, b, c, d}) < 4:
            continue
        assert not _crosses(pos[a], pos[b], pos[c], pos[d]), (
            f"links {a}-{b} and {c}-{d} cross"
        )


def test_a_link_never_reaches_over_a_shaft_standing_between():
    """Nearest-first bracing happily ties two shafts together straight over the
    top of a third. Two short links through that third one brace the same pair
    better and look like they belong there."""
    pos = shaft_field()
    for i, j in link_graph(pos):
        mid = (pos[i] + pos[j]) * 0.5
        radius = float(np.linalg.norm(pos[j] - pos[i])) * 0.5
        for k in range(len(pos)):
            if k in (i, j):
                continue
            assert np.linalg.norm(pos[k] - mid) >= radius * (1.0 - 1e-6), (
                f"link {i}-{j} reaches over shaft {k}"
            )


def test_bracing_is_spread_evenly_and_leaves_nobody_out():
    """The old per-shaft greed gave the first shafts in the list their three
    best neighbours each and left a twelfth of the field with nothing. The cap
    is spent over the whole field instead, so every shaft gets a share."""
    pos = shaft_field()
    edges = link_graph(pos)
    degree = np.zeros(len(pos), dtype=int)
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1
    assert degree.min() >= 1, "no shaft may be left unbraced when it has neighbours"
    # The cap is on neighbours; the connectivity pass may spend one over it to
    # avoid cutting a corner of the field adrift.
    assert degree.max() <= _LINKS_PER_SHAFT + 1


def test_the_lattice_is_one_connected_structure():
    """A brace that ties two shafts to each other and to nothing else is not a
    scaffold. Every shaft within reach of the rest is part of the same one."""
    pos = shaft_field()
    seen = {0}
    stack = [0]
    neighbours: dict[int, list[int]] = {i: [] for i in range(len(pos))}
    for i, j in link_graph(pos):
        neighbours[i].append(j)
        neighbours[j].append(i)
    while stack:
        for nxt in neighbours[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    assert len(seen) == len(pos)


def test_which_shaft_braces_which_does_not_depend_on_shaft_order():
    """Shafts arrive in whatever order the contacts were sampled in. That is not
    a property of the model, so it must not show up in the scaffold."""
    pos = shaft_field()
    order = np.random.default_rng(3).permutation(len(pos))
    shuffled = pos[order]

    def as_positions(edges, table):
        return {frozenset((tuple(np.round(table[i], 9)), tuple(np.round(table[j], 9)))) for i, j in edges}

    assert as_positions(link_graph(pos), pos) == as_positions(link_graph(shuffled), shuffled)


def stubs_and_towers():
    """Contacts alternating between just off the plate and far up, side by side,
    in clear air beside a small block. Every link here ties a stub to a tower,
    which is the case that tells the two shafts' tops apart."""
    return [
        down_point(float(x), 0.0, 16.0 if k % 2 else 30.0)
        for k, x in enumerate(np.arange(12.0, 24.1, 3.0))
    ]


def link_tops(model, points, params):
    """For every rung built, how far its upper end finished below the top of the
    shorter shaft it ties. That is the gap under the model the links leave."""
    shafts = plan(model, points, params)[0]
    tan_a = math.tan(math.radians(link_angle(params)))
    ray = DownRay(model)
    field = AvoidanceField(model, params, top_z=max(float(q.position[2]) for q in points))
    edges = link_graph(np.array([sh.xy for sh in shafts]), params)
    storeys = _link_storeys(shafts, edges, tan_a, params)
    gaps = []
    for i, j in edges:
        ceiling = min(shafts[i].top_z, shafts[j].top_z)
        for rung in _rungs(field, ray, shafts, i, j, tan_a, storeys, params):
            gaps.append(ceiling - float(rung.vertices[:, 2].max()))
    return gaps


@pytest.mark.parametrize(
    "scene",
    [
        (lambda: (mesh_io.drop_to_bed(table(bar_z=28.0)), ledge_grid(bar_z=28.0))),
        (lambda: (mesh_io.drop_to_bed(solid_box()), stubs_and_towers())),
    ],
    ids=["even field", "stubs and towers"],
)
def test_headroom_keeps_the_links_clear_of_the_arms(scene):
    """A shaft's top is where its arms leave for the model, so a link that runs
    all the way up lands in the middle of the arm fan, which is exactly where a
    blade has to reach. `brace_headroom` is how much of that space to give back."""
    model, points = scene()
    for headroom in (0.0, 1.0, 3.0):
        gaps = link_tops(model, points, PARAMS.with_(brace_headroom=headroom))
        assert gaps, f"headroom {headroom} left no links at all"
        assert min(gaps) >= headroom - 1e-6, (
            f"a link came within {min(gaps):.2f} mm of a shaft top, asked for {headroom}"
        )


def test_the_link_angle_can_be_set_and_is_held_inside_the_printable_band():
    """A link overhangs by `90 - angle` down its sides and by `angle` at its
    ends, so only the band between the two prints at all. A value inside it is
    taken as asked; one outside is pulled back to the edge rather than obeyed —
    a support that needs supports is the one thing this generator may not make."""
    limit = PARAMS.printable_overhang_deg
    assert link_angle(PARAMS.with_(brace_angle_deg=None)) == pytest.approx(90 - limit + 2)
    assert link_angle(PARAMS.with_(brace_angle_deg=46.0)) == pytest.approx(46.0)
    assert link_angle(PARAMS.with_(brace_angle_deg=5.0)) == pytest.approx(90 - limit)
    assert link_angle(PARAMS.with_(brace_angle_deg=85.0)) == pytest.approx(limit)


def test_a_steeper_link_angle_costs_span():
    """Rise is `span · tan angle`, so a steeper link needs more height for the
    same reach. On short shafts that is the difference between a braced pair and
    a bare one, which is why the default takes the shallowest angle going."""
    model = mesh_io.drop_to_bed(table(bar_z=28.0))
    points = ledge_grid(bar_z=28.0)
    shallow = build_resin(model, points, PARAMS.with_(brace_angle_deg=90 - PARAMS.printable_overhang_deg))
    steep = build_resin(model, points, PARAMS.with_(brace_angle_deg=PARAMS.printable_overhang_deg))
    assert shallow.n_braces >= steep.n_braces


def rung_heights(model, points, params):
    """Every rung the field builds, as (pair, height of its lower end)."""
    shafts = plan(model, points, params)[0]
    tan_a = math.tan(math.radians(link_angle(params)))
    ray = DownRay(model)
    field = AvoidanceField(model, params, top_z=max(float(q.position[2]) for q in points))
    edges = link_graph(np.array([sh.xy for sh in shafts]), params)
    storeys = _link_storeys(shafts, edges, tan_a, params)
    out = []
    for i, j in edges:
        for rung in _rungs(field, ray, shafts, i, j, tan_a, storeys, params):
            out.append(((i, j), float(rung.vertices[:, 2].min())))
    return out


def tall_pair_points():
    """A row of contacts high enough that the shafts under them have room for a
    ladder rather than a single rung."""
    return [down_point(float(x), 0.0, 34.0) for x in np.arange(3.0, 12.1, 3.0)]


def test_two_shafts_are_tied_at_every_interval_up_their_height():
    """A pair of 30 mm pillars braced once near the plate is a pair of stilts.
    Where there is height for it they get a ladder, and `brace_interval` is the
    rung spacing."""
    model = mesh_io.drop_to_bed(solid_box())
    points = tall_pair_points()
    counts = {}
    for interval in (4.0, 8.0, 16.0):
        per_pair = collections.Counter(
            pair for pair, _ in rung_heights(model, points, PARAMS.with_(brace_interval=interval))
        )
        assert per_pair, f"interval {interval} built nothing"
        counts[interval] = max(per_pair.values())
    assert counts[4.0] > counts[8.0] > counts[16.0], counts
    assert counts[4.0] >= 3, f"a 30 mm pair at 4 mm spacing should be a ladder, got {counts}"


def test_rungs_up_one_pair_keep_their_distance():
    """Whatever the spacing asks for, two links closer together than their own
    combined thickness are one lump with a hole in it."""
    model = mesh_io.drop_to_bed(solid_box())
    points = tall_pair_points()
    for interval in (0.0, 4.0, 8.0):
        params = PARAMS.with_(brace_interval=interval)
        floor = max(interval, params.brace_diameter * 2.0)
        by_pair: dict[tuple[int, int], list[float]] = {}
        for pair, z in rung_heights(model, points, params):
            by_pair.setdefault(pair, []).append(z)
        for pair, zs in by_pair.items():
            zs.sort()
            gaps = np.diff(zs)
            assert all(g >= floor - 1e-6 for g in gaps), (
                f"pair {pair} at interval {interval} has rungs {np.round(gaps, 2)} apart"
            )


def towers_and_stubs():
    """Two tall contacts with a thicket of short ones between them, in clear air
    beside a block. Each tower's actual neighbours are all stubs, so the only
    thing that can brace its upper half is the other tower — which is not a
    neighbour, and is exactly what the tidy graph would never pick."""
    pts = [down_point(12.0, 0.0, 34.0), down_point(21.0, 0.0, 34.0)]
    pts += [down_point(x, 0.0, 14.0) for x in (15.0, 18.0)]
    pts += [down_point(x, 3.0, 14.0) for x in (13.5, 16.5, 19.5)]
    return pts


def highest_link(model, points, params):
    """For each shaft, the height of the topmost link touching it."""
    shafts = plan(model, points, params)[0]
    tan_a = math.tan(math.radians(link_angle(params)))
    ray = DownRay(model)
    field = AvoidanceField(model, params, top_z=max(float(q.position[2]) for q in points))
    links, _ = _link_shafts(field, ray, shafts, params)

    reach = [-np.inf] * len(shafts)
    for m in links:
        v = m.vertices
        top, bottom = float(v[:, 2].max()), float(v[:, 2].min())
        for point, z in ((v[v[:, 2] > top - 1e-6][0], top), (v[v[:, 2] < bottom + 1e-6][0], bottom)):
            near = int(np.argmin([np.linalg.norm(sh.xy_at(z) - point[:2]) for sh in shafts]))
            reach[near] = max(reach[near], top)
    return shafts, reach


def test_a_tall_shaft_is_braced_near_its_top_not_only_at_its_feet():
    """A link can only finish as high as the *shorter* of the two shafts it ties,
    so a tall shaft standing in a thicket of stubs gets braced to the top of the
    stubs and is free above that — the half that flexes, and the half nothing
    else is holding. Having links is not the same as being held, so the test is
    on where the topmost one is, not on how many there are.
    """
    model = mesh_io.drop_to_bed(solid_box())
    shafts, reach = highest_link(model, towers_and_stubs(), PARAMS)

    stubs = max(sh.top_z for sh in shafts if sh.height < 20.0)
    towers = [i for i, sh in enumerate(shafts) if sh.height > 20.0]
    assert len(towers) == 2, "the scene should stand two tall shafts"
    for i in towers:
        assert reach[i] > stubs + PARAMS.brace_interval, (
            f"tower {i} is {shafts[i].top_z:.1f} mm tall and its highest link is at "
            f"{reach[i]:.1f}, barely above the {stubs:.1f} mm stubs around it"
        )


def test_a_ladder_does_not_lose_a_rung_to_a_rounding_tie():
    """Storeys an exact `brace_interval` apart come out a few ulp *short* of it
    once they are tens of millimetres up. A bare `>=` against the spacing then
    drops rungs out of the middle and the top of a ladder, which from the outside
    looks precisely like the cross-links giving up part way up a pillar — and
    reads as a decision the generator made rather than as arithmetic.
    """
    spacing = _rung_spacing(PARAMS)
    heights = [3.3248727362960313]  # a real storey datum from the Templar
    while len(heights) < 10:
        heights.append(heights[-1] + spacing)
    storeys = np.array(heights)

    # The bug, stated: exact arithmetic says every gap is `spacing`.
    assert not all(gap >= spacing for gap in np.diff(storeys)), "pick a datum that drifts"

    kept = _rung_heights(storeys, storeys[0] - 1.0, storeys[-1] + spacing * 0.1, PARAMS)
    assert len(kept) == len(storeys), f"lost {len(storeys) - len(kept)} of {len(storeys)} rungs"


def test_the_ladder_shares_its_heights_with_the_rest_of_the_field():
    """The extra rungs are a grid for the whole field, not a per-pair ladder —
    otherwise stacking links would undo the arrangement that made them tidy."""
    model = mesh_io.drop_to_bed(table(bar_z=28.0))
    points = ledge_grid(bar_z=28.0)
    shafts = plan(model, points, PARAMS)[0]
    tan_a = math.tan(math.radians(link_angle(PARAMS)))
    edges = link_graph(np.array([sh.xy for sh in shafts]), PARAMS)
    storeys = _link_storeys(shafts, edges, tan_a, PARAMS)
    assert len(storeys) >= 1
    gaps = np.diff(storeys)
    assert all(g > PARAMS.brace_interval * 0.5 - 1e-9 for g in gaps), (
        f"storeys bunched up: {np.round(storeys, 2)}"
    )


def test_max_span_bounds_how_far_a_link_reaches():
    """`brace_max_span` is the reach limit, and it is a hard one — no pair
    further apart than this is even a candidate."""
    pos = shaft_field()
    for span in (4.0, 8.0):
        for i, j in link_graph(pos, PARAMS.with_(brace_max_span=span)):
            assert float(np.linalg.norm(pos[j] - pos[i])) <= span + 1e-9


def test_the_whole_field_lays_its_links_on_a_few_shared_storeys():
    """Every pair used to start its own ladder from its own base, so the links
    ended up at as many heights as there were links. They share a handful of
    heights now, and — the point of choosing those heights from the pairs
    rather than off a fixed grid — every pair with room for a link can reach
    one of them."""
    model = mesh_io.drop_to_bed(table(bar_z=28.0))
    shafts = plan(model, ledge_grid(bar_z=28.0))[0]
    tan_a = math.tan(math.radians(link_angle(PARAMS)))
    edges = link_graph(np.array([s.xy for s in shafts]))
    storeys = _link_storeys(shafts, edges, tan_a, PARAMS)

    assert 0 < len(storeys) <= 6, "a field this size should not need many storeys"
    for i, j in edges:
        band = _link_band(shafts[i], shafts[j], tan_a, PARAMS)
        if band is None:
            continue  # too short to hold a printable diagonal at all
        assert any(band[0] <= z <= band[1] for z in storeys), (
            f"shafts {i}-{j} can hold a link but no storey is within reach"
        )


# --------------------------------------------------------------------------- #
# 4. contract
# --------------------------------------------------------------------------- #


def test_output_is_printable_and_deterministic():
    model = mesh_io.drop_to_bed(table(bar_z=12.0))
    points = bar_points(z=12.0)
    a = build_resin(model, points, PARAMS)
    b = build_resin(model, points, PARAMS)
    assert len(a.mesh.faces) == len(b.mesh.faces)

    rep = printability_report(a.mesh, PARAMS, model)
    assert rep["total"] > 0
    assert rep["violations"] == 0, f"{rep['violations']}/{rep['total']} faces overhang"


def test_no_points_means_no_geometry():
    model = mesh_io.drop_to_bed(table())
    assert len(build_resin(model, [], PARAMS).mesh.faces) == 0


def test_geometry_stays_on_the_right_side_of_the_plate():
    model = mesh_io.drop_to_bed(table(bar_z=12.0))
    build = build_resin(model, bar_points(z=12.0), PARAMS)
    assert build.mesh.bounds[0][2] >= -1e-6
    assert build.mesh.bounds[1][2] <= model.bounds[1][2] + PARAMS.tip_penetration + 1e-6


def test_build_supports_builds_the_scaffold():
    """There is one structure; the entry point must produce exactly it."""
    from rsupport.supports import build_supports

    model = mesh_io.drop_to_bed(table(bar_z=12.0))
    points = bar_points(z=12.0)
    assert (
        len(build_supports(model, points, PARAMS).mesh.faces)
        == len(build_resin(model, points, PARAMS).mesh.faces)
    )


# --------------------------------------------------------------------------- #
# 5. the model in the air
# --------------------------------------------------------------------------- #


def test_the_bases_of_a_lifted_model_overlap_into_a_raft():
    """Why the base is as wide as it is.

    With the model held in the air a shaft lands roughly every
    ``support_spacing``, so a base wider than that spacing means neighbouring
    footprints *merge*. What ends up on the glass is one sheet rather than a
    field of separate discs, and a scaffold of 1.2 mm shafts stays put.
    """
    lift = 5.0
    params = PARAMS.with_(lift_height=lift)
    assert params.foot_diameter > params.support_spacing, (
        "a base narrower than the support spacing could not overlap anything"
    )
    model = mesh_io.drop_to_bed(trimesh.creation.box(extents=[14, 14, 10]), lift=lift)

    points = sampling.place_points(model, params)
    shafts, _, _, _, _ = plan(model, points, params)
    feet = np.array([s.xy for s in shafts if not s.on_model])
    assert len(feet) > 4, "a lifted slab should stand on a forest of shafts"

    d = np.linalg.norm(feet[:, None] - feet[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    touching = d.min(axis=1) < params.foot_diameter
    assert touching.mean() > 0.9, (
        f"only {touching.mean():.0%} of bases overlap a neighbour; that is a field "
        "of discs, not a raft"
    )


def test_a_lifted_model_is_held_off_the_plate_by_supports_alone():
    """Nothing of the model may touch the plate, and every support must reach
    it — the whole point of lifting is that the scaffold carries the model."""
    lift = 5.0
    params = PARAMS.with_(lift_height=lift)
    model = mesh_io.drop_to_bed(mesh_io.load("samples/synthetic_mini.stl"), lift=lift)
    assert model.bounds[0][2] == pytest.approx(lift, abs=1e-6)

    build = build_resin(model, sampling.place_points(model, params), params)
    assert build.mesh.bounds[0][2] == pytest.approx(0.0, abs=1e-6)
    assert build.mesh.bounds[1][2] <= model.bounds[1][2] + params.tip_penetration + 1e-6
    rep = printability_report(build.mesh, params, model)
    assert rep["violations"] == 0

