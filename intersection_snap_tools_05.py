bl_info = {
    "name": "Curve / Edge Intersection Snap Tools",
    "author": "OpenAI",
    "version": (1, 5, 0),
    "blender": (3, 6, 0),
    "location": "3D View Header / Object Context Menu",
    "description": "Split curves and edge-only meshes at intersections, dissolve collinear points, and refit circles/arcs",
    "category": "Object",
}

import bpy
import bmesh
from mathutils import Vector, geometry
from math import atan2, cos, sin, pi, sqrt

EPS = 1e-6
POINT_EPS = 1e-5
COLLINEAR_EPS = 1e-4
SAMPLE_STEPS = 32


# ============================================================
# Utility
# ============================================================

def report_msg(op, text, level={'INFO'}):
    op.report(level, text)


def mesh_is_edges_only(obj):
    return (
        obj is not None and
        obj.type == 'MESH' and
        len(obj.data.polygons) == 0 and
        len(obj.data.edges) > 0
    )


def is_supported_object(obj):
    return obj is not None and (obj.type == 'CURVE' or mesh_is_edges_only(obj))


def get_selected_supported_objects(context):
    return [obj for obj in context.selected_objects if is_supported_object(obj)]


def get_selected_curve_objects(context):
    return [obj for obj in context.selected_objects if obj.type == 'CURVE']


def point_key(v, prec=6):
    return (round(v.x, prec), round(v.y, prec), round(v.z, prec))


def clamp01(x):
    return max(0.0, min(1.0, x))


def lerp(a, b, t):
    return a.lerp(b, t)


def point_on_segment_factor(p, a, b):
    ab = b - a
    l2 = ab.length_squared
    if l2 <= EPS:
        return None
    t = (p - a).dot(ab) / l2
    proj = a.lerp(b, t)
    if (proj - p).length > POINT_EPS:
        return None
    if t < -POINT_EPS or t > 1.0 + POINT_EPS:
        return None
    return clamp01(t)


def safe_normal(v):
    if v.length <= EPS:
        return None
    out = v.copy()
    out.normalize()
    return out


# ============================================================
# Cubic Bézier math
# ============================================================

def cubic_bezier_eval(p0, p1, p2, p3, t):
    a = lerp(p0, p1, t)
    b = lerp(p1, p2, t)
    c = lerp(p2, p3, t)
    d = lerp(a, b, t)
    e = lerp(b, c, t)
    f = lerp(d, e, t)
    return f


def split_cubic_bezier(p0, p1, p2, p3, t):
    a = lerp(p0, p1, t)
    b = lerp(p1, p2, t)
    c = lerp(p2, p3, t)
    d = lerp(a, b, t)
    e = lerp(b, c, t)
    f = lerp(d, e, t)
    left = (p0, a, d, f)
    right = (f, e, c, p3)
    return left, right


def split_cubic_bezier_multi(p0, p1, p2, p3, t_values):
    clean = []
    for t in sorted(t_values):
        if POINT_EPS < t < 1.0 - POINT_EPS:
            if not clean or abs(clean[-1] - t) > 1e-6:
                clean.append(t)

    if not clean:
        return [(p0, p1, p2, p3)]

    result = []
    cp0, cp1, cp2, cp3 = p0, p1, p2, p3
    prev_t = 0.0

    for t in clean:
        local_t = (t - prev_t) / max(EPS, (1.0 - prev_t))
        left, right = split_cubic_bezier(cp0, cp1, cp2, cp3, local_t)
        result.append(left)
        cp0, cp1, cp2, cp3 = right
        prev_t = t

    result.append((cp0, cp1, cp2, cp3))
    return result


# ============================================================
# Segment models
# ============================================================

class LinearSegment:
    __slots__ = ("obj", "kind", "spline_index", "a_idx", "b_idx", "a_world", "b_world")

    def __init__(self, obj, kind, spline_index, a_idx, b_idx, a_world, b_world):
        self.obj = obj
        self.kind = kind
        self.spline_index = spline_index
        self.a_idx = a_idx
        self.b_idx = b_idx
        self.a_world = a_world
        self.b_world = b_world


class BezierSegment:
    __slots__ = ("obj", "spline_index", "a_idx", "b_idx", "p0", "p1", "p2", "p3")

    def __init__(self, obj, spline_index, a_idx, b_idx, p0, p1, p2, p3):
        self.obj = obj
        self.spline_index = spline_index
        self.a_idx = a_idx
        self.b_idx = b_idx
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3


# ============================================================
# Segment extraction
# ============================================================

def collect_curve_segments(obj):
    linear_segments = []
    bezier_segments = []
    mw = obj.matrix_world

    for spline_index, spline in enumerate(obj.data.splines):
        if spline.type == 'BEZIER':
            pts = spline.bezier_points
            n = len(pts)
            if n < 2:
                continue
            limit = n if spline.use_cyclic_u else n - 1
            for i in range(limit):
                j = (i + 1) % n
                a = pts[i]
                b = pts[j]
                bezier_segments.append(
                    BezierSegment(
                        obj=obj,
                        spline_index=spline_index,
                        a_idx=i,
                        b_idx=j,
                        p0=mw @ a.co,
                        p1=mw @ a.handle_right,
                        p2=mw @ b.handle_left,
                        p3=mw @ b.co,
                    )
                )
        else:
            pts = spline.points
            n = len(pts)
            if n < 2:
                continue
            limit = n if spline.use_cyclic_u else n - 1
            for i in range(limit):
                j = (i + 1) % n
                linear_segments.append(
                    LinearSegment(
                        obj=obj,
                        kind='CURVE',
                        spline_index=spline_index,
                        a_idx=i,
                        b_idx=j,
                        a_world=mw @ pts[i].co.xyz,
                        b_world=mw @ pts[j].co.xyz,
                    )
                )

    return linear_segments, bezier_segments


def collect_mesh_segments(obj):
    linear_segments = []
    me = obj.data
    mw = obj.matrix_world

    for e in me.edges:
        v0 = me.vertices[e.vertices[0]].co
        v1 = me.vertices[e.vertices[1]].co
        linear_segments.append(
            LinearSegment(
                obj=obj,
                kind='MESH',
                spline_index=-1,
                a_idx=e.vertices[0],
                b_idx=e.vertices[1],
                a_world=mw @ v0,
                b_world=mw @ v1,
            )
        )
    return linear_segments


def collect_segments(objects):
    linear_segments = []
    bezier_segments = []

    for obj in objects:
        if obj.type == 'CURVE':
            ls, bs = collect_curve_segments(obj)
            linear_segments.extend(ls)
            bezier_segments.extend(bs)
        elif mesh_is_edges_only(obj):
            linear_segments.extend(collect_mesh_segments(obj))

    return linear_segments, bezier_segments


# ============================================================
# Intersection helpers
# ============================================================

def linear_linear_intersection(seg1, seg2):
    if seg1.obj == seg2.obj and seg1.kind == 'MESH' and seg2.kind == 'MESH':
        if len({seg1.a_idx, seg1.b_idx, seg2.a_idx, seg2.b_idx}) < 4:
            return None

    result = geometry.intersect_line_line(seg1.a_world, seg1.b_world, seg2.a_world, seg2.b_world)
    if not result:
        return None

    p1, p2 = result
    if (p1 - p2).length > POINT_EPS:
        return None

    hit = (p1 + p2) * 0.5
    t1 = point_on_segment_factor(hit, seg1.a_world, seg1.b_world)
    t2 = point_on_segment_factor(hit, seg2.a_world, seg2.b_world)

    if t1 is None or t2 is None:
        return None

    return hit, t1, t2


def bezier_to_polyline(seg, steps=SAMPLE_STEPS):
    pts = []
    ts = []
    for i in range(steps + 1):
        t = i / steps
        pts.append(cubic_bezier_eval(seg.p0, seg.p1, seg.p2, seg.p3, t))
        ts.append(t)
    return pts, ts


def polyline_intersections(poly_a, ts_a, poly_b, ts_b):
    hits = []

    for i in range(len(poly_a) - 1):
        a0, a1 = poly_a[i], poly_a[i + 1]
        for j in range(len(poly_b) - 1):
            b0, b1 = poly_b[j], poly_b[j + 1]

            result = geometry.intersect_line_line(a0, a1, b0, b1)
            if not result:
                continue

            p1, p2 = result
            if (p1 - p2).length > POINT_EPS:
                continue

            hit = (p1 + p2) * 0.5
            ta_local = point_on_segment_factor(hit, a0, a1)
            tb_local = point_on_segment_factor(hit, b0, b1)
            if ta_local is None or tb_local is None:
                continue

            ta = ts_a[i] + (ts_a[i + 1] - ts_a[i]) * ta_local
            tb = ts_b[j] + (ts_b[j + 1] - ts_b[j]) * tb_local
            hits.append((hit, ta, tb))

    return hits


def bezier_bezier_intersections(seg1, seg2):
    same_segment = (
        seg1.obj == seg2.obj and
        seg1.spline_index == seg2.spline_index and
        seg1.a_idx == seg2.a_idx and
        seg1.b_idx == seg2.b_idx
    )
    if same_segment:
        return []

    poly1, ts1 = bezier_to_polyline(seg1)
    poly2, ts2 = bezier_to_polyline(seg2)
    raw_hits = polyline_intersections(poly1, ts1, poly2, ts2)

    uniq = {}
    for hit, t1, t2 in raw_hits:
        if not (POINT_EPS < t1 < 1.0 - POINT_EPS):
            continue
        if not (POINT_EPS < t2 < 1.0 - POINT_EPS):
            continue
        uniq[(point_key(hit), round(t1, 4), round(t2, 4))] = (hit, t1, t2)

    return list(uniq.values())


def linear_bezier_intersections(linear_seg, bezier_seg):
    poly, ts = bezier_to_polyline(bezier_seg)
    hits = []

    for i in range(len(poly) - 1):
        b0, b1 = poly[i], poly[i + 1]
        result = geometry.intersect_line_line(linear_seg.a_world, linear_seg.b_world, b0, b1)
        if not result:
            continue

        p1, p2 = result
        if (p1 - p2).length > POINT_EPS:
            continue

        hit = (p1 + p2) * 0.5
        tl = point_on_segment_factor(hit, linear_seg.a_world, linear_seg.b_world)
        tb_local = point_on_segment_factor(hit, b0, b1)
        if tl is None or tb_local is None:
            continue

        tb = ts[i] + (ts[i + 1] - ts[i]) * tb_local
        if not (POINT_EPS < tb < 1.0 - POINT_EPS):
            continue

        hits.append((hit, tl, tb))

    uniq = {}
    for hit, tl, tb in hits:
        uniq[(point_key(hit), round(tl, 4), round(tb, 4))] = (hit, tl, tb)

    return list(uniq.values())


# ============================================================
# Build requests
# ============================================================

def build_intersection_requests(objects):
    linear_segments, bezier_segments = collect_segments(objects)

    curve_linear_requests = {}
    curve_bezier_requests = {}
    mesh_requests = {}
    seen = set()

    for i in range(len(linear_segments)):
        s1 = linear_segments[i]
        for j in range(i + 1, len(linear_segments)):
            s2 = linear_segments[j]
            hit_data = linear_linear_intersection(s1, s2)
            if hit_data is None:
                continue

            hit, t1, t2 = hit_data
            key = (
                s1.obj.name, s1.spline_index, s1.a_idx, s1.b_idx,
                s2.obj.name, s2.spline_index, s2.a_idx, s2.b_idx,
                point_key(hit),
            )
            if key in seen:
                continue
            seen.add(key)

            if s1.kind == 'CURVE':
                curve_linear_requests.setdefault(s1.obj.name, []).append(
                    (s1.spline_index, s1.a_idx, s1.b_idx, t1, hit.copy())
                )
            else:
                mesh_requests.setdefault(s1.obj.name, []).append(
                    (tuple(sorted((s1.a_idx, s1.b_idx))), hit.copy())
                )

            if s2.kind == 'CURVE':
                curve_linear_requests.setdefault(s2.obj.name, []).append(
                    (s2.spline_index, s2.a_idx, s2.b_idx, t2, hit.copy())
                )
            else:
                mesh_requests.setdefault(s2.obj.name, []).append(
                    (tuple(sorted((s2.a_idx, s2.b_idx))), hit.copy())
                )

    for i in range(len(bezier_segments)):
        s1 = bezier_segments[i]
        for j in range(i + 1, len(bezier_segments)):
            s2 = bezier_segments[j]
            hits = bezier_bezier_intersections(s1, s2)
            for hit, t1, t2 in hits:
                key = (
                    s1.obj.name, s1.spline_index, s1.a_idx, s1.b_idx,
                    s2.obj.name, s2.spline_index, s2.a_idx, s2.b_idx,
                    point_key(hit),
                )
                if key in seen:
                    continue
                seen.add(key)

                curve_bezier_requests.setdefault(s1.obj.name, []).append(
                    (s1.spline_index, s1.a_idx, s1.b_idx, t1, hit.copy())
                )
                curve_bezier_requests.setdefault(s2.obj.name, []).append(
                    (s2.spline_index, s2.a_idx, s2.b_idx, t2, hit.copy())
                )

    for ls in linear_segments:
        for bs in bezier_segments:
            hits = linear_bezier_intersections(ls, bs)
            for hit, tl, tb in hits:
                key = (
                    ls.obj.name, ls.spline_index, ls.a_idx, ls.b_idx,
                    bs.obj.name, bs.spline_index, bs.a_idx, bs.b_idx,
                    point_key(hit),
                )
                if key in seen:
                    continue
                seen.add(key)

                if ls.kind == 'CURVE':
                    curve_linear_requests.setdefault(ls.obj.name, []).append(
                        (ls.spline_index, ls.a_idx, ls.b_idx, tl, hit.copy())
                    )
                else:
                    mesh_requests.setdefault(ls.obj.name, []).append(
                        (tuple(sorted((ls.a_idx, ls.b_idx))), hit.copy())
                    )

                curve_bezier_requests.setdefault(bs.obj.name, []).append(
                    (bs.spline_index, bs.a_idx, bs.b_idx, tb, hit.copy())
                )

    return curve_linear_requests, curve_bezier_requests, mesh_requests


# ============================================================
# Curve snapshot / rebuild
# ============================================================

def snapshot_curve_data(curve_data):
    out = []

    for spline in curve_data.splines:
        if spline.type == 'BEZIER':
            pts = []
            for bp in spline.bezier_points:
                pts.append({
                    "co": bp.co.copy(),
                    "handle_left": bp.handle_left.copy(),
                    "handle_right": bp.handle_right.copy(),
                    "handle_left_type": bp.handle_left_type,
                    "handle_right_type": bp.handle_right_type,
                    "radius": bp.radius,
                    "tilt": bp.tilt,
                })
        else:
            pts = []
            for p in spline.points:
                pts.append({
                    "co": p.co.copy(),
                    "radius": p.radius,
                    "tilt": p.tilt,
                    "weight": p.weight,
                })

        out.append({
            "type": spline.type,
            "use_cyclic_u": spline.use_cyclic_u,
            "resolution_u": spline.resolution_u,
            "order_u": spline.order_u,
            "use_endpoint_u": spline.use_endpoint_u,
            "use_smooth": spline.use_smooth,
            "points": pts,
        })

    return out


def rebuild_poly_or_nurbs_spline(obj, spline, src, spline_index, linear_requests):
    mw_inv = obj.matrix_world.inverted()
    src_pts = src["points"]
    n = len(src_pts)
    new_pts = []
    added = 0

    per_segment = {}
    for req in linear_requests:
        si, a_idx, b_idx, t, hit_world = req
        if si != spline_index:
            continue
        if POINT_EPS < t < 1.0 - POINT_EPS:
            per_segment.setdefault((a_idx, b_idx), []).append((t, hit_world))

    for i, pt in enumerate(src_pts):
        new_pts.append(pt)

        if n < 2:
            continue
        if not src["use_cyclic_u"] and i == n - 1:
            continue

        j = (i + 1) % n
        hits = per_segment.get((i, j), [])
        if not hits:
            continue

        uniq = {}
        for t, hit_world in hits:
            uniq[point_key(hit_world)] = (t, hit_world)
        ordered = sorted(uniq.values(), key=lambda x: x[0])

        for _, hit_world in ordered:
            local = mw_inv @ hit_world
            new_pts.append({
                "co": Vector((local.x, local.y, local.z, 1.0)),
                "radius": pt["radius"],
                "tilt": pt["tilt"],
                "weight": pt.get("weight", 1.0),
            })
            added += 1

    spline.points.add(len(new_pts) - 1)
    for idx, p in enumerate(new_pts):
        cp = spline.points[idx]
        cp.co = p["co"]
        cp.radius = p["radius"]
        cp.tilt = p["tilt"]
        cp.weight = p.get("weight", 1.0)

    return added


def rebuild_bezier_spline(src, spline_index, bezier_requests):
    src_pts = src["points"]
    count = len(src_pts)

    if count < 2:
        return src_pts[:], 0

    per_segment = {}
    for req in bezier_requests:
        si, a_idx, b_idx, t, _hit_world = req
        if si != spline_index:
            continue
        if POINT_EPS < t < 1.0 - POINT_EPS:
            per_segment.setdefault((a_idx, b_idx), []).append(t)

    limit = count if src["use_cyclic_u"] else count - 1
    segment_groups = []
    added = 0

    for i in range(limit):
        j = (i + 1) % count

        p0 = src_pts[i]["co"]
        p1 = src_pts[i]["handle_right"]
        p2 = src_pts[j]["handle_left"]
        p3 = src_pts[j]["co"]

        t_values = sorted(set(round(t, 6) for t in per_segment.get((i, j), [])))
        t_values = [float(t) for t in t_values]
        segs = split_cubic_bezier_multi(p0, p1, p2, p3, t_values)
        segment_groups.append((i, j, segs))
        added += max(0, len(segs) - 1)

    if not segment_groups:
        return src_pts[:], 0

    flat_segments = []
    flat_owner_j = []
    flat_owner_i = []
    for base_i, base_j, segs in segment_groups:
        for seg in segs:
            flat_segments.append(seg)
            flat_owner_i.append(base_i)
            flat_owner_j.append(base_j)

    new_points = []

    if not src["use_cyclic_u"]:
        first_seg = flat_segments[0]
        first_i = flat_owner_i[0]

        new_points.append({
            "co": first_seg[0],
            "handle_left": src_pts[first_i]["handle_left"],
            "handle_right": first_seg[1],
            "handle_left_type": src_pts[first_i]["handle_left_type"],
            "handle_right_type": 'FREE',
            "radius": src_pts[first_i]["radius"],
            "tilt": src_pts[first_i]["tilt"],
        })

        for idx_seg, seg in enumerate(flat_segments):
            p0, p1, p2, p3 = seg
            owner_j = flat_owner_j[idx_seg]
            is_last = idx_seg == len(flat_segments) - 1

            if is_last:
                hr = src_pts[owner_j]["handle_right"]
                hrt = src_pts[owner_j]["handle_right_type"]
            else:
                hr = flat_segments[idx_seg + 1][1]
                hrt = 'FREE'

            new_points.append({
                "co": p3,
                "handle_left": p2,
                "handle_right": hr,
                "handle_left_type": 'FREE',
                "handle_right_type": hrt,
                "radius": src_pts[owner_j]["radius"],
                "tilt": src_pts[owner_j]["tilt"],
            })

    else:
        total_segments = len(flat_segments)
        first_seg = flat_segments[0]
        last_seg = flat_segments[-1]

        new_points.append({
            "co": first_seg[0],
            "handle_left": last_seg[2],
            "handle_right": first_seg[1],
            "handle_left_type": 'FREE',
            "handle_right_type": 'FREE',
            "radius": src_pts[0]["radius"],
            "tilt": src_pts[0]["tilt"],
        })

        for idx_seg, seg in enumerate(flat_segments):
            p0, p1, p2, p3 = seg
            next_seg = flat_segments[(idx_seg + 1) % total_segments]
            owner_j = flat_owner_j[idx_seg]

            new_points.append({
                "co": p3,
                "handle_left": p2,
                "handle_right": next_seg[1],
                "handle_left_type": 'FREE',
                "handle_right_type": 'FREE',
                "radius": src_pts[owner_j]["radius"],
                "tilt": src_pts[owner_j]["tilt"],
            })

        if len(new_points) > 1 and (new_points[0]["co"] - new_points[-1]["co"]).length <= POINT_EPS:
            new_points.pop()

    return new_points, added


def rebuild_curve_with_insertions(obj, linear_requests, bezier_requests):
    curve_data = obj.data
    old_splines = snapshot_curve_data(curve_data)

    if not linear_requests and not bezier_requests:
        return 0

    curve_data.splines.clear()
    total_added = 0

    for spline_index, src in enumerate(old_splines):
        spline = curve_data.splines.new(src["type"])
        spline.use_cyclic_u = src["use_cyclic_u"]
        spline.resolution_u = src["resolution_u"]
        spline.order_u = src["order_u"]
        spline.use_endpoint_u = src["use_endpoint_u"]
        spline.use_smooth = src["use_smooth"]

        if src["type"] == 'BEZIER':
            new_pts, added = rebuild_bezier_spline(src, spline_index, bezier_requests)
            total_added += added

            spline.bezier_points.add(len(new_pts) - 1)
            for idx, p in enumerate(new_pts):
                bp = spline.bezier_points[idx]
                bp.co = p["co"]
                bp.handle_left = p["handle_left"]
                bp.handle_right = p["handle_right"]
                bp.handle_left_type = p["handle_left_type"]
                bp.handle_right_type = p["handle_right_type"]
                bp.radius = p["radius"]
                bp.tilt = p["tilt"]
        else:
            total_added += rebuild_poly_or_nurbs_spline(obj, spline, src, spline_index, linear_requests)

    curve_data.update_tag()
    return total_added


# ============================================================
# Mesh split
# ============================================================

def split_mesh_edges_at_hits(obj, requests):
    if not requests:
        return 0

    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    mw_inv = obj.matrix_world.inverted()
    edge_hit_map = {}

    for edge_key, hit_world in requests:
        hit_local = mw_inv @ hit_world
        edge_hit_map.setdefault(edge_key, []).append(hit_local)

    created = 0

    for edge in list(bm.edges):
        key = tuple(sorted((edge.verts[0].index, edge.verts[1].index)))
        hits = edge_hit_map.get(key)
        if not hits:
            continue

        a = edge.verts[0].co.copy()
        b = edge.verts[1].co.copy()
        ab = b - a
        l2 = ab.length_squared
        if l2 <= EPS:
            continue

        uniq = {}
        for p in hits:
            t = (p - a).dot(ab) / l2
            if t <= POINT_EPS or t >= 1.0 - POINT_EPS:
                continue
            uniq[point_key(p)] = (t, p)

        ordered = sorted(uniq.values(), key=lambda x: x[0])
        current_edge = edge

        for _, p in ordered:
            if not current_edge.is_valid:
                break

            result = bmesh.ops.bisect_edges(bm, edges=[current_edge], cuts=1)
            new_vert = None
            new_edges = []

            for elem in result.get("geom_split", []):
                if isinstance(elem, bmesh.types.BMVert):
                    new_vert = elem
                elif isinstance(elem, bmesh.types.BMEdge):
                    new_edges.append(elem)

            if new_vert is None:
                continue

            new_vert.co = p
            created += 1

            next_edge = None
            for e in new_edges:
                other = e.other_vert(new_vert)
                if (other.co - b).length < (other.co - a).length:
                    next_edge = e
                    break

            if next_edge is None and new_edges:
                next_edge = new_edges[-1]

            if next_edge is not None:
                current_edge = next_edge

    bm.to_mesh(me)
    me.update()
    bm.free()
    return created


# ============================================================
# Dissolve collinear
# ============================================================

def dissolve_collinear_mesh(obj):
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)

    to_dissolve = []

    for v in bm.verts:
        if len(v.link_edges) != 2:
            continue
        if len(v.link_faces) != 0:
            continue

        e1, e2 = v.link_edges
        a = e1.other_vert(v).co - v.co
        b = e2.other_vert(v).co - v.co

        if a.length <= EPS or b.length <= EPS:
            continue

        a.normalize()
        b.normalize()

        if abs(abs(a.dot(b)) - 1.0) <= COLLINEAR_EPS:
            to_dissolve.append(v)

    count = len(to_dissolve)

    if to_dissolve:
        bmesh.ops.dissolve_verts(bm, verts=to_dissolve)

    bm.to_mesh(me)
    me.update()
    bm.free()
    return count


def dissolve_collinear_curve(obj):
    curve_data = obj.data
    old_splines = snapshot_curve_data(curve_data)

    curve_data.splines.clear()
    removed = 0

    for src in old_splines:
        pts = src["points"]
        n = len(pts)
        cyclic = src["use_cyclic_u"]

        if n <= 2:
            keep = pts[:]
        else:
            keep = []
            for i in range(n):
                if not cyclic and (i == 0 or i == n - 1):
                    keep.append(pts[i])
                    continue

                prev_i = (i - 1) % n
                next_i = (i + 1) % n

                if src["type"] == 'BEZIER':
                    prev_co = pts[prev_i]["co"]
                    curr_co = pts[i]["co"]
                    next_co = pts[next_i]["co"]
                else:
                    prev_co = Vector(pts[prev_i]["co"][:3])
                    curr_co = Vector(pts[i]["co"][:3])
                    next_co = Vector(pts[next_i]["co"][:3])

                v1 = prev_co - curr_co
                v2 = next_co - curr_co

                if v1.length <= EPS or v2.length <= EPS:
                    removed += 1
                    continue

                v1.normalize()
                v2.normalize()

                if abs(abs(v1.dot(v2)) - 1.0) <= COLLINEAR_EPS:
                    removed += 1
                else:
                    keep.append(pts[i])

            if len(keep) < 2:
                keep = pts[:2]

        spline = curve_data.splines.new(src["type"])
        spline.use_cyclic_u = src["use_cyclic_u"]
        spline.resolution_u = src["resolution_u"]
        spline.order_u = src["order_u"]
        spline.use_endpoint_u = src["use_endpoint_u"]
        spline.use_smooth = src["use_smooth"]

        if src["type"] == 'BEZIER':
            spline.bezier_points.add(len(keep) - 1)
            for i, p in enumerate(keep):
                bp = spline.bezier_points[i]
                bp.co = p["co"]
                bp.handle_left = p["handle_left"]
                bp.handle_right = p["handle_right"]
                bp.handle_left_type = p["handle_left_type"]
                bp.handle_right_type = p["handle_right_type"]
                bp.radius = p["radius"]
                bp.tilt = p["tilt"]
        else:
            spline.points.add(len(keep) - 1)
            for i, p in enumerate(keep):
                cp = spline.points[i]
                cp.co = p["co"]
                cp.radius = p["radius"]
                cp.tilt = p["tilt"]
                cp.weight = p.get("weight", 1.0)

    curve_data.update_tag()
    return removed


# ============================================================
# Circle / arc refit
# ============================================================

def bezier_sample_world_points(obj, spline, per_segment=8):
    mw = obj.matrix_world
    pts = spline.bezier_points
    n = len(pts)
    if n < 2:
        return []

    out = []
    limit = n if spline.use_cyclic_u else n - 1

    for i in range(limit):
        j = (i + 1) % n
        a = pts[i]
        b = pts[j]
        p0 = mw @ a.co
        p1 = mw @ a.handle_right
        p2 = mw @ b.handle_left
        p3 = mw @ b.co

        for s in range(per_segment):
            t = s / per_segment
            out.append(cubic_bezier_eval(p0, p1, p2, p3, t))

    out.append(mw @ pts[-1].co)
    return out


def compute_best_fit_plane(points):
    if len(points) < 3:
        return None, None, None

    center = Vector((0.0, 0.0, 0.0))
    for p in points:
        center += p
    center /= len(points)

    n = Vector((0.0, 0.0, 0.0))
    for i in range(len(points) - 1):
        p0 = points[i] - center
        p1 = points[i + 1] - center
        n += p0.cross(p1)

    if points[0] != points[-1]:
        p0 = points[-1] - center
        p1 = points[0] - center
        n += p0.cross(p1)

    nrm = safe_normal(n)
    if nrm is None:
        return None, None, None

    ref = Vector((1.0, 0.0, 0.0))
    if abs(nrm.dot(ref)) > 0.99:
        ref = Vector((0.0, 1.0, 0.0))

    x_axis = safe_normal(ref - nrm * ref.dot(nrm))
    if x_axis is None:
        return None, None, None
    y_axis = nrm.cross(x_axis)
    return center, x_axis, y_axis


def project_points_2d(points, origin, x_axis, y_axis):
    pts2 = []
    for p in points:
        d = p - origin
        pts2.append((d.dot(x_axis), d.dot(y_axis)))
    return pts2


def fit_circle_2d(points2):
    n = len(points2)
    if n < 3:
        return None

    sx = sy = sxx = syy = sxy = 0.0
    sxxx = syyy = sxxy = sxyy = 0.0

    for x, y in points2:
        xx = x * x
        yy = y * y
        sx += x
        sy += y
        sxx += xx
        syy += yy
        sxy += x * y
        sxxx += xx * x
        syyy += yy * y
        sxxy += xx * y
        sxyy += x * yy

    c = n * sxx - sx * sx
    d = n * sxy - sx * sy
    e = n * (sxxx + sxyy) - (sxx + syy) * sx
    g = n * syy - sy * sy
    h = n * (sxxy + syyy) - (sxx + syy) * sy

    denom = 2.0 * (c * g - d * d)
    if abs(denom) < EPS:
        return None

    xc = (g * e - d * h) / denom
    yc = (c * h - d * e) / denom

    rs = []
    for x, y in points2:
        rs.append(sqrt((x - xc) ** 2 + (y - yc) ** 2))
    r = sum(rs) / len(rs)
    resid = sum((ri - r) ** 2 for ri in rs) / max(1, len(rs))

    return xc, yc, r, resid


def signed_angle_delta(a0, a1):
    d = a1 - a0
    while d <= -pi:
        d += 2.0 * pi
    while d > pi:
        d -= 2.0 * pi
    return d


def arc_handle_factor(theta):
    return (4.0 / 3.0) * tan_half_angle(theta / 2.0)


def tan_half_angle(a):
    c = cos(a)
    s = sin(a)
    return s / max(EPS, (1.0 + c))


def rebuild_bezier_spline_as_arc(obj, spline_index, center3d, x_axis, y_axis, angle_list, radius, cyclic):
    curve_data = obj.data
    old_splines = snapshot_curve_data(curve_data)

    if spline_index < 0 or spline_index >= len(old_splines):
        return False

    src = old_splines[spline_index]
    mw_inv = obj.matrix_world.inverted()

    segment_controls = []
    for i in range(len(angle_list) - 1):
        a0 = angle_list[i]
        a1 = angle_list[i + 1]
        theta = a1 - a0
        k = arc_handle_factor(theta)

        r0 = Vector((cos(a0), sin(a0), 0.0))
        r1 = Vector((cos(a1), sin(a1), 0.0))
        t0 = Vector((-sin(a0), cos(a0), 0.0))
        t1 = Vector((-sin(a1), cos(a1), 0.0))

        p0w = center3d + (x_axis * r0.x + y_axis * r0.y) * radius
        p3w = center3d + (x_axis * r1.x + y_axis * r1.y) * radius
        p1w = p0w + (x_axis * t0.x + y_axis * t0.y) * (k * radius)
        p2w = p3w - (x_axis * t1.x + y_axis * t1.y) * (k * radius)

        segment_controls.append((
            mw_inv @ p0w,
            mw_inv @ p1w,
            mw_inv @ p2w,
            mw_inv @ p3w
        ))

    pts = []
    first = segment_controls[0]

    pts.append({
        "co": first[0],
        "handle_left": segment_controls[-1][2] if cyclic else first[0],
        "handle_right": first[1],
        "handle_left_type": 'FREE',
        "handle_right_type": 'FREE',
        "radius": 1.0,
        "tilt": 0.0,
    })

    for i, seg in enumerate(segment_controls):
        p0, p1, p2, p3 = seg
        if i < len(segment_controls) - 1:
            hr = segment_controls[i + 1][1]
        else:
            hr = segment_controls[0][1] if cyclic else p3

        pts.append({
            "co": p3,
            "handle_left": p2,
            "handle_right": hr,
            "handle_left_type": 'FREE',
            "handle_right_type": 'FREE',
            "radius": 1.0,
            "tilt": 0.0,
        })

    if cyclic and len(pts) > 1 and (pts[0]["co"] - pts[-1]["co"]).length <= POINT_EPS:
        pts.pop()

    old_splines[spline_index] = {
        "type": 'BEZIER',
        "use_cyclic_u": cyclic,
        "resolution_u": src["resolution_u"],
        "order_u": src["order_u"],
        "use_endpoint_u": src["use_endpoint_u"],
        "use_smooth": src["use_smooth"],
        "points": pts,
    }

    curve_data.splines.clear()

    for src_spline in old_splines:
        spline = curve_data.splines.new(src_spline["type"])
        spline.use_cyclic_u = src_spline["use_cyclic_u"]
        spline.resolution_u = src_spline["resolution_u"]
        spline.order_u = src_spline["order_u"]
        spline.use_endpoint_u = src_spline["use_endpoint_u"]
        spline.use_smooth = src_spline["use_smooth"]

        if src_spline["type"] == 'BEZIER':
            spline.bezier_points.add(len(src_spline["points"]) - 1)
            for i, p in enumerate(src_spline["points"]):
                bp = spline.bezier_points[i]
                bp.co = p["co"]
                bp.handle_left = p["handle_left"]
                bp.handle_right = p["handle_right"]
                bp.handle_left_type = p["handle_left_type"]
                bp.handle_right_type = p["handle_right_type"]
                bp.radius = p.get("radius", 1.0)
                bp.tilt = p.get("tilt", 0.0)
        else:
            spline.points.add(len(src_spline["points"]) - 1)
            for i, p in enumerate(src_spline["points"]):
                cp = spline.points[i]
                cp.co = p["co"]
                cp.radius = p["radius"]
                cp.tilt = p["tilt"]
                cp.weight = p.get("weight", 1.0)

    curve_data.update_tag()
    return True


def refit_circles_arcs_on_object(obj, max_radius_error=0.01, max_planar_error=0.001):
    data = obj.data
    changed = 0

    for spline_index, spline in enumerate(list(data.splines)):
        if spline.type != 'BEZIER':
            continue

        sample_pts = bezier_sample_world_points(obj, spline, per_segment=8)
        if len(sample_pts) < 6:
            continue

        plane_center, x_axis, y_axis = compute_best_fit_plane(sample_pts)
        if plane_center is None:
            continue

        normal = x_axis.cross(y_axis)
        normal.normalize()

        plane_err = 0.0
        for p in sample_pts:
            plane_err = max(plane_err, abs((p - plane_center).dot(normal)))

        if plane_err > max_planar_error:
            continue

        pts2 = project_points_2d(sample_pts, plane_center, x_axis, y_axis)
        fit = fit_circle_2d(pts2)
        if fit is None:
            continue

        xc, yc, r, resid = fit
        if r <= EPS:
            continue

        radial_errors = []
        angles = []
        for x, y in pts2:
            dx = x - xc
            dy = y - yc
            rr = sqrt(dx * dx + dy * dy)
            radial_errors.append(abs(rr - r) / max(EPS, r))
            angles.append(atan2(dy, dx))

        max_err = max(radial_errors) if radial_errors else 999.0
        if max_err > max_radius_error:
            continue

        center3d = plane_center + x_axis * xc + y_axis * yc

        if spline.use_cyclic_u:
            angle_list = [0.0, 0.5 * pi, pi, 1.5 * pi, 2.0 * pi]
        else:
            start = angles[0]
            end = angles[-1]
            total = signed_angle_delta(start, end)

            if abs(total) < 1e-4:
                continue

            seg_count = max(1, int(abs(total) / (pi / 2.0)) + 1)
            angle_list = [start + total * (i / seg_count) for i in range(seg_count + 1)]

        if rebuild_bezier_spline_as_arc(obj, spline_index, center3d, x_axis, y_axis, angle_list, r, spline.use_cyclic_u):
            changed += 1

    return changed


# ============================================================
# Operators
# ============================================================

class OBJECT_OT_split_at_intersections(bpy.types.Operator):
    bl_idname = "object.split_at_intersections"
    bl_label = "Split at Intersections"
    bl_description = "Create new points/vertices at intersections and self-intersections"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        objs = get_selected_supported_objects(context)

        if not objs:
            report_msg(self, "Select at least one curve or one edge-only mesh.", {'WARNING'})
            return {'CANCELLED'}

        curve_linear_requests, curve_bezier_requests, mesh_requests = build_intersection_requests(objs)

        total_added = 0
        for obj in objs:
            if obj.type == 'CURVE':
                total_added += rebuild_curve_with_insertions(
                    obj,
                    curve_linear_requests.get(obj.name, []),
                    curve_bezier_requests.get(obj.name, []),
                )
            elif mesh_is_edges_only(obj):
                total_added += split_mesh_edges_at_hits(obj, mesh_requests.get(obj.name, []))

        if total_added == 0:
            report_msg(self, "No intersections found.", {'INFO'})
            return {'CANCELLED'}

        report_msg(self, f"Created {total_added} new intersection points/vertices.")
        return {'FINISHED'}


class OBJECT_OT_dissolve_collinear_vertices(bpy.types.Operator):
    bl_idname = "object.dissolve_collinear_vertices_curves"
    bl_label = "Dissolve Collinear"
    bl_description = "Remove intermediate collinear points/vertices"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        objs = get_selected_supported_objects(context)

        if not objs:
            report_msg(self, "Select at least one curve or one edge-only mesh.", {'WARNING'})
            return {'CANCELLED'}

        total_removed = 0
        for obj in objs:
            if obj.type == 'CURVE':
                total_removed += dissolve_collinear_curve(obj)
            elif mesh_is_edges_only(obj):
                total_removed += dissolve_collinear_mesh(obj)

        if total_removed == 0:
            report_msg(self, "No intermediate collinear points found.", {'INFO'})
            return {'CANCELLED'}

        report_msg(self, f"Removed {total_removed} intermediate collinear points/vertices.")
        return {'FINISHED'}


class OBJECT_OT_refit_circles_arcs(bpy.types.Operator):
    bl_idname = "object.refit_circles_arcs"
    bl_label = "Refit Circles / Arcs"
    bl_description = "Detect near-circular Bézier splines and rebuild them as clean circles/arcs"
    bl_options = {'REGISTER', 'UNDO'}

    max_radius_error: bpy.props.FloatProperty(
        name="Max Radius Error",
        default=0.01,
        min=0.0001,
        max=0.25,
    )
    max_planar_error: bpy.props.FloatProperty(
        name="Max Planar Error",
        default=0.001,
        min=0.000001,
        max=0.1,
    )

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        objs = get_selected_curve_objects(context)

        if not objs:
            report_msg(self, "Select at least one curve object.", {'WARNING'})
            return {'CANCELLED'}

        changed = 0
        for obj in objs:
            changed += refit_circles_arcs_on_object(
                obj,
                max_radius_error=self.max_radius_error,
                max_planar_error=self.max_planar_error,
            )

        if changed == 0:
            report_msg(self, "No near-circular Bézier splines found.", {'INFO'})
            return {'CANCELLED'}

        report_msg(self, f"Refit {changed} spline(s) as circles/arcs.")
        return {'FINISHED'}


# ============================================================
# UI
# ============================================================

class VIEW3D_PT_intersection_snap_tools(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'HEADER'
    bl_label = "Intersection Snap Tools"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.operator("object.split_at_intersections", icon='SNAP_VERTEX')
        col.operator("object.dissolve_collinear_vertices_curves", icon='X')
        col.operator("object.refit_circles_arcs", icon='MESH_CIRCLE')


class VIEW3D_MT_intersection_snap_context(bpy.types.Menu):
    bl_label = "Intersection Snap Tools"

    def draw(self, context):
        layout = self.layout
        layout.operator("object.split_at_intersections", icon='SNAP_VERTEX')
        layout.operator("object.dissolve_collinear_vertices_curves", icon='X')
        layout.operator("object.refit_circles_arcs", icon='MESH_CIRCLE')


def draw_view3d_header_extension(self, context):
    row = self.layout.row(align=True)
    row.separator(factor=0.35)
    row.popover(
        panel="VIEW3D_PT_intersection_snap_tools",
        text="Intersections",
        icon='SNAP_VERTEX'
    )


def draw_object_context_menu(self, context):
    selected = get_selected_supported_objects(context)
    if not selected:
        return

    layout = self.layout
    layout.separator()
    layout.menu("VIEW3D_MT_intersection_snap_context", icon='SNAP_VERTEX')


# ============================================================
# Register
# ============================================================

classes = (
    OBJECT_OT_split_at_intersections,
    OBJECT_OT_dissolve_collinear_vertices,
    OBJECT_OT_refit_circles_arcs,
    VIEW3D_PT_intersection_snap_tools,
    VIEW3D_MT_intersection_snap_context,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.VIEW3D_HT_header.append(draw_view3d_header_extension)
    bpy.types.VIEW3D_MT_object_context_menu.append(draw_object_context_menu)


def unregister():
    try:
        bpy.types.VIEW3D_HT_header.remove(draw_view3d_header_extension)
    except Exception:
        pass

    try:
        bpy.types.VIEW3D_MT_object_context_menu.remove(draw_object_context_menu)
    except Exception:
        pass

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()