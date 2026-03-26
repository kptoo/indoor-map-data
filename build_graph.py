"""
build_graph.py
==============
Builds a dense, topologically-snapped routing graph from corridor + connect
GeoJSON files for every terminal.

Changes vs previous version:
  - Subdivides every segment into MAX_SEG_M metre intervals (default 3 m)
    so that POI snap targets are always close to the actual corridor line.

Usage:
    python build_graph.py

Requirements:
    pip install shapely networkx
"""

import json
import os
import math

from shapely.geometry import shape, LineString
from shapely.ops import unary_union, snap as shapely_snap
import networkx as nx


# ── Configuration ─────────────────────────────────────────────────────────────

TERMINALS = {
    "T1":  {"folder": "Terminal_1",        "prefix": "T1"},
    "T2":  {"folder": "Terminal_2",        "prefix": "T2"},
    "T3":  {"folder": "Terminal_3",        "prefix": "T3"},
    "T4":  {"folder": "Terminal_4",        "prefix": "T4"},
    "T6":  {"folder": "Terminal_6",        "prefix": "T6"},
    "T7":  {"folder": "Terminal_7",        "prefix": "T7"},
    "T8":  {"folder": "Terminal_8",        "prefix": "T8"},
    "TB":  {"folder": "Terminal_B",        "prefix": "TB"},
    "Reg": {"folder": "Terminal_Regional", "prefix": "Reg"},
    "W":   {"folder": "Terminal_Wgates",   "prefix": "W"},
}

LEVELS         = ["1", "2", "3"]
SNAP_TOL       = 0.00002   # ~2 m — endpoint snapping tolerance
MAX_STITCH_M   = 50        # max gap to bridge between components
STITCH_PENALTY = 10        # cost multiplier for stitch bridge edges
MAX_SEG_M      = 3.0       # subdivide segments longer than this (metres)


# ── Helpers ───────────────────────────────────────────────────────────────────

def haversine_m(lng1, lat1, lng2, lat2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def rc(v):
    """Round coordinate to 7 decimal places (~1 cm precision)."""
    return round(v, 7)


def load_geojson(path):
    if not os.path.exists(path):
        return {"type": "FeatureCollection", "features": []}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_lines(geojson, level):
    lines = []
    for feat in geojson.get("features", []):
        props = feat.get("properties") or {}
        feat_level = str(props.get("level") or props.get("Level") or "3")
        if feat_level != level:
            continue
        geom = feat.get("geometry")
        if not geom:
            continue
        s = shape(geom)
        if s.geom_type == "LineString":
            lines.append(s)
        elif s.geom_type == "MultiLineString":
            lines.extend(s.geoms)
    return lines


def snap_lines(lines):
    """Snap all line endpoints together to close micro-gaps."""
    if not lines:
        return []
    reference = unary_union(lines)
    snapped = []
    for ln in lines:
        try:
            s = shapely_snap(ln, reference, SNAP_TOL)
            if s.geom_type == "LineString" and len(s.coords) >= 2:
                snapped.append(s)
            elif s.geom_type == "MultiLineString":
                snapped.extend(g for g in s.geoms if len(g.coords) >= 2)
        except Exception:
            snapped.append(ln)
    return snapped


def subdivide_segment(a, b, max_m=MAX_SEG_M):
    """
    Split the segment a→b into sub-segments of at most max_m metres.
    Returns list of coordinate pairs covering the full segment.
    """
    seg_len = haversine_m(a[0], a[1], b[0], b[1])
    if seg_len <= max_m:
        return [(a, b)]

    n = math.ceil(seg_len / max_m)
    pts = []
    for i in range(n):
        t0 = i / n
        t1 = (i + 1) / n
        p0 = (a[0] + t0 * (b[0] - a[0]), a[1] + t0 * (b[1] - a[1]))
        p1 = (a[0] + t1 * (b[0] - a[0]), a[1] + t1 * (b[1] - a[1]))
        pts.append((p0, p1))
    return pts


def build_nx_graph(lines):
    """
    Build a NetworkX graph from snapped lines.
    Every segment is subdivided into MAX_SEG_M metre intervals so
    POI snap targets are always close to the actual corridor geometry.
    """
    G = nx.Graph()
    for ln in lines:
        coords = list(ln.coords)
        for i in range(len(coords) - 1):
            a = (rc(coords[i][0]),   rc(coords[i][1]))
            b = (rc(coords[i+1][0]), rc(coords[i+1][1]))
            if a == b:
                continue
            # Subdivide long segments
            for (p0, p1) in subdivide_segment(a, b):
                na = (rc(p0[0]), rc(p0[1]))
                nb = (rc(p1[0]), rc(p1[1]))
                if na == nb:
                    continue
                cost = haversine_m(na[0], na[1], nb[0], nb[1])
                if G.has_edge(na, nb):
                    if G[na][nb]["cost"] > cost:
                        G[na][nb]["cost"] = cost
                else:
                    G.add_edge(na, nb, cost=cost)
    return G


def stitch(G):
    """Connect isolated components with penalised bridge edges (max MAX_STITCH_M)."""
    while True:
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        if len(comps) <= 1:
            break
        main = comps[0]
        stitched = False
        for i in range(1, len(comps)):
            small = comps[i]
            best_d, best_pair = math.inf, None
            for sn in small:
                for mn in main:
                    d = haversine_m(sn[0], sn[1], mn[0], mn[1])
                    if d < best_d:
                        best_d, best_pair = d, (sn, mn)
            if best_pair and best_d <= MAX_STITCH_M:
                G.add_edge(best_pair[0], best_pair[1],
                           cost=best_d * STITCH_PENALTY, stitch=True)
                main = main | small
                stitched = True
        if not stitched:
            break
    return G


def serialise(G, level):
    node_list = list(G.nodes())
    id_map = {n: str(i) for i, n in enumerate(node_list)}
    nodes = {
        id_map[n]: {"lng": n[0], "lat": n[1], "level": level}
        for n in node_list
    }
    edges = []
    seen = set()
    for u, v, data in G.edges(data=True):
        key = tuple(sorted([id_map[u], id_map[v]]))
        if key in seen:
            continue
        seen.add(key)
        e = {
            "from": id_map[u],
            "to":   id_map[v],
            "cost": round(data["cost"], 4),
            "level": level
        }
        if data.get("stitch"):
            e["stitch"] = True
        edges.append(e)
    return nodes, edges


# ── Per-terminal processing ───────────────────────────────────────────────────

def process_terminal(term_key, folder, prefix):
    print(f"\n{'='*55}\n  {term_key}  ({folder})\n{'='*55}")

    corr_gj = load_geojson(os.path.join(folder, f"{prefix}_corridors.geojson"))
    conn_gj = load_geojson(os.path.join(folder, f"{prefix}_Connect.geojson"))

    all_nodes, all_edges = {}, []

    for level in LEVELS:
        print(f"  L{level} … ", end="", flush=True)
        lines = extract_lines(corr_gj, level) + extract_lines(conn_gj, level)
        if not lines:
            print("no geometry.")
            continue

        snapped = snap_lines(lines)
        G = build_nx_graph(snapped)
        G = stitch(G)

        n_comps = nx.number_connected_components(G)
        print(f"{G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges, "
              f"{n_comps} component(s) ✓")

        nodes, edges = serialise(G, level)
        all_nodes.update(nodes)
        all_edges.extend(edges)

    out_path = os.path.join(folder, "graph.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"nodes": all_nodes, "edges": all_edges},
                  f, separators=(",", ":"))
    print(f"  → {out_path}  ({os.path.getsize(out_path)/1024:.1f} KB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for term_key, cfg in TERMINALS.items():
        if not os.path.isdir(cfg["folder"]):
            print(f"⚠️  Skipping {term_key} — folder not found: {cfg['folder']}")
            continue
        process_terminal(term_key, cfg["folder"], cfg["prefix"])
    print("\n🎉  Done.")


if __name__ == "__main__":
    main()
