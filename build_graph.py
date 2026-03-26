"""
build_graph.py
==============
Pre-processes corridor + connect GeoJSON files for every terminal and outputs
a clean, topologically-snapped routing graph as  <terminal_folder>/graph.json

Usage:
    python build_graph.py

Requirements:
    pip install shapely networkx

Output format (graph.json):
{
  "nodes": {
    "<id>": { "lng": 103.123, "lat": 1.456, "level": "3" }
  },
  "edges": [
    { "from": "<id>", "to": "<id>", "cost": 12.3, "level": "3" }
  ]
}
"""

import json
import os
import math
import itertools
from collections import defaultdict

try:
    from shapely.geometry import shape, MultiLineString, LineString, Point
    from shapely.ops import unary_union, snap as shapely_snap, split
except ImportError:
    raise SystemExit("Install shapely:  pip install shapely")

try:
    import networkx as nx
except ImportError:
    raise SystemExit("Install networkx:  pip install networkx")


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

# All levels to process per terminal
LEVELS = ["1", "2", "3"]

# Snap tolerance in degrees (~1 m at equator ≈ 0.000009°, use 2 m)
SNAP_TOLERANCE = 0.00002

# ── Helpers ─────────────────────────���─────────────────────────────────────────

def haversine_m(lng1, lat1, lng2, lat2):
    """Approximate distance in metres between two lng/lat points."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def round_coord(v, decimals=7):
    return round(v, decimals)


def coord_key(lng, lat):
    return (round_coord(lng), round_coord(lat))


def load_geojson(path):
    if not os.path.exists(path):
        return {"type": "FeatureCollection", "features": []}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_lines(geojson, level):
    """Return list of shapely LineString objects for the given level."""
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


def snap_lines_together(lines, tolerance=SNAP_TOLERANCE):
    """
    Snap all lines to each other so endpoints that are within `tolerance`
    degrees are merged to the same coordinate.  Returns a new list of
    LineStrings with unified coordinates.
    """
    if not lines:
        return []

    # Build a unified reference geometry and snap everything to it
    reference = unary_union(lines)
    snapped = []
    for ln in lines:
        try:
            s = shapely_snap(ln, reference, tolerance)
            if s.geom_type == "LineString" and len(s.coords) >= 2:
                snapped.append(s)
            elif s.geom_type == "MultiLineString":
                snapped.extend(g for g in s.geoms if len(g.coords) >= 2)
        except Exception:
            snapped.append(ln)
    return snapped


def build_graph_for_level(lines):
    """
    Given a list of snapped LineStrings, build a NetworkX graph.
    Nodes are (lng, lat) tuples.  Edge weight = haversine distance in metres.
    Returns nx.Graph.
    """
    G = nx.Graph()

    for ln in lines:
        coords = list(ln.coords)
        for i in range(len(coords) - 1):
            a = (round_coord(coords[i][0]),   round_coord(coords[i][1]))
            b = (round_coord(coords[i+1][0]), round_coord(coords[i+1][1]))
            if a == b:
                continue
            cost = haversine_m(a[0], a[1], b[0], b[1])
            # Keep the cheaper edge if it already exists
            if G.has_edge(a, b):
                if G[a][b]["cost"] > cost:
                    G[a][b]["cost"] = cost
            else:
                G.add_edge(a, b, cost=cost)

    return G


def stitch_components(G, max_gap_m=50):
    """
    Connect disconnected components by adding a bridge edge between the two
    closest nodes across components, but only if they are within max_gap_m
    metres.  Repeats until no more stitchable pairs exist.
    """
    while True:
        comps = list(nx.connected_components(G))
        if len(comps) <= 1:
            break

        # Sort largest first so small orphans attach to the main body
        comps.sort(key=len, reverse=True)
        main = comps[0]
        stitched = False

        for i in range(1, len(comps)):
            small = comps[i]
            best_dist = math.inf
            best_pair = None

            for sn in small:
                for mn in main:
                    d = haversine_m(sn[0], sn[1], mn[0], mn[1])
                    if d < best_dist:
                        best_dist = d
                        best_pair = (sn, mn)

            if best_pair and best_dist <= max_gap_m:
                sn, mn = best_pair
                # Penalise stitch edges 10× so real corridors are always preferred
                G.add_edge(sn, mn, cost=best_dist * 10, stitch=True)
                # Merge into main for subsequent iterations
                main = main | small
                stitched = True

        if not stitched:
            break

    return G


def graph_to_json(G, level):
    """Serialise a NetworkX graph to the JSON format expected by the JS router."""
    nodes = {}
    node_list = list(G.nodes())

    # Assign integer IDs for compact JSON
    id_map = {n: str(i) for i, n in enumerate(node_list)}

    for n, nid in id_map.items():
        nodes[nid] = {"lng": n[0], "lat": n[1], "level": level}

    edges = []
    seen = set()
    for u, v, data in G.edges(data=True):
        eid = (id_map[u], id_map[v])
        if eid in seen or (eid[1], eid[0]) in seen:
            continue
        seen.add(eid)
        edges.append({
            "from": id_map[u],
            "to":   id_map[v],
            "cost": round(data["cost"], 4),
            "level": level,
            **({"stitch": True} if data.get("stitch") else {})
        })

    return nodes, edges


# ── Main ──────────────────────────────────────────────────────────────────────

def process_terminal(term_key, folder, prefix):
    print(f"\n{'='*60}")
    print(f"  Terminal: {term_key}  ({folder})")
    print(f"{'='*60}")

    corr_path = os.path.join(folder, f"{prefix}_corridors.geojson")
    conn_path = os.path.join(folder, f"{prefix}_Connect.geojson")

    corridors_gj = load_geojson(corr_path)
    connect_gj   = load_geojson(conn_path)

    all_nodes = {}
    all_edges = []

    for level in LEVELS:
        print(f"  Level {level} …", end=" ")

        lines = extract_lines(corridors_gj, level) + extract_lines(connect_gj, level)

        if not lines:
            print("no geometry, skipped.")
            continue

        print(f"{len(lines)} segments →", end=" ")

        # 1. Snap endpoints together to close micro-gaps
        snapped = snap_lines_together(lines, SNAP_TOLERANCE)
        print(f"snapped →", end=" ")

        # 2. Build graph
        G = build_graph_for_level(snapped)
        print(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges →", end=" ")

        # 3. Stitch disconnected components
        G = stitch_components(G, max_gap_m=50)
        comps = nx.number_connected_components(G)
        print(f"{comps} component(s) after stitch →", end=" ")

        # 4. Serialise
        nodes, edges = graph_to_json(G, level)
        print(f"serialised ✓")

        all_nodes.update(nodes)
        all_edges.extend(edges)

    out = {"nodes": all_nodes, "edges": all_edges}
    out_path = os.path.join(folder, "graph.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\n  ✅  Written: {out_path}  ({size_kb:.1f} KB)")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    for term_key, cfg in TERMINALS.items():
        folder = cfg["folder"]
        prefix = cfg["prefix"]
        if not os.path.isdir(folder):
            print(f"⚠️  Folder not found, skipping: {folder}")
            continue
        process_terminal(term_key, folder, prefix)

    print("\n🎉  All terminals processed.")


if __name__ == "__main__":
    main()
