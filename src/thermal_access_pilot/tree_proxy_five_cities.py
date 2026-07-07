from __future__ import annotations

import argparse
import json
import math
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union


ROOT = Path("/Users/gk/Code/super-duper-disser/thermal_access_pilot")
RAW_DIR = ROOT / "cache/five_city_tree_proxy/raw"
OUT_DIR = ROOT / "outputs/five_city_tree_proxy"


@dataclass(frozen=True)
class CitySpec:
    slug: str
    title: str
    country: str
    center_lon: float
    center_lat: float
    square_size_m: float = 5000.0
    walk_speed_m_s: float = 1.2
    shadow_weight_a: float = 1.35
    tree_source: str = "official"


CITIES = [
    CitySpec("bergen_norway", "Bergen", "Norway", 5.32415, 60.39299, tree_source="official"),
    CitySpec("bologna_italy", "Bologna", "Italy", 11.3426, 44.4949, tree_source="official"),
    CitySpec("brno_czechia", "Brno", "Czechia", 16.6068, 49.1951, tree_source="official_proxy"),
    CitySpec("coimbra_portugal", "Coimbra", "Portugal", -8.4292, 40.2110, tree_source="osm_fallback"),
    CitySpec("delft_netherlands", "Delft", "Netherlands", 4.3571, 52.0116, tree_source="official_proxy"),
]


def _download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path


def _class_midpoint(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().replace(",", ".")
    numbers = [float(match) for match in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return None
    if text.startswith("<"):
        return numbers[0] / 2.0
    if text.startswith(">"):
        return numbers[0]
    if len(numbers) >= 2:
        return (numbers[0] + numbers[1]) / 2.0
    return numbers[0]


def _radius_from_crown_width_class(value: object) -> float | None:
    midpoint = _class_midpoint(value)
    return None if midpoint is None else midpoint / 2.0


def _utm_crs(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def _study_polygon(city: CitySpec) -> tuple[object, CRS]:
    crs = _utm_crs(city.center_lon, city.center_lat)
    to_metric = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = to_metric.transform(city.center_lon, city.center_lat)
    half = city.square_size_m / 2.0
    return box(x - half, y - half, x + half, y + half), crs


def _city_dirs(city: CitySpec) -> tuple[Path, Path]:
    raw = RAW_DIR / city.slug
    out = OUT_DIR / city.slug
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    return raw, out


def _fetch_osm_buildings_stops_graph(city: CitySpec, polygon, crs: CRS) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, nx.Graph, gpd.GeoDataFrame]:
    polygon_wgs = gpd.GeoSeries([polygon], crs=crs).to_crs(4326).iloc[0]
    buildings = ox.features_from_polygon(polygon_wgs, tags={"building": True}).reset_index()
    buildings = buildings.loc[buildings.geometry.notna()].copy()
    buildings = buildings.to_crs(crs)
    buildings = buildings.loc[buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    buildings["building_id"] = range(len(buildings))
    levels = pd.to_numeric(buildings.get("building:levels"), errors="coerce")
    heights = pd.to_numeric(buildings.get("height"), errors="coerce")
    buildings["height_m"] = heights.where(heights > 0, levels * 3.0).fillna(3.0)

    stop_tags = {
        "public_transport": ["platform", "stop_position"],
        "highway": "bus_stop",
        "railway": ["tram_stop", "station", "halt", "subway_entrance"],
        "amenity": "bus_station",
    }
    stops = ox.features_from_polygon(polygon_wgs, tags=stop_tags).reset_index()
    if stops.empty:
        raise ValueError(f"[{city.slug}] no PT stops in 5x5 km study area")
    stops = stops.loc[stops.geometry.notna()].copy().to_crs(crs)
    stops["geometry"] = stops.geometry.representative_point()
    stops["stop_id"] = range(len(stops))

    graph = ox.graph_from_polygon(polygon_wgs, network_type="walk", simplify=True)
    graph = ox.project_graph(graph, to_crs=crs)
    nodes, edges = ox.graph_to_gdfs(graph)
    nodes = nodes.reset_index().rename(columns={"osmid": "node_id"})
    edges = edges.reset_index()
    graph2 = nx.Graph()
    for row in nodes.itertuples(index=False):
        graph2.add_node(int(row.node_id), geometry=row.geometry)
    for row in edges.itertuples(index=False):
        u, v = int(row.u), int(row.v)
        geom = row.geometry
        length = float(getattr(row, "length", geom.length))
        if graph2.has_edge(u, v) and graph2[u][v]["length_m"] <= length:
            continue
        graph2.add_edge(u, v, geometry=geom, length_m=length)
    return buildings, stops, graph2, nodes[["node_id", "geometry"]].copy().set_crs(crs)


def _load_bologna_trees(city: CitySpec, polygon, crs: CRS) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(
        "https://opendata.comune.bologna.it/api/explore/v2.1/catalog/datasets/popolazione-arborea/exports/geojson?limit=-1"
    ).to_crs(crs)
    gdf = gdf.loc[gdf.intersects(polygon)].copy()
    gdf["height_m"] = gdf["classe_altezza"].apply(_class_midpoint).fillna(6.0)
    gdf["tree_point"] = gdf.geometry.representative_point()
    gdf["radius_m"] = (gdf.geometry.area / math.pi).pow(0.5)
    return gpd.GeoDataFrame(
        {
            "tree_id": gdf.index.astype(str),
            "height_m": gdf["height_m"],
            "radius_m": gdf["radius_m"],
            "source_note": "official_polygon",
        },
        geometry=gdf["tree_point"],
        crs=crs,
    )


def _load_brno_trees(city: CitySpec, polygon, crs: CRS) -> gpd.GeoDataFrame:
    local = _download("https://gis.brno.cz/public/opendata/pz_g_bio2_wgs84.gpkg", RAW_DIR / city.slug / "trees.gpkg")
    bbox = tuple(gpd.GeoSeries([polygon], crs=crs).to_crs(4326).total_bounds.tolist())
    gdf = gpd.read_file(local, bbox=bbox).to_crs(crs)
    gdf = gdf.loc[gdf.intersects(polygon)].copy()
    gdf["geometry"] = gdf.geometry.representative_point()
    return gpd.GeoDataFrame(
        {
            "tree_id": gdf["ogcfid"].astype(str),
            "height_m": 9.0,
            "radius_m": 3.0,
            "source_note": "official_points_default_proxy",
        },
        geometry=gdf.geometry,
        crs=crs,
    )


def _arcgis_geojson_bbox(url: str, polygon, crs: CRS, out_fields: str = "*") -> gpd.GeoDataFrame:
    bbox = gpd.GeoSeries([polygon], crs=crs).to_crs(4326).total_bounds
    minx, miny, maxx, maxy = bbox
    query = (
        f"{url}/query?where=1%3D1&outFields={urllib.parse.quote(out_fields)}"
        f"&geometry={minx},{miny},{maxx},{maxy}&geometryType=esriGeometryEnvelope&inSR=4326"
        "&spatialRel=esriSpatialRelIntersects&f=geojson"
    )
    return gpd.read_file(query).to_crs(crs)


def _load_bergen_trees(city: CitySpec, polygon, crs: CRS) -> gpd.GeoDataFrame:
    url = "https://services2.arcgis.com/uFceZIl8p8frfdd8/arcgis/rest/services/Bytr%C3%A6r_Bergenhus_WFL1/FeatureServer/0"
    gdf = _arcgis_geojson_bbox(url, polygon, crs)
    gdf["height_m"] = gdf["Trehøyde"].apply(_class_midpoint).fillna(8.0)
    gdf["radius_m"] = gdf["Kronebredd"].apply(_radius_from_crown_width_class).fillna(2.5)
    return gpd.GeoDataFrame(
        {
            "tree_id": gdf["ID"].astype(str),
            "height_m": gdf["height_m"],
            "radius_m": gdf["radius_m"],
            "source_note": "official_points",
        },
        geometry=gdf.geometry,
        crs=crs,
    )


def _load_delft_trees(city: CitySpec, polygon, crs: CRS) -> gpd.GeoDataFrame:
    url = "https://services3.arcgis.com/j07voPd56xoB4c87/arcgis/rest/services/Bomen%20in%20beheer%20door%20gemeente%20Delft/FeatureServer/0"
    gdf = _arcgis_geojson_bbox(url, polygon, crs)
    gdf["height_m"] = pd.to_numeric(gdf["HOOGTE"], errors="coerce").fillna(9.0)
    gdf["radius_m"] = (pd.to_numeric(gdf["DIAMETER"], errors="coerce") / 2.0).fillna(3.0)
    return gpd.GeoDataFrame(
        {
            "tree_id": gdf["ID"].astype(str),
            "height_m": gdf["height_m"],
            "radius_m": gdf["radius_m"],
            "source_note": "official_points_default_proxy_when_missing",
        },
        geometry=gdf.geometry,
        crs=crs,
    )


def _load_coimbra_trees(city: CitySpec, polygon, crs: CRS) -> gpd.GeoDataFrame:
    polygon_wgs = gpd.GeoSeries([polygon], crs=crs).to_crs(4326).iloc[0]
    gdf = ox.features_from_polygon(polygon_wgs, tags={"natural": "tree"}).reset_index()
    if gdf.empty:
        return gpd.GeoDataFrame({"tree_id": [], "height_m": [], "radius_m": [], "source_note": []}, geometry=[], crs=crs)
    gdf = gdf.to_crs(crs)
    gdf["geometry"] = gdf.geometry.representative_point()
    return gpd.GeoDataFrame(
        {
            "tree_id": gdf.index.astype(str),
            "height_m": 9.0,
            "radius_m": 3.0,
            "source_note": "osm_tree_fallback_official_portal_unavailable",
        },
        geometry=gdf.geometry,
        crs=crs,
    )


def load_trees(city: CitySpec, polygon, crs: CRS) -> gpd.GeoDataFrame:
    if city.slug == "bologna_italy":
        trees = _load_bologna_trees(city, polygon, crs)
    elif city.slug == "brno_czechia":
        trees = _load_brno_trees(city, polygon, crs)
    elif city.slug == "bergen_norway":
        trees = _load_bergen_trees(city, polygon, crs)
    elif city.slug == "delft_netherlands":
        trees = _load_delft_trees(city, polygon, crs)
    elif city.slug == "coimbra_portugal":
        trees = _load_coimbra_trees(city, polygon, crs)
    else:
        raise KeyError(city.slug)
    return trees.loc[(trees["radius_m"] > 0) & (trees["height_m"] > 0)].reset_index(drop=True)


def _tree_canopies(trees: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    canopies = trees.copy()
    canopies["geometry"] = [geom.buffer(radius, quad_segs=2) for geom, radius in zip(canopies.geometry, canopies["radius_m"])]
    return canopies


def _snap_points_to_nodes(points: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, id_col: str) -> gpd.GeoDataFrame:
    snapped = gpd.sjoin_nearest(points[[id_col, "geometry"]], nodes[["node_id", "geometry"]], how="left", distance_col="snap_m")
    return snapped.drop(columns=["index_right"]).dropna(subset=["node_id"]).astype({"node_id": int})


def _edge_shade_fraction(graph: nx.Graph, canopy_union) -> nx.Graph:
    out = graph.copy()
    for u, v, data in out.edges(data=True):
        geom = data["geometry"]
        length = max(float(data["length_m"]), 1.0)
        inter = geom.intersection(canopy_union) if canopy_union and not canopy_union.is_empty else LineString()
        shaded = float(inter.length) if not inter.is_empty else 0.0
        data["shade_length_m"] = min(shaded, length)
        data["shade_fraction"] = min(max(data["shade_length_m"] / length, 0.0), 1.0)
    return out


def _route_geometry(graph: nx.Graph, path: list[int]) -> LineString:
    coords: list[tuple[float, float]] = []
    for u, v in zip(path, path[1:]):
        segment = list(graph[u][v]["geometry"].coords)
        if coords and coords[-1] == segment[0]:
            coords.extend(segment[1:])
        else:
            coords.extend(segment)
    return LineString(coords) if len(coords) >= 2 else LineString()


def _route_all(graph: nx.Graph, origin_nodes: list[int], stop_nodes: list[int], walk_speed_m_s: float, shadow_weight_a: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    def edge_weight(_, __, data, scenario: str) -> float:
        length = float(data["length_m"])
        if scenario == "baseline":
            return length / walk_speed_m_s
        shade = float(data.get("shade_fraction", 0.0))
        return length * (shade + shadow_weight_a * (1.0 - shade)) / walk_speed_m_s

    rows: list[dict] = []
    routes: list[dict] = []
    for scenario in ["baseline", "shade"]:
        distances, paths = nx.multi_source_dijkstra(
            graph,
            set(stop_nodes),
            weight=lambda u, v, d, sc=scenario: edge_weight(u, v, d, sc),
        )
        for node in origin_nodes:
            if node not in paths:
                rows.append({"origin_node": node, "scenario": scenario, "status": "unreachable"})
                continue
            path = list(reversed(paths[node]))
            length = shade_len = generalized_s = 0.0
            for u, v in zip(path, path[1:]):
                data = graph[u][v]
                length += float(data["length_m"])
                shade_len += float(data.get("shade_length_m", 0.0))
                generalized_s += edge_weight(u, v, data, scenario)
            row = {
                "origin_node": node,
                "scenario": scenario,
                "status": "ok",
                "stop_node": path[-1],
                "length_m": length,
                "time_min": length / walk_speed_m_s / 60.0,
                "shade_length_m": shade_len,
                "shade_fraction": shade_len / length if length else 0.0,
                "generalized_time_min": generalized_s / 60.0,
                "path": path,
            }
            rows.append(row)
            routes.append({k: row[k] for k in ["origin_node", "scenario", "length_m", "time_min", "shade_length_m", "shade_fraction", "generalized_time_min", "stop_node"]} | {"geometry": _route_geometry(graph, path)})
    return pd.DataFrame(rows), pd.DataFrame(routes)


def _routes_gdf_from_rows(rows: list[dict], crs: CRS) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


def route_city(city: CitySpec) -> dict[str, object]:
    raw_dir, out_dir = _city_dirs(city)
    polygon, crs = _study_polygon(city)
    buildings, stops, graph, nodes = _fetch_osm_buildings_stops_graph(city, polygon, crs)
    trees = load_trees(city, polygon, crs)
    canopies = _tree_canopies(trees)
    canopy_union = unary_union(canopies.geometry.tolist()) if not canopies.empty else None
    graph = _edge_shade_fraction(graph, canopy_union)

    origins = buildings[["building_id", "geometry"]].copy()
    origins["geometry"] = origins.geometry.representative_point()
    snapped_origins = _snap_points_to_nodes(origins, nodes, "building_id")
    snapped_stops = _snap_points_to_nodes(stops[["stop_id", "geometry"]].copy(), nodes, "stop_id")
    origin_nodes = sorted(snapped_origins["node_id"].unique().tolist())
    stop_nodes = sorted(snapped_stops["node_id"].unique().tolist())

    result_df, route_df_raw = _route_all(graph, origin_nodes, stop_nodes, city.walk_speed_m_s, city.shadow_weight_a)
    by_origin = snapped_origins.groupby("node_id")["building_id"].apply(list).to_dict()
    expanded_rows = []
    for row in result_df.to_dict("records"):
        for building_id in by_origin.get(row["origin_node"], []):
            expanded_rows.append({**row, "building_id": int(building_id)})
    results = pd.DataFrame(expanded_rows)
    routes = _routes_gdf_from_rows(route_df_raw.to_dict("records"), crs)

    tables_dir = out_dir / "tables"
    maps_dir = out_dir / "maps"
    tables_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    buildings.to_parquet(tables_dir / "buildings.parquet")
    stops.to_parquet(tables_dir / "stops.parquet")
    trees.to_parquet(tables_dir / "trees.parquet")
    canopies.to_parquet(tables_dir / "tree_canopies.parquet")
    results.drop(columns=["path"], errors="ignore").to_parquet(tables_dir / "building_routes.parquet")
    routes.to_parquet(tables_dir / "routes.parquet")

    render_city_maps(city, crs, buildings, stops, trees, canopies, graph, results, routes, maps_dir)

    baseline = results.loc[(results["scenario"] == "baseline") & (results["status"] == "ok")].copy()
    shade = results.loc[(results["scenario"] == "shade") & (results["status"] == "ok")].copy()
    compare = baseline.merge(shade, on="building_id", suffixes=("_baseline", "_shade"))
    compare["delta_generalized_time_min"] = compare["generalized_time_min_shade"] - compare["generalized_time_min_baseline"]
    compare["route_changed"] = compare["stop_node_baseline"].astype(str) != compare["stop_node_shade"].astype(str)
    summary = {
        "city": city.slug,
        "tree_source": city.tree_source,
        "buildings": int(len(buildings)),
        "stops": int(len(stops)),
        "trees": int(len(trees)),
        "median_baseline_min": float(compare["time_min_baseline"].median()) if not compare.empty else None,
        "median_shade_generalized_min": float(compare["generalized_time_min_shade"].median()) if not compare.empty else None,
        "median_delta_generalized_min": float(compare["delta_generalized_time_min"].median()) if not compare.empty else None,
        "route_changed_count": int(compare["route_changed"].sum()) if not compare.empty else 0,
        "coimbra_note": "official tree portal unavailable; OSM natural=tree fallback used" if city.slug == "coimbra_portugal" else None,
        "brno_note": "official tree layer lacks crown/height; default proxy radius=3m height=9m used" if city.slug == "brno_czechia" else None,
        "delft_note": "official point layer used; default proxy radius/height filled where missing" if city.slug == "delft_netherlands" else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def render_city_maps(city: CitySpec, crs: CRS, buildings: gpd.GeoDataFrame, stops: gpd.GeoDataFrame, trees: gpd.GeoDataFrame, canopies: gpd.GeoDataFrame, graph: nx.Graph, results: pd.DataFrame, routes: gpd.GeoDataFrame, maps_dir: Path) -> None:
    buildings_web = buildings.to_crs(3857)
    stops_web = stops.to_crs(3857)
    trees_web = trees.to_crs(3857)
    canopies_web = canopies.to_crs(3857)
    edge_rows = [{"u": u, "v": v, "shade_fraction": d.get("shade_fraction", 0.0), "geometry": d["geometry"]} for u, v, d in graph.edges(data=True)]
    edges_web = gpd.GeoDataFrame(edge_rows, geometry="geometry", crs=crs).to_crs(3857)

    fig, ax = plt.subplots(figsize=(10, 10))
    buildings_web.boundary.plot(ax=ax, color="#d0d0d0", linewidth=0.2)
    if not canopies_web.empty:
        canopies_web.plot(ax=ax, color="#4caf50", alpha=0.45, linewidth=0)
    stops_web.plot(ax=ax, color="#1565c0", markersize=10)
    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, attribution=False)
    except Exception:
        pass
    ax.legend(
        handles=[
            Patch(facecolor="#4caf50", edgecolor="none", alpha=0.45, label="Tree canopy proxy"),
            Line2D([0], [0], color="#d0d0d0", lw=2, label="Buildings"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#1565c0", markersize=7, label="PT stops"),
        ],
        loc="lower left",
        frameon=True,
        fontsize=9,
    )
    ax.set_axis_off()
    ax.set_title(f"{city.title}: buildings, stops, and tree-canopy proxy")
    fig.savefig(maps_dir / "01_inputs.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    baseline = results.loc[(results["scenario"] == "baseline") & (results["status"] == "ok")].copy()
    shade = results.loc[(results["scenario"] == "shade") & (results["status"] == "ok")].copy()
    if baseline.empty or shade.empty:
        return
    compare = baseline.merge(shade, on="building_id", suffixes=("_baseline", "_shade"))
    compare["delta_shade_fraction"] = compare["shade_fraction_shade"] - compare["shade_fraction_baseline"]
    sample_origin_nodes = compare.sort_values(["delta_shade_fraction", "generalized_time_min_shade"], ascending=[False, False]).head(20)["origin_node_baseline"].tolist()
    base_routes = routes.loc[(routes["scenario"] == "baseline") & (routes["origin_node"].isin(sample_origin_nodes))].to_crs(3857)
    shade_routes = routes.loc[(routes["scenario"] == "shade") & (routes["origin_node"].isin(sample_origin_nodes))].to_crs(3857)

    fig, ax = plt.subplots(figsize=(10, 10))
    edges_web.plot(ax=ax, column="shade_fraction", cmap="Greys", linewidth=1.0, alpha=0.7, legend=True, legend_kwds={"label": "Share of edge under tree-canopy proxy (0–1)"})
    base_routes.plot(ax=ax, color="#7f1734", linewidth=2.0, alpha=0.8)
    shade_routes.plot(ax=ax, color="#2f9e44", linewidth=2.0, alpha=0.85)
    stops_web.plot(ax=ax, color="#1565c0", markersize=14)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#7f1734", lw=2, label="Baseline route"),
            Line2D([0], [0], color="#2f9e44", lw=2, label="Shade-aware route"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#1565c0", markersize=7, label="PT stops"),
        ],
        loc="lower left",
        frameon=True,
        fontsize=9,
    )
    ax.set_axis_off()
    ax.set_title(f"{city.title}: route sample over tree shade proxy")
    fig.savefig(maps_dir / "02_routes_over_shade.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_summary_figure() -> Path:
    rows = []
    for city in CITIES:
        path = OUT_DIR / city.slug / "maps" / "02_routes_over_shade.png"
        if path.exists():
            rows.append((city.title, path))
    return OUT_DIR / "summary_ready.txt"


def run_all() -> None:
    summaries = [route_city(city) for city in CITIES]
    pd.DataFrame(summaries).to_csv(OUT_DIR / "summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", choices=[city.slug for city in CITIES] + ["all"], default="all")
    args = parser.parse_args()
    if args.city == "all":
        run_all()
        return
    city = next(city for city in CITIES if city.slug == args.city)
    route_city(city)


if __name__ == "__main__":
    main()
