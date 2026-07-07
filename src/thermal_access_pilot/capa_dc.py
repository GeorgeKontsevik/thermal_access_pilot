from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.transform import rowcol
from rasterio.warp import transform_geom
from shapely.geometry import LineString, Point, shape

from .routing import route_all_origins

IMAGE_SERVER = "https://gis.nnvl.noaa.gov/arcgis/rest/services/HINDZ/Afternoon_Air_Temperature_in_Cities/ImageServer"
WASHINGTON_RASTER_ID = 1
DC_BBOX = (-77.16, 38.78, -76.88, 39.02)  # west, south, east, north
DC_CENTER_LON = -77.04
DC_CENTER_LAT = 38.90
BUILDING_LIMIT = None


def dc_center_square_bbox(center_lon: float = DC_CENTER_LON, center_lat: float = DC_CENTER_LAT, side_m: float = 1500) -> tuple[float, float, float, float]:
    half = side_m / 2
    lat_delta = half / 111_320
    lon_delta = half / (111_320 * math.cos(math.radians(center_lat)))
    return (center_lon - lon_delta, center_lat - lat_delta, center_lon + lon_delta, center_lat + lat_delta)


def fahrenheit_to_celsius(value_f: float) -> float:
    return (float(value_f) - 32.0) * 5.0 / 9.0


def temperature_display_limits(values: np.ndarray) -> tuple[float, float]:
    valid = values[np.isfinite(values) & (values > 40) & (values < 130)]
    if len(valid) == 0:
        return 85.0, 102.0
    lo, hi = np.percentile(valid, [5, 95])
    return round(float(lo), 2), round(float(hi), 2)


def noaa_capa_export_url(bbox: tuple[float, float, float, float], size: int = 1200, response_format: str = "json") -> str:
    params = {
        "f": response_format,
        "bbox": ",".join(str(v) for v in bbox),
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": f"{size},{size}",
        "format": "tiff",
        "pixelType": "F32",
        "mosaicRule": json.dumps({"mosaicMethod": "esriMosaicLockRaster", "lockRasterIds": [WASHINGTON_RASTER_ID]}),
    }
    return f"{IMAGE_SERVER}/exportImage?{urllib.parse.urlencode(params)}"


def fetch_noaa_capa_raster(target: Path, bbox: tuple[float, float, float, float] = DC_BBOX) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target
    tmp = target.with_suffix(".tmp.tif")
    request = urllib.request.Request(noaa_capa_export_url(bbox, response_format="image"), headers={"User-Agent": "thermal-access-pilot/0.1"})
    with urllib.request.urlopen(request, timeout=90) as response, tmp.open("wb") as fh:
        fh.write(response.read())
    tmp.replace(target)
    return target


def fetch_noaa_metadata() -> dict:
    url = f"{IMAGE_SERVER}/query?{urllib.parse.urlencode({'f': 'json', 'where': 'objectid=1', 'outFields': '*', 'returnGeometry': 'false'})}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)["features"][0]["attributes"]


def _sample_raster_line_f(geometry_projected: LineString, src_crs: str, raster, transform, dst_crs) -> tuple[float, float]:
    geom_wgs = shape(transform_geom(src_crs, dst_crs, geometry_projected.__geo_interface__))
    distances = np.linspace(0, geom_wgs.length, max(2, int(geometry_projected.length // 25)))
    values = []
    for distance in distances:
        point = geom_wgs.interpolate(float(distance))
        row, col = rowcol(transform, point.x, point.y)
        if 0 <= row < raster.shape[0] and 0 <= col < raster.shape[1]:
            value = float(raster[row, col])
            if np.isfinite(value) and 40 < value < 130:
                values.append(value)
    if not values:
        return float("nan"), 0.0
    arr = np.asarray(values)
    return float(arr.mean()), float((arr >= 94.0).mean())


def _load_osm_inputs(bbox: tuple[float, float, float, float]):
    print("stage=osm_walk_graph")
    graph = ox.graph_from_bbox(bbox, network_type="walk", simplify=True, retain_all=False)
    graph = ox.project_graph(graph)
    nodes, edges = ox.graph_to_gdfs(graph)

    print("stage=osm_building_points")
    buildings = ox.features_from_bbox(bbox, {"building": True})
    buildings = buildings.loc[buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    buildings = buildings.to_crs(nodes.crs)
    buildings = buildings.reset_index(drop=True)
    buildings["building_id"] = buildings.index.astype(int)
    building_polygons = buildings.copy()
    # Buildings are intentionally only origin points here. No floors, population,
    # DSM, or height modelling: the NOAA/CAPA layer already supplies heat.
    buildings["geometry"] = buildings.geometry.representative_point()

    print(f"osm buildings={len(buildings)}")

    print("stage=osm_pt_stops")
    stops = ox.features_from_bbox(
        bbox,
        {"highway": "bus_stop", "public_transport": ["platform", "stop_position"], "railway": ["tram_stop", "station", "subway_entrance"]},
    )
    stops = stops.loc[stops.geometry.geom_type.isin(["Point", "Polygon", "MultiPolygon"])].copy()
    stops = stops.to_crs(nodes.crs)
    stops["geometry"] = stops.geometry.representative_point()
    print(f"osm stops={len(stops)}")
    return graph, nodes, edges, building_polygons, buildings, stops


def _simple_graph(projected_graph, nodes: gpd.GeoDataFrame, raster_path: Path) -> nx.Graph:
    _, edges = ox.graph_to_gdfs(projected_graph)
    out = nx.Graph()
    for node_id, row in nodes.iterrows():
        out.add_node(node_id, geometry=row.geometry)
    with rasterio.open(raster_path) as src:
        raster = src.read(1)
        transform = src.transform
        raster_crs = src.crs
    for (u, v, _), row in edges.iterrows():
        geom = row.geometry
        if geom is None:
            geom = LineString([nodes.loc[u].geometry, nodes.loc[v].geometry])
        length = float(row.get("length", geom.length))
        mean_f, hot_fraction = _sample_raster_line_f(geom, str(nodes.crs), raster, transform, raster_crs)
        if out.has_edge(u, v) and length >= out[u][v]["length_m"]:
            continue
        out.add_edge(
            u,
            v,
            length_m=length,
            geometry=geom,
            mean_temp_f=mean_f,
            hot_fraction=0.0 if np.isnan(mean_f) else hot_fraction,
            hot_length_m=0.0 if np.isnan(mean_f) else length * hot_fraction,
        )
    return out


def _nearest_node_ids(points: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame) -> pd.Series:
    node_geoms = nodes.geometry
    sindex = node_geoms.sindex
    matched = []
    for geom in points.geometry:
        nearest = sindex.nearest(geom, return_all=False)
        pos = int(np.asarray(nearest).reshape(-1)[-1])
        matched.append(node_geoms.index[pos])
    return pd.Series(matched, index=points.index)


def _line_for_path(graph: nx.Graph, path: list) -> LineString:
    coords = []
    for u, v in zip(path, path[1:]):
        segment = list(graph[u][v]["geometry"].coords)
        coords.extend(segment[1:] if coords and coords[-1] == segment[0] else segment)
    return LineString(coords) if len(coords) >= 2 else LineString()


def select_before_after_origin(results: pd.DataFrame) -> tuple[int, int]:
    base = results.loc[results["penalty"] == 0.0, ["building_id", "origin_node", "time_min"]].rename(columns={"time_min": "base_time_min"})
    heat = results.loc[results["penalty"] == 1.0, ["building_id", "generalized_time_min"]]
    merged = heat.merge(base, on="building_id")
    merged["delta"] = merged["generalized_time_min"] - merged["base_time_min"]
    row = merged.sort_values(["delta", "building_id"], ascending=[False, True]).iloc[0]
    return int(row["building_id"]), int(row["origin_node"])


def run_capa_dc(output_dir: Path = Path("outputs/capa_dc_center_1500m"), bbox: tuple[float, float, float, float] | None = None) -> None:
    bbox = bbox or dc_center_square_bbox()
    output_dir.mkdir(parents=True, exist_ok=True)
    print("stage=noaa_capa_raster")
    raster_path = fetch_noaa_capa_raster(output_dir / "inputs/noaa_capa_washington_afternoon_20180828_f.tif", bbox)
    metadata = fetch_noaa_metadata()

    graph, nodes, _, building_polygons, buildings, stops = _load_osm_inputs(bbox)
    print("stage=edge_heat_sampling")
    simple = _simple_graph(graph, nodes, raster_path)
    nodes = nodes.loc[nodes.index.isin(simple.nodes)].copy()

    if BUILDING_LIMIT is not None and len(buildings) > BUILDING_LIMIT:
        buildings = buildings.sample(BUILDING_LIMIT, random_state=42).sort_index()
    building_nodes = _nearest_node_ids(buildings, nodes).dropna().astype(int)
    stop_nodes = _nearest_node_ids(stops, nodes).dropna().astype(int).unique().tolist()
    origins = sorted(set(building_nodes.tolist()) & set(simple.nodes))
    stop_nodes = sorted(set(stop_nodes) & set(simple.nodes))
    print(f"snapped building_nodes={len(origins)} stop_nodes={len(stop_nodes)}")

    scenarios = [0.0, 0.5, 1.0]
    result_rows = []
    route_rows = []
    building_by_node = pd.DataFrame({"building_id": buildings.loc[building_nodes.index, "building_id"].values, "node": building_nodes.values}).groupby("node")["building_id"].apply(list).to_dict()
    for penalty in scenarios:
        print(f"stage=routing penalty={penalty}")
        routed = route_all_origins(simple, origins, stop_nodes, penalty=penalty, walk_speed_m_s=1.4)
        for node, route in routed.items():
            for building_id in building_by_node.get(node, []):
                result_rows.append({"building_id": int(building_id), "origin_node": int(node), "penalty": penalty, **route})
            if route.get("status") == "ok":
                route_rows.append({"origin_node": int(node), "penalty": penalty, **{k: route[k] for k in ["length_m", "time_min", "hot_length_m", "hot_fraction", "generalized_time_min", "stop_node"]}, "geometry": _line_for_path(simple, route["path"])})

    results = pd.DataFrame(result_rows)
    routes = gpd.GeoDataFrame(route_rows, crs=nodes.crs)
    (output_dir / "tables").mkdir(exist_ok=True)
    (output_dir / "routes").mkdir(exist_ok=True)
    results.drop(columns=["path"], errors="ignore").to_parquet(output_dir / "tables/building_results.parquet")
    routes.to_parquet(output_dir / "routes/routes.parquet")
    building_polygons.to_parquet(output_dir / "inputs/building_polygons.parquet")
    buildings.to_parquet(output_dir / "inputs/building_points.parquet")
    stops.to_parquet(output_dir / "inputs/stops.parquet")
    walk_edges = ox.graph_to_gdfs(graph, nodes=False).reset_index()[["u", "v", "length", "geometry"]]
    walk_edges.to_parquet(output_dir / "inputs/walk_edges.parquet")

    print("stage=maps")
    _render_maps(output_dir, raster_path, nodes.crs, building_polygons, buildings, stops, routes, results, output_dir / "inputs/walk_edges.parquet")

    summary = {
        "source": "NOAA/CAPA Heat Watch Afternoon Air Temperature in Cities",
        "image_server": IMAGE_SERVER,
        "raster_id": WASHINGTON_RASTER_ID,
        "campaign_name": metadata.get("name"),
        "city_name": metadata.get("city_name"),
        "units": metadata.get("units") or metadata.get("units_abbreviation") or "F",
        "threshold_f": 94.0,
        "threshold_c": fahrenheit_to_celsius(94.0),
        "bbox": list(bbox),
        "approx_side_m": 1500,
        "building_points": int(len(buildings)),
        "stops": int(len(stop_nodes)),
        "route_rows": int(len(results)),
        "ok_route_rows": int((results["status"] == "ok").sum()),
        "baseline_mean_time_min": float(results.loc[results["penalty"] == 0, "time_min"].mean()),
        "baseline_mean_hot_fraction": float(results.loc[results["penalty"] == 0, "hot_fraction"].mean()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("stage=done")


def _render_maps(output_dir: Path, raster_path: Path, crs, building_polygons, building_points, stops, routes, results, walk_edges_path: Path | None = None) -> None:
    maps = output_dir / "maps"
    maps.mkdir(exist_ok=True)
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    def raster_bg(ax):
        with rasterio.open(raster_path) as src:
            arr = src.read(1)
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        vmin, vmax = temperature_display_limits(arr)
        shown = np.where(np.isfinite(arr) & (arr > 40) & (arr < 130), arr, np.nan)
        image = ax.imshow(shown, extent=extent, origin="upper", cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
        return image

    fig, ax = plt.subplots(figsize=(8, 8))
    image = raster_bg(ax)
    if walk_edges_path and walk_edges_path.exists():
        gpd.read_parquet(walk_edges_path).to_crs("EPSG:4326").plot(ax=ax, color="#333333", linewidth=0.35, alpha=0.45)
    building_polygons.to_crs("EPSG:4326").plot(ax=ax, facecolor="#222222", edgecolor="none", alpha=0.28)
    stops.to_crs("EPSG:4326").plot(ax=ax, color="#0047ff", markersize=12, alpha=0.9)
    ax.set_title("NOAA/CAPA Washington DC afternoon air temperature, Aug 28 2018 (°F)")
    ax.set_axis_off()
    fig.colorbar(image, ax=ax, shrink=0.7, label="°F, local 5–95% stretch")
    fig.savefig(maps / "01_noaa_capa_heat_map.png", dpi=180)
    plt.close(fig)

    routes_wgs = routes.to_crs("EPSG:4326")
    stops_wgs = stops.to_crs("EPSG:4326")
    fig, ax = plt.subplots(figsize=(8, 8))
    image = raster_bg(ax)
    if walk_edges_path and walk_edges_path.exists():
        gpd.read_parquet(walk_edges_path).to_crs("EPSG:4326").plot(ax=ax, color="#333333", linewidth=0.35, alpha=0.35)
    routes_wgs.loc[routes_wgs["penalty"] == 0].head(60).plot(ax=ax, color="white", linewidth=0.7, alpha=0.7)
    routes_wgs.loc[routes_wgs["penalty"] == 1.0].head(60).plot(ax=ax, color="cyan", linewidth=0.7, alpha=0.8)
    stops_wgs.head(200).plot(ax=ax, color="blue", markersize=4)
    ax.set_title("Building-to-stop routes: baseline white, heat-aware cyan")
    ax.set_axis_off()
    fig.colorbar(image, ax=ax, shrink=0.7, label="°F, local 5–95% stretch")
    fig.savefig(maps / "02_routes_on_capa_heat.png", dpi=180)
    plt.close(fig)

    base = results.loc[results["penalty"] == 0, ["building_id", "time_min"]].rename(columns={"time_min": "base_time_min"})
    heat = results.loc[results["penalty"] == 1.0, ["building_id", "generalized_time_min"]].merge(base, on="building_id")
    heat["delta_min"] = heat["generalized_time_min"] - heat["base_time_min"]
    b = building_points.merge(heat[["building_id", "delta_min"]], on="building_id", how="left").to_crs("EPSG:4326")
    fig, ax = plt.subplots(figsize=(8, 8))
    image = raster_bg(ax)
    b.plot(ax=ax, column="delta_min", cmap="viridis", markersize=4, legend=True)
    ax.set_title("Generalized walking-time increase from CAPA hot exposure")
    ax.set_axis_off()
    fig.colorbar(image, ax=ax, shrink=0.7, label="°F, local 5–95% stretch")
    fig.savefig(maps / "03_building_time_change.png", dpi=180)
    plt.close(fig)

    building_id, origin_node = select_before_after_origin(results)
    selected_building = building_polygons.loc[building_polygons["building_id"] == building_id].to_crs("EPSG:4326")
    selected_routes = routes.loc[(routes["origin_node"] == origin_node) & (routes["penalty"].isin([0.0, 1.0]))].to_crs("EPSG:4326")
    baseline_route = selected_routes.loc[selected_routes["penalty"] == 0.0]
    heat_route = selected_routes.loc[selected_routes["penalty"] == 1.0]
    stop_ids = selected_routes["stop_node"].dropna().astype(int).unique().tolist()
    stop_points = stops_wgs.copy()
    fig, ax = plt.subplots(figsize=(8, 8))
    image = raster_bg(ax)
    if walk_edges_path and walk_edges_path.exists():
        gpd.read_parquet(walk_edges_path).to_crs("EPSG:4326").plot(ax=ax, color="#333333", linewidth=0.35, alpha=0.35)
    building_polygons.to_crs("EPSG:4326").plot(ax=ax, facecolor="#111111", edgecolor="none", alpha=0.18)
    stops_wgs.plot(ax=ax, color="#0047ff", markersize=10, alpha=0.7)
    selected_building.plot(ax=ax, facecolor="#ffd400", edgecolor="black", linewidth=1.2, alpha=0.95)
    baseline_route.plot(ax=ax, color="white", linewidth=4.0, alpha=0.95)
    baseline_route.plot(ax=ax, color="black", linewidth=1.4, alpha=0.95)
    heat_route.plot(ax=ax, color="cyan", linewidth=2.8, alpha=0.95)
    ax.set_title(f"Before/after route for building {building_id}: baseline black/white, heat-aware cyan")
    ax.set_axis_off()
    fig.colorbar(image, ax=ax, shrink=0.7, label="°F, local 5–95% stretch")
    fig.savefig(maps / "04_before_after_route.png", dpi=180)
    plt.close(fig)

    for path in maps.glob("*.png"):
        img = Image.open(path)
        if img.size[0] < 1000 or img.size[1] < 700:
            raise ValueError(f"Map too small: {path} {img.size}")


def main() -> None:
    run_capa_dc(Path("outputs/capa_dc_center_1500m"))


if __name__ == "__main__":
    main()
