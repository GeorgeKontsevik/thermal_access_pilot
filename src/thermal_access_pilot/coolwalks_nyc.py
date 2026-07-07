from __future__ import annotations

import argparse
import json
import math
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import mapclassify
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import py7zr
import contextily as ctx
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path("/Users/gk/Code/super-duper-disser/thermal_access_pilot")
RAW_DIR = ROOT / "cache/nyc_coolwalks/raw"
JULIA_BIN = Path("/opt/homebrew/bin/julia")
JULIA_ENV = ROOT / "cache/nyc_coolwalks/julia_env"
JULIA_SCRIPT = Path(__file__).with_name("coolwalks_nyc_shadow_graph.jl")


@dataclass(frozen=True)
class StudyArea:
    name: str
    center_lon: float
    center_lat: float
    square_size_m: float
    walk_speed_m_s: float = 1.2
    shadow_weight_a: float = 1.35


UWS_1500M = StudyArea(
    name="nyc_uws_1500m",
    center_lon=-73.9795,
    center_lat=40.7815,
    square_size_m=1500.0,
)

SOHO_1500M = StudyArea(
    name="nyc_soho_1500m",
    center_lon=-74.0007,
    center_lat=40.7240,
    square_size_m=1500.0,
)


def area_work_dir(area: StudyArea) -> Path:
    return ROOT / "cache/nyc_coolwalks" / area.name


def area_out_dir(area: StudyArea) -> Path:
    return ROOT / "outputs" / area.name


def _norm(text: object) -> str:
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def _json_get(url: str) -> object:
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def _download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    urllib.request.urlretrieve(url, path)
    return path


def _classify_shadow_length_fisher_jenks(values: list[float], k: int = 5) -> tuple[list[str], list[str]]:
    series = pd.Series(values, dtype="float64").fillna(0).clip(lower=0)
    positive = series[series > 0]
    labels = pd.Series("0 m", index=series.index, dtype="object")
    class_labels = ["0 m"]
    if positive.empty:
        return labels.tolist(), class_labels
    fj = mapclassify.FisherJenks(positive.to_numpy(), k=min(k, positive.nunique()))
    lower = 0.0
    for upper in fj.bins:
        mask = (series > lower) & (series <= float(upper))
        label = f"{int(lower)}–{int(upper)} m"
        labels.loc[mask] = label
        class_labels.append(label)
        lower = float(upper)
    return labels.tolist(), class_labels


def _extract_7z(archive_path: Path, target_dir: Path) -> Path:
    marker = target_dir / ".done"
    if marker.exists():
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        archive.extractall(path=target_dir)
    marker.write_text("ok\n")
    return target_dir


def square_bbox(area: StudyArea) -> tuple[float, float, float, float]:
    to_metric = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)
    to_wgs84 = Transformer.from_crs("EPSG:32618", "EPSG:4326", always_xy=True)
    x, y = to_metric.transform(area.center_lon, area.center_lat)
    half = area.square_size_m / 2.0
    min_lon, min_lat = to_wgs84.transform(x - half, y - half)
    max_lon, max_lat = to_wgs84.transform(x + half, y + half)
    return min_lon, min_lat, max_lon, max_lat


def _find_first(paths: list[Path], pattern: str) -> Path:
    for path in paths:
        if pattern in path.name.lower():
            return path
    raise FileNotFoundError(pattern)


def fetch_buildings(area: StudyArea, force: bool = False) -> gpd.GeoDataFrame:
    path = area_work_dir(area) / "buildings.geojson"
    path.parent.mkdir(parents=True, exist_ok=True)
    if force and path.exists():
        path.unlink()
    if not path.exists():
        min_lon, min_lat, max_lon, max_lat = square_bbox(area)
        where = f"within_box(the_geom,{max_lat},{min_lon},{min_lat},{max_lon})"
        url = (
            "https://data.cityofnewyork.us/resource/5zhs-2jue.geojson?"
            "$select=the_geom,doitt_id,height_roof&$limit=50000&$where="
            + urllib.parse.quote(where)
        )
        path.write_text(json.dumps(_json_get(url)))
    gdf = gpd.read_file(path).to_crs(4326)
    gdf["building_id"] = gdf["doitt_id"].astype(str)
    gdf["height_m"] = gdf["height_roof"].astype(float) * 0.3048
    gdf["heightroof"] = gdf["height_roof"]
    return gdf.loc[gdf["height_m"] > 0].reset_index(drop=True)


def fetch_subway_entrances(area: StudyArea, force: bool = False) -> gpd.GeoDataFrame:
    path = area_work_dir(area) / "subway_entrances.geojson"
    path.parent.mkdir(parents=True, exist_ok=True)
    if force and path.exists():
        path.unlink()
    if not path.exists():
        min_lon, min_lat, max_lon, max_lat = square_bbox(area)
        where = (
            f"entrance_latitude between {min_lat} and {max_lat} "
            f"AND entrance_longitude between {min_lon} and {max_lon}"
        )
        url = (
            "https://data.ny.gov/resource/i9wp-a4ja.json?"
            "$select=stop_name,station_id,complex_id,daytime_routes,entrance_type,entry_allowed,exit_allowed,"
            "entrance_latitude,entrance_longitude&$limit=50000&$where="
            + urllib.parse.quote(where)
        )
        rows = _json_get(url)
        df = pd.DataFrame(rows)
        df["geometry"] = gpd.points_from_xy(df["entrance_longitude"].astype(float), df["entrance_latitude"].astype(float), crs=4326)
        gpd.GeoDataFrame(df, geometry="geometry", crs=4326).to_file(path, driver="GeoJSON")
    return gpd.read_file(path).to_crs(4326)


def _tree_attr_columns(columns: list[str]) -> tuple[str | None, str | None, str | None, str | None]:
    by_norm = {_norm(col): col for col in columns}
    id_col = by_norm.get("fid") or by_norm.get("treeid") or by_norm.get("id")
    reg_col = by_norm.get("reg") or by_norm.get("region") or by_norm.get("borough")
    height_col = None
    area_col = None
    for col in columns:
        norm = _norm(col)
        if height_col is None and ("hmax" in norm or norm == "heightm" or norm == "value"):
            height_col = col
        if area_col is None and "area" in norm:
            area_col = col
    return id_col, reg_col, height_col, area_col


def _load_tree_attrs_csv(csv_path: Path) -> pd.DataFrame:
    with csv_path.open("r", encoding="utf-8", errors="ignore") as handle:
        header = handle.readline().strip().split(",")
    id_col, reg_col, height_col, area_col = _tree_attr_columns(header)
    if not all([id_col, reg_col, height_col, area_col]):
        raise ValueError(f"Could not resolve tree attribute columns in {csv_path.name}: {header[:20]}")
    usecols = [id_col, reg_col, height_col, area_col]
    attrs = pd.read_csv(csv_path, usecols=usecols)
    attrs = attrs.rename(columns={id_col: "fid", reg_col: "reg", height_col: "height_m", area_col: "crown_area_m2"})
    attrs["fid"] = attrs["fid"].astype(str)
    attrs["reg"] = attrs["reg"].astype(str)
    attrs["height_m"] = pd.to_numeric(attrs["height_m"], errors="coerce")
    attrs["crown_area_m2"] = pd.to_numeric(attrs["crown_area_m2"], errors="coerce")
    return attrs


def prepare_trees(area: StudyArea, force: bool = False) -> gpd.GeoDataFrame:
    archive_path = _download("https://ndownloader.figshare.com/files/36735453", RAW_DIR / "TreePoint.7z")
    extracted_dir = _extract_7z(archive_path, RAW_DIR / "TreePoint")
    shapefiles = sorted(extracted_dir.rglob("*.shp"))
    if not shapefiles:
        raise FileNotFoundError(f"No shapefiles found in {extracted_dir}")
    shp_path = _find_first(shapefiles, "manhp")
    gdf = gpd.read_file(shp_path).to_crs(4326)

    min_lon, min_lat, max_lon, max_lat = square_bbox(area)
    gdf = gdf.cx[min_lon:max_lon, min_lat:max_lat].copy()
    if gdf.empty:
        raise ValueError("No tree points in study bbox")

    id_col, reg_col, height_col, area_col = _tree_attr_columns(list(gdf.columns))
    if not height_col or not area_col:
        attrs_csv = _download("https://ndownloader.figshare.com/files/38666738", RAW_DIR / "NYTrees_FID_Region_Att.csv")
        attrs = _load_tree_attrs_csv(attrs_csv)
        if id_col is None:
            raise ValueError(f"Could not resolve tree id column in {shp_path.name}: {list(gdf.columns)}")
        local_reg = "Manhattan"
        if reg_col is not None:
            gdf["reg"] = gdf[reg_col].astype(str)
        else:
            gdf["reg"] = local_reg
        gdf["fid"] = gdf[id_col].astype(str)
        attrs = attrs.loc[attrs["reg"].str.contains(local_reg, case=False, na=False)].copy()
        gdf = gdf.merge(attrs, on=["fid", "reg"], how="left")
    else:
        gdf = gdf.rename(columns={id_col or "FID": "fid", height_col: "height_m", area_col: "crown_area_m2"})
        if "reg" not in gdf.columns:
            gdf["reg"] = "Manhattan"

    gdf["height_m"] = pd.to_numeric(gdf["height_m"], errors="coerce")
    gdf["crown_area_m2"] = pd.to_numeric(gdf["crown_area_m2"], errors="coerce")
    gdf = gdf.dropna(subset=["height_m", "crown_area_m2"]).copy()
    gdf = gdf.loc[(gdf["height_m"] > 0) & (gdf["crown_area_m2"] > 0)].copy()
    gdf["radius_m"] = (gdf["crown_area_m2"] / math.pi).pow(0.5)
    gdf["tree_id"] = gdf["fid"].astype(str)
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
    out = gdf[["tree_id", "height_m", "radius_m", "lon", "lat", "geometry"]].copy()
    out.to_file(area_work_dir(area) / "trees.geojson", driver="GeoJSON")
    return out


def prepare_origins(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    origins = buildings[["building_id", "height_m", "geometry"]].copy()
    origins["geometry"] = origins.geometry.representative_point()
    return origins


def ensure_julia_env() -> None:
    if not JULIA_BIN.exists():
        raise FileNotFoundError(f"Julia not found at {JULIA_BIN}")
    JULIA_ENV.mkdir(parents=True, exist_ok=True)
    sentinel = JULIA_ENV / ".ready"
    if sentinel.exists():
        return
    bootstrap = f"""
import Pkg
Pkg.activate(raw\"{JULIA_ENV}\")
Pkg.add(["JSON", "TimeZones", "DataFrames", "GeoDataFrames", "Extents", "ArchGDAL", "Graphs", "MetaGraphs"])
Pkg.develop(path=raw\"/tmp/CoolWalksUtils.jl\")
Pkg.develop(path=raw\"/tmp/Folium.jl\")
Pkg.develop(path=raw\"/tmp/ShadowGraphs.jl\")
Pkg.develop(path=raw\"/tmp/CompositeBuildings.jl\")
Pkg.develop(path=raw\"/tmp/TreeLoaders.jl\")
Pkg.develop(path=raw\"/tmp/MinistryOfCoolWalks.jl\")
Pkg.instantiate()
"""
    subprocess.run([str(JULIA_BIN), "-e", bootstrap], check=True)
    sentinel.write_text("ok\n")


def build_shadow_graph(area: StudyArea, timestamp_local: str, force: bool = False) -> None:
    prefix = area_work_dir(area) / "shadow_graph"
    edges_csv = prefix.with_name(prefix.name + "_edges.csv")
    if not force and edges_csv.exists():
        return
    ensure_julia_env()
    min_lon, min_lat, max_lon, max_lat = square_bbox(area)
    config = {
        "buildings_path": str(area_work_dir(area) / "buildings.geojson"),
        "trees_path": str(area_work_dir(area) / "trees.geojson"),
        "output_prefix": str(prefix),
        "timestamp_local": timestamp_local,
        "min_lon": min_lon,
        "min_lat": min_lat,
        "max_lon": max_lon,
        "max_lat": max_lat,
    }
    config_path = area_work_dir(area) / "shadow_graph_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    subprocess.run(
        [str(JULIA_BIN), f"--project={JULIA_ENV}", str(JULIA_SCRIPT), str(config_path)],
        check=True,
    )


def load_shadow_graph(area: StudyArea) -> tuple[nx.DiGraph, gpd.GeoDataFrame]:
    work_dir = area_work_dir(area)
    edge_df = pd.read_csv(work_dir / "shadow_graph_edges.csv")
    node_df = pd.read_csv(work_dir / "shadow_graph_nodes.csv")
    if {"sg_lon", "sg_lat"}.issubset(node_df.columns):
        nodes = gpd.GeoDataFrame(
            node_df,
            geometry=gpd.points_from_xy(node_df["sg_lon"], node_df["sg_lat"], crs=4326),
        )
    else:
        node_geom_col = next(col for col in node_df.columns if _norm(col) in {"sgpointgeometry", "pointgeom", "geometry"} or "point" in _norm(col))
        nodes = gpd.GeoDataFrame(
            node_df,
            geometry=gpd.GeoSeries.from_wkt(node_df[node_geom_col]),
            crs=4326,
        )
    nodes["node_id"] = node_df["vertex_id"] if "vertex_id" in node_df.columns else node_df.index.astype(int) + 1

    geom_candidates = [col for col in edge_df.columns if "geometry" in _norm(col)]
    geom_col = "sg_street_geometry" if "sg_street_geometry" in edge_df.columns else geom_candidates[0]
    edge_df["geometry"] = gpd.GeoSeries.from_wkt(edge_df[geom_col], on_invalid="ignore")
    graph = nx.Graph()
    for row in nodes.itertuples(index=False):
        graph.add_node(int(row.node_id))
    for row in edge_df.itertuples(index=False):
        if bool(getattr(row, "sg_helper", False)):
            continue
        u = int(getattr(row, "src_id"))
        v = int(getattr(row, "dst_id"))
        length_value = getattr(row, "sg_street_length", None)
        if length_value is None or pd.isna(length_value):
            length_value = getattr(row, "sg_length_m", None)
        if length_value is None or pd.isna(length_value):
            length_value = getattr(row, "length", None)
        if length_value is None or pd.isna(length_value):
            if row.geometry is None:
                continue
            length_value = row.geometry.length
        if row.geometry is None:
            continue
        length_m = float(length_value)
        shadow_length = float(getattr(row, "sg_shadow_length", 0.0) or 0.0)
        shadow_length = max(0.0, min(shadow_length, length_m))
        edge_data = {
            "length_m": length_m,
            "shadow_length_m": shadow_length,
            "sun_length_m": max(0.0, length_m - shadow_length),
            "geometry": row.geometry,
        }
        if graph.has_edge(u, v):
            if length_m < float(graph[u][v]["length_m"]):
                graph[u][v].update(edge_data)
        else:
            graph.add_edge(u, v, **edge_data)
    return graph, nodes[["node_id", "geometry"]]


def load_shadow_edges(area: StudyArea) -> gpd.GeoDataFrame:
    work_dir = area_work_dir(area)
    edge_df = pd.read_csv(work_dir / "shadow_graph_edges.csv")
    edge_df = edge_df.loc[~edge_df["sg_helper"].fillna(False)].copy()
    edge_df = edge_df.dropna(subset=["sg_street_geometry", "sg_street_length"]).copy()
    edge_df["geometry"] = gpd.GeoSeries.from_wkt(edge_df["sg_street_geometry"])
    edge_df["sg_street_length"] = pd.to_numeric(edge_df["sg_street_length"], errors="coerce")
    edge_df["sg_shadow_length"] = pd.to_numeric(edge_df["sg_shadow_length"], errors="coerce").fillna(0.0)
    edge_df = edge_df.loc[edge_df["sg_street_length"] > 0].copy()
    edge_df["shadow_fraction"] = (edge_df["sg_shadow_length"] / edge_df["sg_street_length"]).clip(0, 1)
    return gpd.GeoDataFrame(edge_df, geometry="geometry", crs=4326)


def _snap_points(points: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, id_col: str) -> gpd.GeoDataFrame:
    metric_crs = "EPSG:32618"
    points_metric = points[[id_col, "geometry"]].to_crs(metric_crs)
    nodes_metric = nodes.to_crs(metric_crs)
    snapped = gpd.sjoin_nearest(points_metric, nodes_metric, how="left", distance_col="snap_distance_m")
    snapped = snapped.to_crs(points.crs)
    snapped = snapped.drop(columns=["index_right"]).dropna(subset=["node_id"]).copy()
    snapped["node_id"] = snapped["node_id"].astype(int)
    return snapped


def _cost(data: dict, a: float) -> float:
    return float(data["shadow_length_m"] + a * data["sun_length_m"])


def _route_geometry(graph: nx.DiGraph, path: list[int]) -> LineString:
    lines = [graph[u][v]["geometry"] for u, v in zip(path, path[1:]) if graph.has_edge(u, v)]
    merged = unary_union(lines)
    if isinstance(merged, LineString):
        return merged
    if hasattr(merged, "geoms"):
        coords: list[tuple[float, float]] = []
        for geom in merged.geoms:
            coords.extend(list(geom.coords))
        return LineString(coords)
    raise TypeError(type(merged))


def route_all_buildings(
    graph: nx.DiGraph,
    buildings: gpd.GeoDataFrame,
    stops: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    area: StudyArea,
) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    origin_nodes = _snap_points(buildings[["building_id", "geometry"]].assign(geometry=buildings.geometry.representative_point()), nodes, "building_id")
    stop_nodes = _snap_points(stops[["station_id", "geometry"]].copy(), nodes, "station_id")
    stop_node_ids = stop_nodes["node_id"].drop_duplicates().tolist()
    if not stop_node_ids:
        raise ValueError("No snapped stops")

    summaries: list[dict] = []
    route_rows: list[dict] = []
    for label, a in [("baseline", 1.0), ("coolwalks", area.shadow_weight_a)]:
        distances, paths = nx.multi_source_dijkstra(
            graph,
            stop_node_ids,
            weight=lambda u, v, data: _cost(data, a),
        )
        for row in origin_nodes.itertuples(index=False):
            building_id = str(row.building_id)
            origin = int(row.node_id)
            if origin not in paths:
                summaries.append({"scenario": label, "building_id": building_id, "status": "unreachable"})
                continue
            path = list(reversed(paths[origin]))
            geom = _route_geometry(graph, path)
            total_len = 0.0
            shadow_len = 0.0
            felt_len = 0.0
            for u, v in zip(path, path[1:]):
                data = graph[u][v]
                total_len += float(data["length_m"])
                shadow_len += float(data["shadow_length_m"])
                felt_len += _cost(data, a)
            summaries.append(
                {
                    "scenario": label,
                    "building_id": building_id,
                    "status": "ok",
                    "origin_node": origin,
                    "stop_node": path[-1],
                    "length_m": total_len,
                    "shadow_length_m": shadow_len,
                    "shadow_fraction": shadow_len / total_len if total_len else 0.0,
                    "walk_time_min": total_len / area.walk_speed_m_s / 60.0,
                    "generalized_time_min": felt_len / area.walk_speed_m_s / 60.0,
                    "snap_distance_m": float(row.snap_distance_m),
                }
            )
            route_rows.append({"scenario": label, "building_id": building_id, "geometry": geom})
    summary_df = pd.DataFrame(summaries)
    routes_gdf = gpd.GeoDataFrame(route_rows, geometry="geometry", crs=4326)
    return summary_df, routes_gdf


def deduplicate_results(summary_df: pd.DataFrame, routes_gdf: gpd.GeoDataFrame) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    ok = summary_df.loc[summary_df["status"] == "ok"].copy()
    if ok.empty:
        return summary_df, routes_gdf
    best_idx = ok.groupby(["scenario", "building_id"])["generalized_time_min"].idxmin()
    best = ok.loc[best_idx].copy()
    best = best.sort_values(["scenario", "building_id"]).reset_index(drop=True)
    route_keys = best[["scenario", "building_id"]].drop_duplicates()
    routes = routes_gdf.merge(route_keys, on=["scenario", "building_id"], how="inner")
    routes = routes.drop_duplicates(subset=["scenario", "building_id"]).reset_index(drop=True)
    return best, routes


def render_maps(area: StudyArea, buildings: gpd.GeoDataFrame, trees: gpd.GeoDataFrame, stops: gpd.GeoDataFrame, summary_df: pd.DataFrame, routes_gdf: gpd.GeoDataFrame) -> None:
    maps_dir = area_out_dir(area) / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    shadow_edges = load_shadow_edges(area)
    buildings_web = buildings.to_crs(3857)
    trees_web = trees.to_crs(3857)
    stops_web = stops.to_crs(3857)
    shadow_web = shadow_edges.to_crs(3857)

    baseline = summary_df.loc[summary_df["scenario"] == "baseline"].copy()
    cool = summary_df.loc[summary_df["scenario"] == "coolwalks"].copy()
    compare = baseline.merge(
        cool,
        on="building_id",
        suffixes=("_baseline", "_cool"),
    )
    compare["delta_generalized_time_min"] = compare["generalized_time_min_cool"] - compare["generalized_time_min_baseline"]
    compare["delta_shadow_fraction"] = compare["shadow_fraction_cool"] - compare["shadow_fraction_baseline"]
    compare["route_changed"] = (
        compare["stop_node_baseline"].astype(str) != compare["stop_node_cool"].astype(str)
    ) | (compare["delta_shadow_fraction"].abs() > 1e-6)

    base_buildings = buildings.merge(
        compare[["building_id", "generalized_time_min_baseline", "shadow_fraction_baseline"]],
        on="building_id",
        how="left",
    )
    cool_buildings = buildings.merge(
        compare[["building_id", "generalized_time_min_cool", "shadow_fraction_cool"]],
        on="building_id",
        how="left",
    )
    diff_buildings = buildings.merge(
        compare[["building_id", "delta_generalized_time_min", "delta_shadow_fraction", "route_changed"]],
        on="building_id",
        how="left",
    )

    def add_common_legend(ax, include_routes: bool = False) -> None:
        handles: list = [
            Patch(facecolor="#d9d9d9", edgecolor="black", label="Buildings"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2f7d32", markersize=6, label="Trees"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#0b57d0", markersize=8, label="Subway entrances"),
        ]
        if include_routes:
            handles.extend(
                [
                    Line2D([0], [0], color="#888888", lw=2, label="Baseline route"),
                    Line2D([0], [0], color="#2f9e44", lw=2, label="Shade-aware route"),
                ]
            )
        ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=9)

    fig, ax = plt.subplots(figsize=(10, 10))
    base_buildings.plot(
        ax=ax,
        column="generalized_time_min_baseline",
        cmap="viridis",
        linewidth=0.1,
        edgecolor="none",
        legend=True,
        legend_kwds={"label": "Generalized walk time to nearest subway entrance (min)"},
    )
    trees.plot(ax=ax, color="#2f7d32", markersize=4, alpha=0.5)
    stops.plot(ax=ax, color="#0b57d0", markersize=18)
    add_common_legend(ax)
    ax.set_axis_off()
    ax.set_title("Baseline walk generalized time to nearest subway entrance")
    fig.savefig(maps_dir / "01_baseline_buildings.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 10))
    cool_buildings.plot(
        ax=ax,
        column="generalized_time_min_cool",
        cmap="plasma",
        linewidth=0.1,
        edgecolor="none",
        legend=True,
        legend_kwds={"label": "Shade-aware generalized walk time (min)"},
    )
    trees.plot(ax=ax, color="#2f7d32", markersize=4, alpha=0.5)
    stops.plot(ax=ax, color="#0b57d0", markersize=18)
    add_common_legend(ax)
    ax.set_axis_off()
    ax.set_title(f"Shade-aware generalized time (a={area.shadow_weight_a:g})")
    fig.savefig(maps_dir / "02_coolwalks_buildings.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 10))
    diff_buildings.plot(
        ax=ax,
        column="delta_generalized_time_min",
        cmap="coolwarm",
        linewidth=0.1,
        edgecolor="none",
        legend=True,
        legend_kwds={"label": "Change in generalized walk time (coolwalks - baseline, min)"},
    )
    changed = diff_buildings.loc[diff_buildings["route_changed"] == True]
    if not changed.empty:
        changed.boundary.plot(ax=ax, color="black", linewidth=0.3)
    trees.plot(ax=ax, color="#2f7d32", markersize=3, alpha=0.35)
    stops.plot(ax=ax, color="#0b57d0", markersize=14)
    handles = [
        Patch(facecolor="#d9d9d9", edgecolor="black", label="Buildings"),
        Line2D([0], [0], color="black", lw=1, label="Changed-route building"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2f7d32", markersize=6, label="Trees"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#0b57d0", markersize=8, label="Subway entrances"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=9)
    ax.set_axis_off()
    ax.set_title("Generalized-time change: coolwalks - baseline")
    fig.savefig(maps_dir / "03_difference_buildings.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    route_sample = (
        compare.sort_values(["route_changed", "delta_shadow_fraction"], ascending=[False, False])
        .head(25)["building_id"]
        .tolist()
    )
    base_routes = routes_gdf.loc[(routes_gdf["scenario"] == "baseline") & (routes_gdf["building_id"].isin(route_sample))]
    cool_routes = routes_gdf.loc[(routes_gdf["scenario"] == "coolwalks") & (routes_gdf["building_id"].isin(route_sample))]

    fig, ax = plt.subplots(figsize=(10, 10))
    buildings.boundary.plot(ax=ax, color="#cccccc", linewidth=0.2)
    base_routes.plot(ax=ax, color="#888888", linewidth=1.1, alpha=0.8)
    stops.plot(ax=ax, color="#0b57d0", markersize=18)
    handles = [
        Patch(facecolor="white", edgecolor="#cccccc", label="Buildings"),
        Line2D([0], [0], color="#888888", lw=2, label="Baseline route"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#0b57d0", markersize=8, label="Subway entrances"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=9)
    ax.set_axis_off()
    ax.set_title("Baseline routes sample")
    fig.savefig(maps_dir / "04_routes_baseline_sample.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 10))
    buildings.boundary.plot(ax=ax, color="#cccccc", linewidth=0.2)
    cool_routes.plot(ax=ax, color="#2f9e44", linewidth=1.1, alpha=0.85)
    stops.plot(ax=ax, color="#0b57d0", markersize=18)
    handles = [
        Patch(facecolor="white", edgecolor="#cccccc", label="Buildings"),
        Line2D([0], [0], color="#2f9e44", lw=2, label="Shade-aware route"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#0b57d0", markersize=8, label="Subway entrances"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=9)
    ax.set_axis_off()
    ax.set_title("Shade-aware routes sample")
    fig.savefig(maps_dir / "05_routes_coolwalks_sample.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 10))
    buildings_web.boundary.plot(ax=ax, color="white", linewidth=0.35, alpha=0.8)
    trees_web.plot(ax=ax, color="#4caf50", markersize=2, alpha=0.35)
    stops_web.plot(ax=ax, color="#00b0ff", markersize=16)
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, attribution=False, zoom=16)
    handles = [
        Line2D([0], [0], color="white", lw=2, label="Buildings"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4caf50", markersize=6, label="Trees"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#00b0ff", markersize=8, label="Subway entrances"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=9)
    ax.set_axis_off()
    ax.set_title("Satellite overview of study area")
    fig.savefig(maps_dir / "06_satellite_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 10))
    shadow_web.plot(
        ax=ax,
        column="shadow_fraction",
        cmap="magma_r",
        linewidth=1.4,
        legend=True,
        legend_kwds={"label": "Share of street segment in shade (0-1)"},
    )
    stops_web.plot(ax=ax, color="#0b57d0", markersize=12)
    trees_web.plot(ax=ax, color="#2f7d32", markersize=1.5, alpha=0.25)
    handles = [
        Line2D([0], [0], color="#444444", lw=2, label="Street network"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2f7d32", markersize=6, label="Trees"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#0b57d0", markersize=8, label="Subway entrances"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=9)
    ax.set_axis_off()
    ax.set_title("Street-network shade fraction")
    fig.savefig(maps_dir / "07_shadow_fraction_network.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 10))
    shadow_len = shadow_web["sg_shadow_length"].fillna(0).clip(lower=0)
    shadow_labels, class_labels = _classify_shadow_length_fisher_jenks(shadow_len.tolist(), k=5)
    shadow_plot = shadow_web.assign(
        shadow_len_class=pd.Categorical(shadow_labels, categories=class_labels, ordered=True)
    )
    positive_color_count = max(len(class_labels) - 1, 0)
    palette = [plt.cm.viridis(x) for x in np.linspace(0.08, 0.98, positive_color_count)] if positive_color_count else []
    class_colors = ["#bdbdbd", *palette]
    cmap = ListedColormap(class_colors)
    shadow_plot.plot(
        ax=ax,
        column="shadow_len_class",
        cmap=cmap,
        linewidth=1.6,
        categorical=True,
    )
    stops_web.plot(ax=ax, color="#0b57d0", markersize=12)
    handles = [
        Line2D([0], [0], color="#444444", lw=2, label="Street network"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#0b57d0", markersize=8, label="Subway entrances"),
    ]
    handles.extend(
        Patch(facecolor=color, edgecolor="none", label=label)
        for label, color in zip(class_labels, class_colors)
    )
    ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=8, title="Shadowed length")
    ax.set_axis_off()
    ax.set_title("Street-network shadow length")
    fig.savefig(maps_dir / "08_shadow_length_network.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    base_routes_web = base_routes.to_crs(3857)
    cool_routes_web = cool_routes.to_crs(3857)
    fig, ax = plt.subplots(figsize=(10, 10))
    shadow_web.plot(
        ax=ax,
        column="shadow_fraction",
        cmap="Greys",
        linewidth=1.0,
        alpha=0.6,
        legend=True,
        legend_kwds={"label": "Share of street segment in shade (0-1)"},
    )
    base_routes_web.plot(ax=ax, color="#7f1734", linewidth=2.0, alpha=0.82)
    cool_routes_web.plot(ax=ax, color="#2f9e44", linewidth=2.0, alpha=0.85)
    stops_web.plot(ax=ax, color="#0b57d0", markersize=16)
    handles = [
        Line2D([0], [0], color="#7f1734", lw=2, label="Baseline route"),
        Line2D([0], [0], color="#2f9e44", lw=2, label="Shade-aware route"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#0b57d0", markersize=8, label="Subway entrances"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=9)
    ax.set_axis_off()
    ax.set_title("Routes over shadow network")
    fig.savefig(maps_dir / "09_routes_over_shadow_network.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(area: StudyArea = UWS_1500M, force: bool = False) -> None:
    work_dir = area_work_dir(area)
    out_dir = area_out_dir(area)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    buildings = fetch_buildings(area, force=force)
    stops = fetch_subway_entrances(area, force=force)
    trees = prepare_trees(area, force=force)
    buildings.to_file(work_dir / "buildings.geojson", driver="GeoJSON")
    stops.to_file(work_dir / "subway_entrances.geojson", driver="GeoJSON")

    build_shadow_graph(area, timestamp_local="2026-08-15T15:00:00", force=force)
    graph, nodes = load_shadow_graph(area)
    summary_df, routes_gdf = route_all_buildings(graph, buildings, stops, nodes, area)
    summary_df, routes_gdf = deduplicate_results(summary_df, routes_gdf)

    compare = summary_df.pivot(index="building_id", columns="scenario", values="generalized_time_min")
    if {"baseline", "coolwalks"}.issubset(compare.columns):
        compare["delta"] = compare["coolwalks"] - compare["baseline"]
    summary_df.to_csv(out_dir / "route_summary.csv", index=False)
    routes_gdf.to_file(out_dir / "routes.geojson", driver="GeoJSON")
    render_maps(area, buildings, trees, stops, summary_df, routes_gdf)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--area", choices=["uws", "soho"], default="uws")
    args = parser.parse_args()
    area = SOHO_1500M if args.area == "soho" else UWS_1500M
    run(area=area, force=args.force)


if __name__ == "__main__":
    main()
