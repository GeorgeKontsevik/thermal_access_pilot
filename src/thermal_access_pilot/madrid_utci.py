from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import rasterio
import solweig
from affine import Affine
from pyproj import CRS, Transformer
from rasterio.features import rasterize
from rasterio.plot import show
from rasterio.transform import rowcol
from shapely.geometry import LineString, Point, box

from .config import PilotConfig
from .local_inputs import LocalInputs, resolve_heights
from .surfaces import build_surfaces
from .weather import select_extreme_hour


ROOT = Path("/Users/gk/Code/super-duper-disser/thermal_access_pilot")
OUT_DIR = ROOT / "outputs" / "madrid_utci_5km"

UTCI_CLASS_ORDER = [
    "cold_stress",
    "no_thermal_stress",
    "moderate_heat_stress",
    "strong_heat_stress",
    "very_strong_heat_stress",
    "extreme_heat_stress",
]
UTCI_CLASS_TO_CODE = {name: idx for idx, name in enumerate(UTCI_CLASS_ORDER)}
UTCI_COST_FACTORS = {
    "cold_stress": 1.15,
    "no_thermal_stress": 1.0,
    "moderate_heat_stress": 1.10,
    "strong_heat_stress": 1.25,
    "very_strong_heat_stress": 1.50,
    "extreme_heat_stress": 1.80,
}


@dataclass(frozen=True)
class MadridSpec:
    center_lon: float = -3.7038
    center_lat: float = 40.4168
    core_radius_m: float = 2500.0
    model_halo_m: float = 500.0
    pixel_size_m: float = 2.0
    walk_speed_m_s: float = 1.2
    max_snap_distance_m: float = 100.0
    crs: str = "EPSG:25830"
    shadow_weight_note: str = "UTCI threshold-based edge cost, not shortest-time only"


def utci_category(value: float) -> str:
    if value < 9:
        return "cold_stress"
    if value < 26:
        return "no_thermal_stress"
    if value < 32:
        return "moderate_heat_stress"
    if value < 38:
        return "strong_heat_stress"
    if value <= 46:
        return "very_strong_heat_stress"
    return "extreme_heat_stress"


def _tile_codes_for_lonlat(latitude: float, longitude: float) -> dict[str, str]:
    lat_floor = math.floor(latitude)
    lon_floor = math.floor(longitude)
    srtm_ns = f"N{lat_floor:02d}" if lat_floor >= 0 else f"S{abs(lat_floor):02d}"
    srtm_ew = f"E{lon_floor:03d}" if lon_floor >= 0 else f"W{abs(lon_floor):03d}"
    cover_lat = math.floor(latitude / 3) * 3
    cover_lon = math.floor(longitude / 3) * 3
    cover_ns = f"N{cover_lat:02d}" if cover_lat >= 0 else f"S{abs(cover_lat):02d}"
    cover_ew = f"E{cover_lon:03d}" if cover_lon >= 0 else f"W{abs(cover_lon):03d}"
    return {
        "srtm_ns": srtm_ns,
        "srtm_tile": f"{srtm_ns}{srtm_ew}",
        "cover_tile": f"{cover_ns}{cover_ew}",
    }


def _download(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target
    tmp = target.with_suffix(target.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=180) as response, tmp.open("wb") as fh:
        shutil.copyfileobj(response, fh)
    tmp.replace(target)
    return target


def fetch_madrid_weather(latitude: float, longitude: float, start_date: str = "2025-06-01", end_date: str = "2025-08-31") -> pd.DataFrame:
    params = urllib.parse.urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,shortwave_radiation,wind_speed_10m,surface_pressure",
            "timezone": "Europe/Madrid",
        }
    )
    with urllib.request.urlopen(f"https://archive-api.open-meteo.com/v1/archive?{params}", timeout=120) as response:
        payload = json.load(response)
    hourly = payload["hourly"]
    frame = pd.DataFrame(
        {
            "datetime_local": pd.to_datetime(hourly["time"]),
            "ta_c": hourly["temperature_2m"],
            "rh": hourly["relative_humidity_2m"],
            "dewpoint_c": hourly["dew_point_2m"],
            "global_rad_w_m2": hourly["shortwave_radiation"],
            "ws": hourly["wind_speed_10m"],
            "pressure": hourly["surface_pressure"],
        }
    )
    return frame.dropna(subset=["ta_c", "rh", "global_rad_w_m2", "ws", "pressure"]).copy()


def fetch_external_tiles(spec: MadridSpec) -> dict[str, Path]:
    raw = OUT_DIR / "inputs" / "external" / "raw"
    codes = _tile_codes_for_lonlat(spec.center_lat, spec.center_lon)
    srtm_gz = _download(
        f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{codes['srtm_ns']}/{codes['srtm_tile']}.hgt.gz",
        raw / "srtm" / f"{codes['srtm_tile']}.hgt.gz",
    )
    srtm = srtm_gz.with_suffix("")
    if not srtm.exists():
        with gzip.open(srtm_gz, "rb") as src, srtm.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    cover_tile = codes["cover_tile"]
    worldcover = _download(
        f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{cover_tile}_Map.tif",
        raw / "worldcover" / f"ESA_WorldCover_10m_2021_v200_{cover_tile}_Map.tif",
    )
    canopy = _download(
        f"https://libdrive.ethz.ch/index.php/s/cO8or7iOe5dT2Rt/download?path=%2F3deg_cogs&files=ETH_GlobalCanopyHeight_10m_2020_{cover_tile}_Map.tif",
        raw / "canopy" / f"ETH_GlobalCanopyHeight_10m_2020_{cover_tile}_Map.tif",
    )
    return {"srtm": srtm, "worldcover": worldcover, "canopy": canopy}


def _study_geometries(spec: MadridSpec) -> tuple[Point, object, object]:
    tr = Transformer.from_crs("EPSG:4326", spec.crs, always_xy=True)
    x, y = tr.transform(spec.center_lon, spec.center_lat)
    center = Point(x, y)
    core = center.buffer(spec.core_radius_m)
    model_r = spec.core_radius_m + spec.model_halo_m
    model_area = box(x - model_r, y - model_r, x + model_r, y + model_r)
    return center, core, model_area


def build_local_inputs(spec: MadridSpec) -> LocalInputs:
    center, core, model_area = _study_geometries(spec)
    polygon_wgs = gpd.GeoSeries([model_area], crs=spec.crs).to_crs(4326).iloc[0]

    buildings = ox.features_from_polygon(polygon_wgs, tags={"building": True}).reset_index()
    buildings = buildings.loc[buildings.geometry.notna()].to_crs(spec.crs)
    buildings = buildings.loc[buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    buildings["building_id"] = range(len(buildings))
    buildings["storey"] = buildings.get("building:levels")
    buildings["height"] = buildings.get("height")
    buildings = resolve_heights(buildings)

    origins = buildings[["building_id", "height_m", "height_rule", "geometry"]].copy()
    origins = origins.loc[origins.intersects(core)].copy()
    origins["geometry"] = origins.geometry.representative_point()

    stop_tags = {
        "public_transport": ["platform", "stop_position"],
        "highway": "bus_stop",
        "railway": ["tram_stop", "station", "halt", "subway_entrance"],
        "amenity": "bus_station",
    }
    stops = ox.features_from_polygon(polygon_wgs, tags=stop_tags).reset_index()
    stops = stops.loc[stops.geometry.notna()].to_crs(spec.crs)
    stops["geometry"] = stops.geometry.representative_point()
    stops["stop_id"] = range(len(stops))

    graph = ox.graph_from_polygon(polygon_wgs, network_type="walk", simplify=True)
    graph = ox.project_graph(graph, to_crs=spec.crs)
    nodes, edges = ox.graph_to_gdfs(graph)
    nodes = nodes.reset_index().rename(columns={"osmid": "node_id"})
    edges = edges.reset_index()
    out_graph = nx.Graph()
    for row in nodes.itertuples(index=False):
        out_graph.add_node(int(row.node_id), geometry=row.geometry)
    for row in edges.itertuples(index=False):
        u, v = int(row.u), int(row.v)
        geom = row.geometry
        length = float(getattr(row, "length", geom.length))
        if out_graph.has_edge(u, v) and out_graph[u][v]["length_m"] <= length:
            continue
        out_graph.add_edge(u, v, geometry=geom, length_m=length)

    nodes = nodes[["node_id", "geometry"]].copy().set_crs(spec.crs)
    return LocalInputs(center=center, core=core, model_area=model_area, buildings=buildings, origins=origins, nodes=nodes, stops=stops, graph=out_graph)


def build_cfg(spec: MadridSpec) -> PilotConfig:
    return PilotConfig(
        repo_root=ROOT.parent,
        city_bundle=ROOT,
        output_dir=OUT_DIR,
        center_lon=spec.center_lon,
        center_lat=spec.center_lat,
        core_radius_m=spec.core_radius_m,
        model_halo_m=spec.model_halo_m,
        crs=spec.crs,
        pixel_size_m=spec.pixel_size_m,
        walk_speed_m_s=spec.walk_speed_m_s,
        hot_threshold_c=32.0,
        max_snap_distance_m=spec.max_snap_distance_m,
        penalties=(0.0,),
    )


def _copy_profile(src_path: Path) -> dict:
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.update(dtype="float32", compress="deflate")
    return profile


def _write_like(path: Path, values: np.ndarray, like: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **_copy_profile(like)) as dst:
        dst.write(values.astype("float32"), 1)


def run_utci(spec: MadridSpec, surfaces) -> dict[str, Path | str]:
    out_dir = OUT_DIR / "thermal"
    out_dir.mkdir(parents=True, exist_ok=True)
    utci_path = out_dir / "headline_utci.tif"
    tmrt_path = out_dir / "headline_tmrt.tif"
    shadow_path = out_dir / "headline_shadow.tif"
    category_path = out_dir / "utci_category.tif"
    metadata_path = out_dir / "thermal_metadata.json"

    weather = fetch_madrid_weather(spec.center_lat, spec.center_lon, start_date="2025-06-01", end_date="2025-08-31")
    weather_path = out_dir / "selected_day_weather.parquet"
    weather.to_parquet(weather_path)
    selected = select_extreme_hour(weather)

    engine = "solweig"
    error = None
    try:
        surface = solweig.SurfaceData.prepare(
            dsm=surfaces.dsm,
            dem=surfaces.dem,
            cdsm=surfaces.cdsm,
            land_cover=surfaces.land_cover,
            working_dir=out_dir / "solweig_work",
            cdsm_relative=True,
        )
        location = solweig.Location(latitude=spec.center_lat, longitude=spec.center_lon, utc_offset=2)
        record = solweig.Weather(
            datetime=selected["datetime_local"].to_pydatetime(),
            ta=float(selected["ta_c"]),
            rh=float(selected["rh"]),
            global_rad=float(selected["global_rad_w_m2"]),
            ws=float(selected["ws"]),
            pressure=float(selected["pressure"]),
            timestep_minutes=60,
        )
        _ = solweig.calculate(
            surface=surface,
            weather=[record],
            location=location,
            output_dir=out_dir / "solweig",
            outputs=["tmrt", "utci", "shadow"],
        )
        utci_candidates = sorted((out_dir / "solweig").glob("**/utci*.tif"))
        tmrt_candidates = sorted((out_dir / "solweig").glob("**/tmrt*.tif"))
        shadow_candidates = sorted((out_dir / "solweig").glob("**/shadow*.tif"))
        for src, dst in [(utci_candidates[-1], utci_path), (tmrt_candidates[-1], tmrt_path), (shadow_candidates[-1], shadow_path)]:
            with rasterio.open(src) as dataset:
                values = dataset.read(1)
                profile = dataset.profile.copy()
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
            with rasterio.open(dst, "w", **profile) as target:
                target.write(values, 1)
    except Exception as exc:
        engine = "diagnostic_surface_not_solweig"
        error = repr(exc)
        with rasterio.open(surfaces.dsm) as dsm_src, rasterio.open(surfaces.dem) as dem_src, rasterio.open(surfaces.cdsm) as cdsm_src:
            dsm = dsm_src.read(1)
            dem = dem_src.read(1)
            cdsm = cdsm_src.read(1)
        height = np.maximum(dsm - dem, 0)
        shade = np.clip((height + cdsm) / 20, 0, 1)
        tmrt = float(selected["ta_c"]) + 28 * (1 - shade) + 2 * (height > 0)
        utci = float(selected["ta_c"]) + 0.28 * (tmrt - float(selected["ta_c"])) - 1.2 * float(selected["ws"])
        _write_like(utci_path, utci, surfaces.dsm)
        _write_like(tmrt_path, tmrt, surfaces.dsm)
        _write_like(shadow_path, shade, surfaces.dsm)

    with rasterio.open(utci_path) as src:
        utci = src.read(1)
        profile = src.profile.copy()
    category = np.vectorize(lambda x: UTCI_CLASS_TO_CODE[utci_category(float(x))])(utci)
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.update(dtype="uint8", compress="deflate", nodata=0)
    with rasterio.open(category_path, "w", **profile) as dst:
        dst.write(category.astype("uint8"), 1)
    meta = {
        "engine": engine,
        "error": error,
        "selected_hour": str(selected["datetime_local"]),
        "ta_c": float(selected["ta_c"]),
        "rh": float(selected["rh"]),
        "global_rad_w_m2": float(selected["global_rad_w_m2"]),
        "ws": float(selected["ws"]),
        "utci_min": float(np.nanmin(utci)),
        "utci_mean": float(np.nanmean(utci)),
        "utci_max": float(np.nanmax(utci)),
        "share_gt_26c": float(np.nanmean(utci >= 26)),
        "share_gt_32c": float(np.nanmean(utci >= 32)),
        "share_gt_38c": float(np.nanmean(utci >= 38)),
        "share_gt_46c": float(np.nanmean(utci > 46)),
    }
    metadata_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"utci": utci_path, "tmrt": tmrt_path, "shadow": shadow_path, "category": category_path, "metadata": metadata_path, "engine": engine}


def _snap_points_to_nodes(points: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, id_col: str, max_distance_m: float) -> gpd.GeoDataFrame:
    snapped = gpd.sjoin_nearest(points[[id_col, "geometry"]], nodes[["node_id", "geometry"]], how="left", distance_col="snap_m", max_distance=max_distance_m)
    return snapped.drop(columns=["index_right"]).dropna(subset=["node_id"]).astype({"node_id": int})


def _sample_edge_categories(geometry: LineString, raster: np.ndarray, transform: Affine, step_m: float = 2.0) -> dict[str, float]:
    length = max(float(geometry.length), step_m)
    distances = np.arange(0, length + 0.001, step_m)
    values: list[float] = []
    for distance in distances:
        point = geometry.interpolate(min(float(distance), geometry.length))
        row, col = rowcol(transform, point.x, point.y)
        if 0 <= row < raster.shape[0] and 0 <= col < raster.shape[1]:
            value = float(raster[row, col])
            if np.isfinite(value):
                values.append(value)
    if not values:
        return {
            "mean_utci_c": float("nan"),
            "max_utci_c": float("nan"),
            "utci_factor": 1.0,
            **{f"share_{name}": 0.0 for name in UTCI_CLASS_ORDER},
        }
    counts = {name: 0 for name in UTCI_CLASS_ORDER}
    factors = []
    for value in values:
        cat = utci_category(value)
        counts[cat] += 1
        factors.append(UTCI_COST_FACTORS[cat])
    total = len(values)
    return {
        "mean_utci_c": float(np.mean(values)),
        "max_utci_c": float(np.max(values)),
        "utci_factor": float(np.mean(factors)),
        **{f"share_{name}": counts[name] / total for name in UTCI_CLASS_ORDER},
    }


def attach_edge_utci(graph: nx.Graph, utci_path: Path) -> nx.Graph:
    with rasterio.open(utci_path) as src:
        raster = src.read(1)
        transform = src.transform
    out = graph.copy()
    for _, _, data in out.edges(data=True):
        stats = _sample_edge_categories(data["geometry"], raster, transform)
        for key, value in stats.items():
            data[key] = value
    return out


def _edge_weight(_, __, data, scenario: str, walk_speed_m_s: float) -> float:
    factor = 1.0 if scenario == "baseline" else float(data.get("utci_factor", 1.0))
    return float(data["length_m"]) * factor / walk_speed_m_s


def route_all(graph: nx.Graph, origin_nodes: list[int], stop_nodes: list[int], walk_speed_m_s: float) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    rows: list[dict] = []
    route_rows: list[dict] = []
    for scenario in ["baseline", "utci"]:
        _, paths = nx.multi_source_dijkstra(graph, set(stop_nodes), weight=lambda u, v, d, sc=scenario: _edge_weight(u, v, d, sc, walk_speed_m_s))
        for node in origin_nodes:
            if node not in paths:
                rows.append({"origin_node": node, "scenario": scenario, "status": "unreachable"})
                continue
            path = list(reversed(paths[node]))
            length = generalized_s = 0.0
            shares = {name: 0.0 for name in UTCI_CLASS_ORDER}
            mean_values = []
            max_values = []
            for u, v in zip(path, path[1:]):
                data = graph[u][v]
                length += float(data["length_m"])
                generalized_s += _edge_weight(u, v, data, scenario, walk_speed_m_s)
                for name in UTCI_CLASS_ORDER:
                    shares[name] += float(data.get(f"share_{name}", 0.0)) * float(data["length_m"])
                mean_values.append(float(data.get("mean_utci_c", np.nan)))
                max_values.append(float(data.get("max_utci_c", np.nan)))
            valid_mean = [value for value in mean_values if np.isfinite(value)]
            valid_max = [value for value in max_values if np.isfinite(value)]
            row = {
                "origin_node": node,
                "scenario": scenario,
                "status": "ok",
                "stop_node": path[-1],
                "length_m": length,
                "time_min": length / walk_speed_m_s / 60.0,
                "generalized_time_min": generalized_s / 60.0,
                "route_mean_utci_c": float(np.mean(valid_mean)) if valid_mean else float("nan"),
                "route_max_utci_c": float(np.max(valid_max)) if valid_max else float("nan"),
                "path": path,
            }
            for name in UTCI_CLASS_ORDER:
                row[f"share_{name}"] = shares[name] / length if length else 0.0
            rows.append(row)
            route_rows.append({k: row[k] for k in row if k != "path"} | {"geometry": _path_geometry(graph, path)})
    return pd.DataFrame(rows), gpd.GeoDataFrame(route_rows, geometry="geometry", crs="EPSG:25830")


def _path_geometry(graph: nx.Graph, path: list[int]) -> LineString:
    coords: list[tuple[float, float]] = []
    for u, v in zip(path, path[1:]):
        segment = list(graph[u][v]["geometry"].coords)
        if coords and coords[-1] == segment[0]:
            coords.extend(segment[1:])
        else:
            coords.extend(segment)
    return LineString(coords)


def _edge_key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u <= v else (v, u)


def _dominant_utci_class(data: dict) -> str:
    return max(UTCI_CLASS_ORDER, key=lambda name: float(data.get(f"share_{name}", 0.0)))


def _street_edges_gdf(graph: nx.Graph, crs: str) -> gpd.GeoDataFrame:
    rows = []
    for u, v, data in graph.edges(data=True):
        rows.append(
            {
                "u": u,
                "v": v,
                "edge_key": _edge_key(u, v),
                "geometry": data["geometry"],
                "dominant_utci_class": _dominant_utci_class(data),
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


def _changed_route_edges(graph: nx.Graph, node_results: pd.DataFrame, crs: str) -> gpd.GeoDataFrame:
    baseline = node_results.loc[node_results["scenario"] == "baseline", ["origin_node", "path"]].rename(columns={"path": "path_baseline"})
    utci = node_results.loc[node_results["scenario"] == "utci", ["origin_node", "path"]].rename(columns={"path": "path_utci"})
    merged = baseline.merge(utci, on="origin_node", how="inner")
    merged = merged.loc[merged["path_baseline"].apply(tuple) != merged["path_utci"].apply(tuple)].copy()
    baseline_counts: dict[tuple[int, int], int] = {}
    utci_counts: dict[tuple[int, int], int] = {}
    for _, row in merged.iterrows():
        for key_store, path in ((baseline_counts, row["path_baseline"]), (utci_counts, row["path_utci"])):
            for u, v in zip(path, path[1:]):
                key = _edge_key(int(u), int(v))
                key_store[key] = key_store.get(key, 0) + 1
    rows = []
    seen = set(baseline_counts) | set(utci_counts)
    for u, v, data in graph.edges(data=True):
        key = _edge_key(u, v)
        if key not in seen:
            continue
        rows.append(
            {
                "u": u,
                "v": v,
                "geometry": data["geometry"],
                "baseline_count": baseline_counts.get(key, 0),
                "utci_count": utci_counts.get(key, 0),
            }
        )
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    if len(out):
        out["dominant_scenario"] = np.where(out["utci_count"] > out["baseline_count"], "utci", "baseline")
    else:
        out["dominant_scenario"] = pd.Series(dtype="object")
    return out


def render_maps(
    spec: MadridSpec,
    local: LocalInputs,
    thermal: dict[str, Path | str],
    graph: nx.Graph,
    node_results: pd.DataFrame,
) -> None:
    maps_dir = OUT_DIR / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    gpd.GeoSeries([local.core], crs=spec.crs).boundary.plot(ax=ax, color="black", linewidth=1)
    local.buildings.boundary.plot(ax=ax, color="#bdbdbd", linewidth=0.15)
    local.stops.plot(ax=ax, color="#1565c0", markersize=10)
    ax.set_title("Madrid 5×5 km: buildings and PT stops")
    ax.set_axis_off()
    fig.savefig(maps_dir / "01_inputs.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    with rasterio.open(thermal["utci"]) as src:
        show(src.read(1), transform=src.transform, ax=axes[0], cmap="magma")
    axes[0].set_title("UTCI °C")
    axes[0].set_axis_off()
    with rasterio.open(thermal["category"]) as src:
        show(src.read(1), transform=src.transform, ax=axes[1], cmap="RdYlBu_r")
    axes[1].set_title("UTCI threshold categories")
    axes[1].set_axis_off()
    fig.savefig(maps_dir / "02_utci_fields.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    street_edges = _street_edges_gdf(graph, spec.crs)
    street_palette = {
        "cold_stress": "#2b8cbe",
        "no_thermal_stress": "#91bfdb",
        "moderate_heat_stress": "#fee08b",
        "strong_heat_stress": "#fdae61",
        "very_strong_heat_stress": "#f46d43",
        "extreme_heat_stress": "#d73027",
    }
    fig, ax = plt.subplots(figsize=(10, 8))
    local.buildings.boundary.plot(ax=ax, color="#e0e0e0", linewidth=0.1)
    for cls in UTCI_CLASS_ORDER:
        subset = street_edges.loc[street_edges["dominant_utci_class"] == cls]
        if len(subset):
            subset.plot(ax=ax, color=street_palette[cls], linewidth=0.8)
    local.stops.plot(ax=ax, color="#1565c0", markersize=6, alpha=0.55)
    ax.legend(
        handles=[Line2D([0], [0], color=street_palette[cls], lw=2, label=cls.replace("_", " ")) for cls in UTCI_CLASS_ORDER],
        loc="lower left",
        frameon=True,
        fontsize=8,
    )
    ax.set_title("Street segments by dominant UTCI class")
    ax.set_axis_off()
    fig.savefig(maps_dir / "03_street_utci_classes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    changed_edges = _changed_route_edges(graph, node_results.loc[node_results["status"] == "ok"].copy(), spec.crs)
    fig, ax = plt.subplots(figsize=(10, 8))
    local.buildings.boundary.plot(ax=ax, color="#ebebeb", linewidth=0.1)
    if len(changed_edges):
        baseline_only = changed_edges.loc[changed_edges["dominant_scenario"] == "baseline"]
        utci_only = changed_edges.loc[changed_edges["dominant_scenario"] == "utci"]
        if len(baseline_only):
            baseline_only.plot(ax=ax, color="#7f1734", linewidth=1.3, alpha=0.9)
        if len(utci_only):
            utci_only.plot(ax=ax, color="#2f9e44", linewidth=1.3, alpha=0.9)
    local.stops.plot(ax=ax, color="#1565c0", markersize=6, alpha=0.5)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#7f1734", lw=2, label="Baseline changed-route segments"),
            Line2D([0], [0], color="#2f9e44", lw=2, label="UTCI-aware changed-route segments"),
        ],
        loc="lower left",
        frameon=True,
        fontsize=8,
    )
    ax.set_title("Street segments used by changed routes only")
    ax.set_axis_off()
    fig.savefig(maps_dir / "04_street_changed_routes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    spec = MadridSpec()
    cfg = build_cfg(spec)
    local = build_local_inputs(spec)
    tiles = fetch_external_tiles(spec)
    surfaces = build_surfaces(cfg, local, tiles)
    thermal = run_utci(spec, surfaces)

    graph = attach_edge_utci(local.graph, thermal["utci"])
    snapped_origins = _snap_points_to_nodes(local.origins, local.nodes, "building_id", spec.max_snap_distance_m)
    snapped_stops = _snap_points_to_nodes(local.stops[["stop_id", "geometry"]].copy(), local.nodes, "stop_id", spec.max_snap_distance_m)
    origin_nodes = sorted(snapped_origins["node_id"].unique().tolist())
    stop_nodes = sorted(snapped_stops["node_id"].unique().tolist())
    by_origin = snapped_origins.groupby("node_id")["building_id"].apply(list).to_dict()

    result_df, route_df = route_all(graph, origin_nodes, stop_nodes, spec.walk_speed_m_s)
    expanded_rows = []
    for row in result_df.to_dict("records"):
        for building_id in by_origin.get(row["origin_node"], []):
            expanded_rows.append({**row, "building_id": int(building_id)})
    results = pd.DataFrame(expanded_rows)

    tables_dir = OUT_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results.drop(columns=["path"], errors="ignore").to_parquet(tables_dir / "building_results.parquet")
    route_df.to_parquet(tables_dir / "routes.parquet")
    render_maps(spec, local, thermal, graph, result_df)

    baseline = results.loc[(results["scenario"] == "baseline") & (results["status"] == "ok")].copy()
    utci = results.loc[(results["scenario"] == "utci") & (results["status"] == "ok")].copy()
    compare = baseline.merge(utci, on="building_id", suffixes=("_baseline", "_utci"))
    compare["delta_generalized_time_min"] = compare["generalized_time_min_utci"] - compare["time_min_baseline"]
    summary = {
        "city": "madrid_spain",
        "core_size_m": spec.core_radius_m * 2,
        "selected_hour": json.loads(Path(thermal["metadata"]).read_text(encoding="utf-8"))["selected_hour"],
        "engine": thermal["engine"],
        "buildings": int(len(local.buildings)),
        "origins": int(len(local.origins)),
        "stops": int(len(local.stops)),
        "median_baseline_min": float(compare["time_min_baseline"].median()),
        "median_utci_generalized_min": float(compare["generalized_time_min_utci"].median()),
        "median_delta_generalized_min": float(compare["delta_generalized_time_min"].median()),
        "median_baseline_route_max_utci_c": float(compare["route_max_utci_c_baseline"].median()),
        "share_routes_with_very_strong_or_extreme_baseline": float(((compare["share_very_strong_heat_stress_baseline"] + compare["share_extreme_heat_stress_baseline"]) > 0).mean()),
        "share_routes_with_very_strong_or_extreme_utci": float(((compare["share_very_strong_heat_stress_utci"] + compare["share_extreme_heat_stress_utci"]) > 0).mean()),
        "utci_cost_factors": UTCI_COST_FACTORS,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    _ = parser.parse_args()
    run()


if __name__ == "__main__":
    main()
