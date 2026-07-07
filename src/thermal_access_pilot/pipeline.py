from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import LineString

from .config import PilotConfig
from .external import fetch_external_tiles
from .local_inputs import load_local_inputs, snap_origins_to_graph
from .maps import render_maps, validate_maps
from .routing import attach_edge_exposure, route_all_origins
from .surfaces import build_surfaces
from .thermal import run_thermal


def _path_geometry(graph, path: list) -> LineString:
    coords = []
    for u, v in zip(path, path[1:]):
        line = graph[u][v]["geometry"]
        segment = list(line.coords)
        if coords and coords[-1] == segment[0]:
            coords.extend(segment[1:])
        else:
            coords.extend(segment)
    return LineString(coords) if len(coords) >= 2 else LineString()


def _write_route_outputs(cfg: PilotConfig, local, thermal: dict[str, Path | str]) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    out_tables = cfg.output_dir / "tables"
    out_routes = cfg.output_dir / "routes"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_routes.mkdir(parents=True, exist_ok=True)

    with rasterio.open(thermal["utci"]) as src:
        raster = src.read(1)
        transform = src.transform
    graph = attach_edge_exposure(local.graph, raster, transform, cfg.hot_threshold_c)

    edge_rows = []
    for u, v, data in graph.edges(data=True):
        edge_rows.append({"u": u, "v": v, **{k: data[k] for k in ["length_m", "mean_utci_c", "max_utci_c", "hot_fraction", "hot_length_m", "sampled_fraction"]}, "geometry": data["geometry"]})
    gpd.GeoDataFrame(edge_rows, crs=cfg.crs).to_parquet(out_routes / "exposed_walk_edges.parquet")

    graph_nodes = local.nodes.loc[local.nodes["node_id"].isin(graph.nodes)].copy()
    snapped = snap_origins_to_graph(local.origins, graph_nodes, cfg.max_snap_distance_m)
    origins = sorted(snapped["node_id"].unique().tolist())
    stops = sorted(local.stops.loc[local.stops["node_id"].isin(graph.nodes), "node_id"].unique().tolist())
    scenarios = (0.0,) + tuple(cfg.penalties)

    result_rows = []
    route_rows = []
    by_node = snapped.groupby("node_id")["building_id"].apply(list).to_dict()
    for penalty in scenarios:
        routed = route_all_origins(graph, origins, stops, penalty=penalty, walk_speed_m_s=cfg.walk_speed_m_s)
        for node, route in routed.items():
            for building_id in by_node.get(node, []):
                result_rows.append({"building_id": int(building_id), "origin_node": int(node), "penalty": penalty, **route})
            if route.get("status") == "ok":
                route_rows.append({"origin_node": int(node), "penalty": penalty, **{k: route[k] for k in ["length_m", "time_min", "hot_length_m", "hot_fraction", "generalized_time_min", "stop_node"]}, "geometry": _path_geometry(graph, route["path"])})

    results = pd.DataFrame(result_rows)
    routes = gpd.GeoDataFrame(route_rows, crs=cfg.crs)
    results.drop(columns=["path"], errors="ignore").to_parquet(out_tables / "building_results.parquet")
    routes.to_parquet(out_routes / "routes.parquet")
    return results, routes


def run(cfg: PilotConfig, force: bool = False) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    print("stage=local_inputs")
    local = load_local_inputs(cfg)
    print(f"local buildings={len(local.buildings)} origins={len(local.origins)} graph_nodes={local.graph.number_of_nodes()} stops={len(local.stops)}")

    print("stage=external_tiles")
    tiles = fetch_external_tiles(cfg)
    print("external tiles=" + ",".join(sorted(tiles)))

    print("stage=surfaces")
    surfaces = build_surfaces(cfg, local, tiles)

    print("stage=thermal")
    thermal = run_thermal(cfg, surfaces, force=force)
    print(f"thermal engine={thermal['engine']}")

    print("stage=routing")
    results, routes = _write_route_outputs(cfg, local, thermal)
    ok = results.loc[results["status"] == "ok"]
    print(f"routing rows={len(results)} ok={len(ok)} routes={len(routes)}")

    print("stage=maps")
    render_maps(cfg, local, thermal, results, routes)
    validate_maps(cfg.output_dir)

    summary = {
        "city": "kaliningrad_russia",
        "center_lon": cfg.center_lon,
        "center_lat": cfg.center_lat,
        "core_radius_m": cfg.core_radius_m,
        "model_radius_m": cfg.model_radius_m,
        "thermal_engine": thermal["engine"],
        "hot_threshold_c": cfg.hot_threshold_c,
        "buildings_in_model": len(local.buildings),
        "building_origins_in_core": len(local.origins),
        "snapped_building_route_rows": int(len(results)),
        "ok_route_rows": int(len(ok)),
        "penalties": [0.0, *cfg.penalties],
        "baseline_mean_time_min": float(ok.loc[ok["penalty"] == 0, "time_min"].mean()),
        "baseline_mean_hot_fraction": float(ok.loc[ok["penalty"] == 0, "hot_fraction"].mean()),
    }
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "surfaces": {k: str(v) for k, v in surfaces.__dict__.items()},
        "thermal": {k: str(v) for k, v in thermal.items()},
        "tables": ["tables/building_results.parquet", "routes/routes.parquet", "routes/exposed_walk_edges.parquet"],
        "maps": [f"maps/{name}" for name in ["01_inputs.png", "02_thermal_fields.png", "03_routes_examples.png", "04_building_exposure.png", "05_time_change.png", "06_sensitivity.png"]],
    }
    (cfg.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("stage=done")
