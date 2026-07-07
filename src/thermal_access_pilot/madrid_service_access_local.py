from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import rasterio
from matplotlib.lines import Line2D
from shapely.geometry import LineString

REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = REPO_ROOT / "thermal_access_pilot" / "outputs" / "madrid_service_access_2km"
CITY_DIR = ROOT / "joint_inputs_baseline" / "madrid_spain"
OUT = ROOT / "local_access"
UTCI_PATH = REPO_ROOT / "thermal_access_pilot" / "outputs" / "madrid_utci_5km" / "thermal" / "headline_utci.tif"
CRS = "EPSG:25830"
WALK_M_PER_MIN = 84.0
PT_M_PER_MIN = 360.0
PT_BOARDING_MIN = 2.0
THRESHOLD_MIN = 15.0
SERVICES = ["school", "polyclinic"]

LABEL_ORDER = [
    "ok_walk",
    "ok_pt_only",
    "failed_no_pt_path",
    "failed_access_gt_threshold",
    "failed_egress_gt_threshold",
    "failed_access_egress_sum_gt_threshold",
    "failed_in_vehicle_gt_threshold",
    "failed_transfer_gt_threshold",
    "failed_multi_component_gt_threshold",
    "failed_total_gt_threshold_no_single_component_gt_threshold",
]
PLOT_ORDER = [
    "failed_no_pt_path",
    "failed_transfer_gt_threshold",
    "failed_access_gt_threshold",
    "failed_egress_gt_threshold",
    "failed_access_egress_sum_gt_threshold",
    "failed_in_vehicle_gt_threshold",
    "failed_multi_component_gt_threshold",
    "failed_total_gt_threshold_no_single_component_gt_threshold",
    "ok_pt_only",
    "ok_walk",
]
LABEL_COLORS = {
    "ok_walk": "#16a34a",
    "ok_pt_only": "#2563eb",
    "failed_no_pt_path": "#475569",
    "failed_access_gt_threshold": "#f59e0b",
    "failed_egress_gt_threshold": "#fb7185",
    "failed_access_egress_sum_gt_threshold": "#f97316",
    "failed_in_vehicle_gt_threshold": "#dc2626",
    "failed_transfer_gt_threshold": "#7c3aed",
    "failed_multi_component_gt_threshold": "#8b5cf6",
    "failed_total_gt_threshold_no_single_component_gt_threshold": "#6b7280",
}
LABEL_RU = {
    "ok_walk": "доступно пешком",
    "ok_pt_only": "доступно на ОТ",
    "failed_no_pt_path": "нет пути по ОТ",
    "failed_access_gt_threshold": "дом - остановка > 15 мин",
    "failed_egress_gt_threshold": "остановка - сервис > 15 мин",
    "failed_access_egress_sum_gt_threshold": "оба пеших участка > 15 мин",
    "failed_in_vehicle_gt_threshold": "поездка в ОТ > 15 мин",
    "failed_transfer_gt_threshold": "пересадки > 15 мин",
    "failed_multi_component_gt_threshold": "несколько компонент > 15 мин",
    "failed_total_gt_threshold_no_single_component_gt_threshold": "сумма > 15 мин, без доминирующей компоненты",
}
SERVICE_RU = {"school": "школы", "polyclinic": "поликлиники"}
UTCI_FACTORS = {
    "cold_stress": 1.15,
    "no_thermal_stress": 1.0,
    "moderate_heat_stress": 1.10,
    "strong_heat_stress": 1.25,
    "very_strong_heat_stress": 1.50,
    "extreme_heat_stress": 1.80,
}


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


def _factor_for_line(line, src: rasterio.io.DatasetReader) -> float:
    if line is None or line.is_empty:
        return 1.0
    vals = []
    for distance in np.linspace(0.0, float(line.length), max(3, int(math.ceil(float(line.length) / 20.0)) + 1)):
        point = line.interpolate(float(distance))
        value = float(next(src.sample([(point.x, point.y)]))[0])
        if np.isfinite(value):
            vals.append(UTCI_FACTORS[utci_category(value)])
    return float(np.mean(vals)) if vals else 1.0


def _load_inputs() -> tuple[gpd.GeoDataFrame, dict[str, gpd.GeoDataFrame], gpd.GeoDataFrame]:
    buildings = gpd.read_parquet(CITY_DIR / "derived_layers" / "buildings_floor_enriched.parquet").to_crs(CRS)
    buildings = buildings.reset_index(drop=True)
    buildings["building_idx"] = buildings.index.astype(int)
    buildings["geometry"] = buildings.geometry.representative_point()
    services = {}
    for name in SERVICES:
        frame = gpd.read_parquet(CITY_DIR / "pipeline_2" / "services_raw" / f"{name}.parquet").to_crs(CRS).reset_index(drop=True)
        frame["geometry"] = frame.geometry.representative_point()
        services[name] = frame
    boundary = gpd.read_parquet(CITY_DIR / "analysis_territory" / "buffer.parquet").to_crs(CRS)
    return buildings, services, boundary


def _configure_osmnx() -> None:
    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.requests_timeout = 180
    ox.settings.overpass_url = "https://maps.mail.ru/osm/tools/overpass/api"


def _cache_paths() -> tuple[Path, Path, Path]:
    cache = OUT / "cache"
    return cache / "walk_graph.pkl", cache / "walk_edges.parquet", cache / "stops.parquet"


def _fetch_or_load_osm(boundary: gpd.GeoDataFrame):
    graph_path, edges_path, stops_path = _cache_paths()
    if graph_path.exists() and edges_path.exists() and stops_path.exists():
        with graph_path.open("rb") as fh:
            return pickle.load(fh), gpd.read_parquet(edges_path), gpd.read_parquet(stops_path)

    _configure_osmnx()
    polygon = boundary.to_crs(4326).geometry.iloc[0]
    print("stage=osmnx_walk_graph")
    raw_graph = ox.graph_from_polygon(polygon, network_type="walk", simplify=True, retain_all=False)
    raw_graph = ox.project_graph(raw_graph, to_crs=CRS)
    nodes, raw_edges = ox.graph_to_gdfs(raw_graph)

    print("stage=osmnx_pt_stops")
    tags = {
        "highway": "bus_stop",
        "public_transport": ["platform", "stop_position"],
        "railway": ["tram_stop", "station", "subway_entrance"],
    }
    stops = ox.features_from_polygon(polygon, tags)
    stops = stops.loc[stops.geometry.geom_type.isin(["Point", "Polygon", "MultiPolygon"])].copy()
    stops = stops.to_crs(CRS)
    stops["geometry"] = stops.geometry.representative_point()
    stops = gpd.clip(stops, boundary).reset_index(drop=True)

    graph = nx.Graph()
    for node, row in nodes.iterrows():
        graph.add_node(node, x=float(row.geometry.x), y=float(row.geometry.y), geometry=row.geometry)

    with rasterio.open(UTCI_PATH) as src:
        for (u, v, _key), row in raw_edges.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                geom = LineString([nodes.loc[u].geometry, nodes.loc[v].geometry])
            length = float(row.get("length", geom.length))
            factor = _factor_for_line(geom, src)
            if graph.has_edge(u, v) and length >= graph[u][v]["length_m"]:
                continue
            graph.add_edge(
                u,
                v,
                length_m=length,
                baseline_time=length / WALK_M_PER_MIN,
                heat_time=length * factor / WALK_M_PER_MIN,
                utci_factor=factor,
                geometry=geom,
            )

    rows = [{"u": u, "v": v, **{k: d[k] for k in ["length_m", "baseline_time", "heat_time", "utci_factor"]}, "geometry": d["geometry"]} for u, v, d in graph.edges(data=True)]
    edges = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    with graph_path.open("wb") as fh:
        pickle.dump(graph, fh)
    edges.to_parquet(edges_path)
    stops.to_parquet(stops_path)
    return graph, edges, stops


def _nearest_nodes(graph: nx.Graph, points: gpd.GeoDataFrame) -> pd.Series:
    node_ids = np.array(list(graph.nodes))
    coords = np.array([(graph.nodes[n]["x"], graph.nodes[n]["y"]) for n in node_ids])
    point_coords = np.column_stack([points.geometry.x.to_numpy(), points.geometry.y.to_numpy()])
    nearest = []
    for start in range(0, len(point_coords), 512):
        batch = point_coords[start : start + 512]
        dist2 = ((batch[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
        nearest.extend(node_ids[np.argmin(dist2, axis=1)].tolist())
    return pd.Series(nearest, index=points.index)


def _nearest_stop_by_super_source(graph: nx.Graph, stop_nodes: list, weight: str):
    super_node = "__stops__"
    graph.add_node(super_node, x=0.0, y=0.0)
    for node in stop_nodes:
        graph.add_edge(super_node, node, baseline_time=0.0, heat_time=0.0, length_m=0.0)
    dist, paths = nx.single_source_dijkstra(graph, super_node, weight=weight)
    graph.remove_node(super_node)
    nearest = {node: path[1] for node, path in paths.items() if node != super_node and len(path) > 1}
    return dist, nearest


def _classify(row: pd.Series) -> str:
    if row["walk_time_min"] <= THRESHOLD_MIN:
        return "ok_walk"
    if not np.isfinite(row["pt_total_min"]):
        return "failed_no_pt_path"
    if row["pt_total_min"] <= THRESHOLD_MIN:
        return "ok_pt_only"
    access_bad = row["pt_access_min"] > THRESHOLD_MIN
    egress_bad = row["pt_egress_min"] > THRESHOLD_MIN
    if access_bad and egress_bad:
        return "failed_access_egress_sum_gt_threshold"
    if access_bad:
        return "failed_access_gt_threshold"
    if egress_bad:
        return "failed_egress_gt_threshold"
    if row["pt_in_vehicle_min"] > THRESHOLD_MIN:
        return "failed_in_vehicle_gt_threshold"
    return "failed_total_gt_threshold_no_single_component_gt_threshold"


def _pt_best_by_origin_stop(graph: nx.Graph, stop_nodes: list, egress_by_stop: np.ndarray) -> dict:
    stop_xy = np.array([(graph.nodes[n]["x"], graph.nodes[n]["y"]) for n in stop_nodes])
    out = {}
    for i, node in enumerate(stop_nodes):
        vehicle = np.linalg.norm(stop_xy - stop_xy[i], axis=1) / PT_M_PER_MIN
        score = vehicle + egress_by_stop
        j = int(np.nanargmin(score))
        out[node] = (float(vehicle[j]), float(egress_by_stop[j]), stop_nodes[j])
    return out


def _compute_one(graph, service_name, stop_nodes, building_nodes, service_nodes, scenario) -> pd.DataFrame:
    weight = "heat_time" if scenario == "heat" else "baseline_time"
    walk_dist = nx.multi_source_dijkstra_path_length(graph, service_nodes, weight=weight)
    stop_dist, nearest_stop = _nearest_stop_by_super_source(graph, stop_nodes, weight)
    stop_egress = np.array([walk_dist.get(node, np.inf) for node in stop_nodes], dtype=float)
    pt_by_stop = _pt_best_by_origin_stop(graph, stop_nodes, stop_egress) if np.isfinite(stop_egress).any() else {}

    rows = []
    for idx, origin_node in building_nodes.items():
        access = float(stop_dist.get(origin_node, np.inf))
        origin_stop = nearest_stop.get(origin_node)
        if origin_stop in pt_by_stop:
            in_vehicle, egress, destination_stop = pt_by_stop[origin_stop]
            pt_total = access + PT_BOARDING_MIN + in_vehicle + egress
        else:
            in_vehicle, egress, destination_stop, pt_total = np.inf, np.inf, None, np.inf
        walk_time = float(walk_dist.get(origin_node, np.inf))
        rows.append(
            {
                "building_idx": int(idx),
                "service_name": service_name,
                "scenario": scenario,
                "origin_node": origin_node,
                "walk_time_min": walk_time,
                "pt_access_min": access,
                "pt_in_vehicle_min": in_vehicle,
                "pt_egress_min": egress,
                "pt_total_min": pt_total,
                "effective_time_min": min(walk_time, pt_total),
                "origin_stop_node": origin_stop,
                "destination_stop_node": destination_stop,
            }
        )
    out = pd.DataFrame(rows)
    out["access_diagnosis_label"] = out.apply(_classify, axis=1)
    return out


def compute_access() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    buildings, services, boundary = _load_inputs()
    graph, edges, stops = _fetch_or_load_osm(boundary)
    stops = stops.loc[~stops.geometry.is_empty].copy()
    building_nodes = _nearest_nodes(graph, buildings)
    stop_nodes = sorted(set(_nearest_nodes(graph, stops).tolist()) & set(graph.nodes))
    print(f"stage=access buildings={len(buildings)} services={sum(len(v) for v in services.values())} stops={len(stop_nodes)} edges={graph.number_of_edges()}")

    rows = []
    for service_name, service_points in services.items():
        service_nodes = sorted(set(_nearest_nodes(graph, service_points).tolist()) & set(graph.nodes))
        for scenario in ["baseline", "heat"]:
            print(f"stage=compute scenario={scenario} service={service_name}")
            rows.append(_compute_one(graph, service_name, stop_nodes, building_nodes, service_nodes, scenario))
    results = pd.concat(rows, ignore_index=True)

    tables = OUT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    buildings.to_parquet(tables / "buildings.parquet")
    stops.to_parquet(tables / "stops.parquet")
    edges.to_parquet(tables / "walk_edges_heat_weighted.parquet")
    results.to_parquet(tables / "service_access_long.parquet")
    _write_comparison(results)
    _render_maps(buildings, services, boundary, edges, stops, results)
    return results


def _write_comparison(results: pd.DataFrame) -> None:
    comp = OUT / "comparison"
    comp.mkdir(parents=True, exist_ok=True)
    base = results.loc[results["scenario"] == "baseline"].copy()
    heat = results.loc[results["scenario"] == "heat"].copy()
    merged = base.merge(heat, on=["building_idx", "service_name"], suffixes=("_baseline", "_heat"))
    merged["delta_effective_min"] = merged["effective_time_min_heat"] - merged["effective_time_min_baseline"]
    merged["delta_walk_min"] = merged["walk_time_min_heat"] - merged["walk_time_min_baseline"]
    merged.to_parquet(comp / "building_service_delta.parquet")

    metric_rows = []
    label_rows = []
    for service in SERVICES:
        sub = merged.loc[merged["service_name"] == service]
        metric_rows.append(
            {
                "service": service,
                "n_buildings": int(len(sub)),
                "median_effective_delta_min": float(sub["delta_effective_min"].replace([np.inf, -np.inf], np.nan).median()),
                "p95_effective_delta_min": float(sub["delta_effective_min"].replace([np.inf, -np.inf], np.nan).quantile(0.95)),
                "median_walk_delta_min": float(sub["delta_walk_min"].replace([np.inf, -np.inf], np.nan).median()),
                "p95_walk_delta_min": float(sub["delta_walk_min"].replace([np.inf, -np.inf], np.nan).quantile(0.95)),
                "label_changed_count": int((sub["access_diagnosis_label_baseline"] != sub["access_diagnosis_label_heat"]).sum()),
                "label_changed_share": float((sub["access_diagnosis_label_baseline"] != sub["access_diagnosis_label_heat"]).mean()),
                "ok_15min_baseline_share": float(sub["access_diagnosis_label_baseline"].isin(["ok_walk", "ok_pt_only"]).mean()),
                "ok_15min_heat_share": float(sub["access_diagnosis_label_heat"].isin(["ok_walk", "ok_pt_only"]).mean()),
                "lost_15min_count": int(
                    (
                        sub["access_diagnosis_label_baseline"].isin(["ok_walk", "ok_pt_only"])
                        & ~sub["access_diagnosis_label_heat"].isin(["ok_walk", "ok_pt_only"])
                    ).sum()
                ),
            }
        )
        for label in LABEL_ORDER:
            baseline_count = int((sub["access_diagnosis_label_baseline"] == label).sum())
            heat_count = int((sub["access_diagnosis_label_heat"] == label).sum())
            label_rows.append({"service": service, "label": label, "count_baseline": baseline_count, "count_heat": heat_count, "delta_count": heat_count - baseline_count})
        (
            sub.groupby(["access_diagnosis_label_baseline", "access_diagnosis_label_heat"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .to_csv(comp / f"{service}_label_transitions.csv", index=False)
        )
    pd.DataFrame(metric_rows).to_csv(comp / "metric_summary.csv", index=False)
    pd.DataFrame(label_rows).to_csv(comp / "label_summary.csv", index=False)
    (comp / "metric_summary.json").write_text(json.dumps(metric_rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _plot_base(ax, boundary: gpd.GeoDataFrame, edges: gpd.GeoDataFrame) -> None:
    boundary.plot(ax=ax, facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.0)
    edges.plot(ax=ax, color="#94a3b8", linewidth=0.22, alpha=0.34)
    minx, miny, maxx, maxy = boundary.total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def _legend_handles():
    return [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LABEL_COLORS[label], markeredgecolor="none", markersize=8, label=LABEL_RU[label])
        for label in LABEL_ORDER
    ]


def _render_label_map(buildings, boundary, edges, stops, service_points, sub, title, path):
    points = buildings[["building_idx", "geometry"]].merge(sub[["building_idx", "access_diagnosis_label"]], on="building_idx", how="left")
    points = gpd.GeoDataFrame(points, geometry="geometry", crs=buildings.crs)
    fig, ax = plt.subplots(figsize=(9, 9), dpi=220)
    _plot_base(ax, boundary, edges)
    stops.plot(ax=ax, color="#2563eb", markersize=5, alpha=0.55)
    service_points.plot(ax=ax, color="#111827", markersize=26, marker="*", alpha=0.9)
    for label in PLOT_ORDER:
        pts = points.loc[points["access_diagnosis_label"] == label]
        if not pts.empty:
            ax.scatter(pts.geometry.x, pts.geometry.y, s=4.0, c=LABEL_COLORS[label], alpha=0.75, linewidths=0, rasterized=True)
    ax.legend(handles=_legend_handles(), loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=9)
    ax.set_title(title, fontsize=14, pad=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _render_delta_map(buildings, boundary, edges, stops, service_points, merged, service, path):
    points = buildings[["building_idx", "geometry"]].merge(
        merged.loc[merged["service_name"] == service, ["building_idx", "delta_effective_min"]],
        on="building_idx",
        how="left",
    )
    points = gpd.GeoDataFrame(points, geometry="geometry", crs=buildings.crs)
    finite = points["delta_effective_min"].replace([np.inf, -np.inf], np.nan)
    vmax = max(1.0, float(finite.quantile(0.98)))
    fig, ax = plt.subplots(figsize=(9, 9), dpi=220)
    _plot_base(ax, boundary, edges)
    stops.plot(ax=ax, color="#2563eb", markersize=5, alpha=0.35)
    service_points.plot(ax=ax, color="#111827", markersize=26, marker="*", alpha=0.9)
    zero = points.loc[finite.fillna(0).abs() < 0.05]
    nonzero = points.loc[finite.fillna(0).abs() >= 0.05]
    if not zero.empty:
        ax.scatter(zero.geometry.x, zero.geometry.y, s=3.5, c="#d1d5db", alpha=0.55, linewidths=0, rasterized=True, label="≈ 0 мин")
    sc = ax.scatter(
        nonzero.geometry.x,
        nonzero.geometry.y,
        s=4.2,
        c=nonzero["delta_effective_min"],
        cmap="magma_r",
        vmin=0,
        vmax=vmax,
        alpha=0.82,
        linewidths=0,
        rasterized=True,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.72)
    cbar.set_label("изменение лучшего времени до сервиса, мин")
    ax.set_title(f"Madrid 2×2 км — {SERVICE_RU[service]} — heat minus baseline", fontsize=14, pad=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _render_maps(buildings, services, boundary, edges, stops, results) -> None:
    maps = OUT / "maps"
    maps.mkdir(parents=True, exist_ok=True)
    comp = gpd.read_parquet(OUT / "comparison" / "building_service_delta.parquet")
    for service in SERVICES:
        service_points = services[service]
        for scenario in ["baseline", "heat"]:
            sub = results.loc[(results["service_name"] == service) & (results["scenario"] == scenario)]
            _render_label_map(
                buildings,
                boundary,
                edges,
                stops,
                service_points,
                sub,
                f"Madrid 2×2 км — {SERVICE_RU[service]} — {'с учетом жары' if scenario == 'heat' else 'baseline'}",
                maps / f"{service}_{scenario}.png",
            )
        _render_delta_map(buildings, boundary, edges, stops, service_points, comp, service, maps / f"{service}_delta.png")


def main() -> None:
    compute_access()


if __name__ == "__main__":
    main()
