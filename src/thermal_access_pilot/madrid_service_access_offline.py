from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.lines import Line2D
from shapely.geometry import LineString, Point

REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = REPO_ROOT / "thermal_access_pilot" / "outputs" / "madrid_service_access_2km"
CITY_DIR = ROOT / "joint_inputs_baseline" / "madrid_spain"
UTCI_ROOT = REPO_ROOT / "thermal_access_pilot" / "outputs" / "madrid_utci_5km"
OUT = ROOT / "offline_service_access"
CRS = "EPSG:25830"
WALK_M_PER_MIN = 84.0
WALK_NETWORK_FACTOR = 1.25
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


def _coords(gdf: gpd.GeoDataFrame) -> np.ndarray:
    return np.column_stack([gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()])


def _nearest_row_indices(source_xy: np.ndarray, target_xy: np.ndarray, batch_size: int = 512) -> np.ndarray:
    out = []
    for start in range(0, len(source_xy), batch_size):
        batch = source_xy[start : start + batch_size]
        dist2 = ((batch[:, None, :] - target_xy[None, :, :]) ** 2).sum(axis=2)
        out.extend(np.argmin(dist2, axis=1).tolist())
    return np.asarray(out, dtype=int)


def _pair_factor(a_xy: np.ndarray, b_xy: np.ndarray) -> np.ndarray:
    mids = (a_xy + b_xy) / 2.0
    with rasterio.open(UTCI_ROOT / "thermal" / "headline_utci.tif") as src:
        values = [float(v[0]) for v in src.sample([(float(x), float(y)) for x, y in mids])]
    return np.asarray([UTCI_FACTORS[utci_category(v)] if np.isfinite(v) else 1.0 for v in values], dtype=float)


def _min_walk_to_services(origin_xy: np.ndarray, service_xy: np.ndarray, scenario: str) -> tuple[np.ndarray, np.ndarray]:
    best_time = np.full(len(origin_xy), np.inf)
    best_idx = np.full(len(origin_xy), -1, dtype=int)
    for start in range(0, len(origin_xy), 512):
        batch = origin_xy[start : start + 512]
        dist = np.linalg.norm(batch[:, None, :] - service_xy[None, :, :], axis=2) * WALK_NETWORK_FACTOR
        if scenario == "heat":
            factors = []
            for service in service_xy:
                factors.append(_pair_factor(batch, np.repeat(service[None, :], len(batch), axis=0)))
            dist = dist * np.column_stack(factors)
        time = dist / WALK_M_PER_MIN
        idx = np.argmin(time, axis=1)
        rows = np.arange(len(batch))
        best_time[start : start + len(batch)] = time[rows, idx]
        best_idx[start : start + len(batch)] = idx
    return best_time, best_idx


def _load_inputs():
    buildings_poly = gpd.read_parquet(CITY_DIR / "derived_layers" / "buildings_floor_enriched.parquet").to_crs(CRS).reset_index(drop=True)
    buildings = buildings_poly.copy()
    buildings["building_idx"] = buildings.index.astype(int)
    buildings["geometry"] = buildings.geometry.representative_point()
    services = {}
    for service in SERVICES:
        gdf = gpd.read_parquet(CITY_DIR / "pipeline_2" / "services_raw" / f"{service}.parquet").to_crs(CRS).reset_index(drop=True)
        gdf["geometry"] = gdf.geometry.representative_point()
        services[service] = gdf
    boundary = gpd.read_parquet(CITY_DIR / "analysis_territory" / "buffer.parquet").to_crs(CRS)
    routes = gpd.read_parquet(UTCI_ROOT / "tables" / "routes.parquet").to_crs(CRS)
    return buildings, buildings_poly, services, boundary, routes


def _route_points(routes: gpd.GeoDataFrame):
    rows = []
    for _, row in routes.iterrows():
        coords = list(row.geometry.coords)
        if len(coords) < 2:
            continue
        rows.append(
            {
                "origin_node": row["origin_node"],
                "scenario": row["scenario"],
                "stop_node": row["stop_node"],
                "time_min": row["time_min"],
                "generalized_time_min": row["generalized_time_min"],
                "origin_geometry": Point(coords[0]),
                "stop_geometry": Point(coords[-1]),
            }
        )
    df = pd.DataFrame(rows)
    base = df.loc[df["scenario"] == "baseline"].copy()
    heat = df.loc[df["scenario"] == "utci"].copy()
    merged = base.merge(
        heat[["origin_node", "stop_node", "generalized_time_min"]],
        on=["origin_node", "stop_node"],
        how="left",
        suffixes=("_baseline", "_heat"),
    )
    origins = gpd.GeoDataFrame(merged, geometry="origin_geometry", crs=CRS)
    stops = (
        gpd.GeoDataFrame(df[["stop_node", "stop_geometry"]].drop_duplicates("stop_node"), geometry="stop_geometry", crs=CRS)
        .reset_index(drop=True)
        .rename_geometry("geometry")
    )
    return origins, stops


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


def compute_access() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    buildings, buildings_poly, services, boundary, routes = _load_inputs()
    origins, stops = _route_points(routes)
    bxy = _coords(buildings)
    origin_idx = _nearest_row_indices(bxy, _coords(origins))
    snapped = origins.iloc[origin_idx].reset_index(drop=True)
    buildings["origin_node"] = snapped["origin_node"].to_numpy()
    buildings["origin_stop_node"] = snapped["stop_node"].to_numpy()
    buildings["access_baseline_min"] = snapped["time_min"].to_numpy()
    buildings["access_heat_min"] = snapped["generalized_time_min_heat"].fillna(snapped["generalized_time_min_baseline"]).to_numpy()

    stop_xy = _coords(stops)
    rows = []
    for service_name, service_gdf in services.items():
        service_xy = _coords(service_gdf)
        for scenario in ["baseline", "heat"]:
            print(f"stage=offline_compute scenario={scenario} service={service_name}")
            direct_walk, _ = _min_walk_to_services(bxy, service_xy, scenario)
            egress, _ = _min_walk_to_services(stop_xy, service_xy, scenario)
            stop_best = {}
            for i, stop_node in enumerate(stops["stop_node"].to_numpy()):
                vehicle = np.linalg.norm(stop_xy - stop_xy[i], axis=1) / PT_M_PER_MIN
                score = vehicle + egress
                j = int(np.nanargmin(score))
                stop_best[stop_node] = (float(vehicle[j]), float(egress[j]), stops["stop_node"].iloc[j])
            access_col = "access_heat_min" if scenario == "heat" else "access_baseline_min"
            for idx, b in buildings.iterrows():
                in_vehicle, egress_min, dest_stop = stop_best.get(b["origin_stop_node"], (np.inf, np.inf, None))
                pt_total = float(b[access_col]) + PT_BOARDING_MIN + in_vehicle + egress_min
                row = {
                    "building_idx": int(idx),
                    "service_name": service_name,
                    "scenario": scenario,
                    "walk_time_min": float(direct_walk[idx]),
                    "pt_access_min": float(b[access_col]),
                    "pt_in_vehicle_min": in_vehicle,
                    "pt_egress_min": egress_min,
                    "pt_total_min": pt_total,
                    "effective_time_min": min(float(direct_walk[idx]), pt_total),
                    "origin_stop_node": b["origin_stop_node"],
                    "destination_stop_node": dest_stop,
                }
                row["access_diagnosis_label"] = _classify(pd.Series(row))
                rows.append(row)
    results = pd.DataFrame(rows)
    tables = OUT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    buildings.to_parquet(tables / "buildings_points.parquet")
    buildings_poly.to_parquet(tables / "buildings_polygons.parquet")
    stops.to_parquet(tables / "stops_from_existing_routes.parquet")
    results.to_parquet(tables / "service_access_long.parquet")
    _write_comparison(results)
    _render_maps(buildings, buildings_poly, services, boundary, stops, routes, results)
    _write_manifest(buildings, services, stops, routes)
    return results


def _write_manifest(buildings, services, stops, routes) -> None:
    manifest = {
        "method": "offline Madrid 2km service accessibility; existing Madrid UTCI building-to-stop routes supply PT access leg; heat applied only to walking legs",
        "buildings": int(len(buildings)),
        "services": {k: int(len(v)) for k, v in services.items()},
        "route_origin_rows": int(routes["origin_node"].nunique()),
        "stops_from_existing_routes": int(stops["stop_node"].nunique()),
        "threshold_min": THRESHOLD_MIN,
        "walk_network_factor_for_non-routed_service_legs": WALK_NETWORK_FACTOR,
        "pt_vehicle_speed_m_per_min": PT_M_PER_MIN,
        "pt_boarding_min": PT_BOARDING_MIN,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_comparison(results: pd.DataFrame) -> None:
    comp = OUT / "comparison"
    comp.mkdir(parents=True, exist_ok=True)
    base = results.loc[results["scenario"] == "baseline"].copy()
    heat = results.loc[results["scenario"] == "heat"].copy()
    merged = base.merge(heat, on=["building_idx", "service_name"], suffixes=("_baseline", "_heat"))
    merged["delta_effective_min"] = merged["effective_time_min_heat"] - merged["effective_time_min_baseline"]
    merged["delta_walk_min"] = merged["walk_time_min_heat"] - merged["walk_time_min_baseline"]
    merged["delta_pt_total_min"] = merged["pt_total_min_heat"] - merged["pt_total_min_baseline"]
    merged.to_parquet(comp / "building_service_delta.parquet")

    metric_rows = []
    label_rows = []
    for service in SERVICES:
        sub = merged.loc[merged["service_name"] == service]
        ok_base = sub["access_diagnosis_label_baseline"].isin(["ok_walk", "ok_pt_only"])
        ok_heat = sub["access_diagnosis_label_heat"].isin(["ok_walk", "ok_pt_only"])
        metric_rows.append(
            {
                "service": service,
                "n_buildings": int(len(sub)),
                "median_effective_delta_min": float(sub["delta_effective_min"].replace([np.inf, -np.inf], np.nan).median()),
                "p95_effective_delta_min": float(sub["delta_effective_min"].replace([np.inf, -np.inf], np.nan).quantile(0.95)),
                "median_walk_delta_min": float(sub["delta_walk_min"].replace([np.inf, -np.inf], np.nan).median()),
                "p95_walk_delta_min": float(sub["delta_walk_min"].replace([np.inf, -np.inf], np.nan).quantile(0.95)),
                "median_pt_delta_min": float(sub["delta_pt_total_min"].replace([np.inf, -np.inf], np.nan).median()),
                "p95_pt_delta_min": float(sub["delta_pt_total_min"].replace([np.inf, -np.inf], np.nan).quantile(0.95)),
                "label_changed_count": int((sub["access_diagnosis_label_baseline"] != sub["access_diagnosis_label_heat"]).sum()),
                "label_changed_share": float((sub["access_diagnosis_label_baseline"] != sub["access_diagnosis_label_heat"]).mean()),
                "ok_15min_baseline_share": float(ok_base.mean()),
                "ok_15min_heat_share": float(ok_heat.mean()),
                "lost_15min_count": int((ok_base & ~ok_heat).sum()),
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


def _plot_base(ax, boundary, routes):
    boundary.plot(ax=ax, facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.0)
    routes.loc[routes["scenario"] == "baseline"].plot(ax=ax, color="#94a3b8", linewidth=0.22, alpha=0.25)
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


def _render_label_map(buildings, services, boundary, stops, routes, sub, title, path):
    points = buildings[["building_idx", "geometry"]].merge(sub[["building_idx", "access_diagnosis_label"]], on="building_idx", how="left")
    points = gpd.GeoDataFrame(points, geometry="geometry", crs=buildings.crs)
    fig, ax = plt.subplots(figsize=(9, 9), dpi=220)
    _plot_base(ax, boundary, routes)
    stops.plot(ax=ax, color="#2563eb", markersize=4, alpha=0.35)
    services.plot(ax=ax, color="#111827", markersize=30, marker="*", alpha=0.92)
    for label in PLOT_ORDER:
        pts = points.loc[points["access_diagnosis_label"] == label]
        if not pts.empty:
            ax.scatter(pts.geometry.x, pts.geometry.y, s=4.0, c=LABEL_COLORS[label], alpha=0.75, linewidths=0, rasterized=True)
    ax.legend(handles=_legend_handles(), loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=9)
    ax.set_title(title, fontsize=14, pad=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _render_delta_map(buildings, services, boundary, stops, routes, merged, service, path):
    points = buildings[["building_idx", "geometry"]].merge(
        merged.loc[merged["service_name"] == service, ["building_idx", "delta_effective_min"]],
        on="building_idx",
        how="left",
    )
    points = gpd.GeoDataFrame(points, geometry="geometry", crs=buildings.crs)
    finite = points["delta_effective_min"].replace([np.inf, -np.inf], np.nan)
    vmax = max(1.0, float(finite.quantile(0.98)))
    fig, ax = plt.subplots(figsize=(9, 9), dpi=220)
    _plot_base(ax, boundary, routes)
    stops.plot(ax=ax, color="#2563eb", markersize=4, alpha=0.25)
    services.plot(ax=ax, color="#111827", markersize=30, marker="*", alpha=0.92)
    zero = points.loc[finite.fillna(0).abs() < 0.05]
    nonzero = points.loc[finite.fillna(0).abs() >= 0.05]
    if not zero.empty:
        ax.scatter(zero.geometry.x, zero.geometry.y, s=3.5, c="#d1d5db", alpha=0.55, linewidths=0, rasterized=True)
    sc = ax.scatter(nonzero.geometry.x, nonzero.geometry.y, s=4.2, c=nonzero["delta_effective_min"], cmap="magma_r", vmin=0, vmax=vmax, alpha=0.82, linewidths=0, rasterized=True)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.72)
    cbar.set_label("изменение лучшего времени до сервиса, мин")
    ax.set_title(f"Madrid 2×2 км — {SERVICE_RU[service]} — heat minus baseline", fontsize=14, pad=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _render_maps(buildings, _buildings_poly, services, boundary, stops, routes, results) -> None:
    maps = OUT / "maps"
    maps.mkdir(parents=True, exist_ok=True)
    merged = pd.read_parquet(OUT / "comparison" / "building_service_delta.parquet")
    routes_clip = gpd.clip(routes, boundary)
    for service in SERVICES:
        for scenario in ["baseline", "heat"]:
            sub = results.loc[(results["service_name"] == service) & (results["scenario"] == scenario)]
            _render_label_map(
                buildings,
                services[service],
                boundary,
                stops,
                routes_clip,
                sub,
                f"Madrid 2×2 км — {SERVICE_RU[service]} — {'с учетом жары' if scenario == 'heat' else 'baseline'}",
                maps / f"{service}_{scenario}.png",
            )
        _render_delta_map(buildings, services[service], boundary, stops, routes_clip, merged, service, maps / f"{service}_delta.png")


def main() -> None:
    compute_access()


if __name__ == "__main__":
    main()
