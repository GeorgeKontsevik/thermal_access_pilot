from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
from matplotlib.lines import Line2D
from pyproj import Transformer
from shapely.geometry import Point, box

from aggregated_spatial_pipeline.blocksnet_data_pipeline.pipeline import _get_buildings_raw, _get_pipeline2_services_raw
from aggregated_spatial_pipeline.pipeline.run_pipeline2_prepare_solver_inputs import _extract_service_raw_from_common

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPUTE_PYTHON = REPO_ROOT / "thermal_access_pilot" / ".venv" / "bin" / "python"
OLD_MADRID_DIR = REPO_ROOT / "aggregated_spatial_pipeline" / "outputs" / "old" / "joint_inputs" / "madrid_spain"
UTCI_ROOT = REPO_ROOT / "thermal_access_pilot" / "outputs" / "madrid_utci_5km"
OUT_ROOT = REPO_ROOT / "thermal_access_pilot" / "outputs" / "madrid_service_access_2km"
CITY = "madrid_spain"
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
    core_radius_m: float = 1000.0
    crs: str = "EPSG:25830"


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


def _study_box(spec: MadridSpec) -> gpd.GeoDataFrame:
    tr = Transformer.from_crs("EPSG:4326", spec.crs, always_xy=True)
    x, y = tr.transform(spec.center_lon, spec.center_lat)
    geom = box(x - spec.core_radius_m, y - spec.core_radius_m, x + spec.core_radius_m, y + spec.core_radius_m)
    return gpd.GeoDataFrame({"name": [CITY]}, geometry=[geom], crs=spec.crs)


def _clip_buildings(spec: MadridSpec, out_path: Path) -> None:
    study = _study_box(spec)
    buildings = _get_buildings_raw(study.to_crs(4326).geometry.iloc[0])
    clipped = gpd.clip(buildings.to_crs(spec.crs), study).to_crs(4326).reset_index(drop=True)
    clipped["is_living"] = 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clipped.to_parquet(out_path)


def _write_boundary(spec: MadridSpec, boundary_path: Path) -> None:
    boundary = _study_box(spec).to_crs(4326)
    boundary_path.parent.mkdir(parents=True, exist_ok=True)
    boundary.to_parquet(boundary_path)


def _write_services_raw(spec: MadridSpec, out_dir: Path) -> None:
    study_geom = _study_box(spec).to_crs(4326).geometry.iloc[0]
    common = _get_pipeline2_services_raw(study_geom)
    common_path = out_dir.parent.parent / "blocksnet_raw_osm" / "services_pipeline2_raw.parquet"
    common_path.parent.mkdir(parents=True, exist_ok=True)
    common.to_parquet(common_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    for service in SERVICES:
        raw = _extract_service_raw_from_common(common, service)
        raw.to_parquet(out_dir / f"{service}.parquet")


def _build_graph_bundle(city_dir: Path) -> None:
    env = os.environ.copy()
    endpoint = "https://maps.mail.ru/osm/tools/overpass/api/interpreter"
    env.update(
        {
            "PYTHONPATH": str(REPO_ROOT),
            "OVERPASS_URL": endpoint,
            "OVERPASS_USER_AGENT": "super-duper-disser/1.0 (research)",
            "OVERPASS_TIMEOUT": "240",
            "OVERPASS_MAX_RETRIES": "2",
        }
    )
    subprocess.run(
        [
            str(REPO_ROOT / "iduedu-fork" / ".venv" / "bin" / "python"),
            "-m",
            "aggregated_spatial_pipeline.intermodal_graph_data_pipeline.build_bundle_external",
            "--place",
            "Madrid, Spain",
            "--boundary-path",
            str(city_dir / "analysis_territory" / "buffer.parquet"),
            "--output-dir",
            str(city_dir / "intermodal_graph_iduedu"),
            "--overpass-url",
            endpoint,
            "--osm-timeout-s",
            "240",
            "--no-extra-stop-tags",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )


def _sample_factor(line, src: rasterio.io.DatasetReader, step_m: float = 20.0) -> float:
    if line is None or line.is_empty:
        return 1.0
    length = float(line.length)
    sample_count = max(3, int(np.ceil(length / step_m)) + 1)
    distances = np.linspace(0.0, length, sample_count)
    coords = [(line.interpolate(dist).x, line.interpolate(dist).y) for dist in distances]
    vals = []
    for item in src.sample(coords):
        value = float(item[0])
        if np.isfinite(value):
            vals.append(UTCI_COST_FACTORS[utci_category(value)])
    return float(np.mean(vals)) if vals else 1.0


def _build_heat_graph(bundle_dir: Path, raster_path: Path | None = None) -> None:
    edges_path = bundle_dir / "intermodal_graph_iduedu" / "graph_edges.parquet"
    graph_path = bundle_dir / "intermodal_graph_iduedu" / "graph.pkl"
    utci_path = raster_path or UTCI_ROOT / "thermal" / "headline_utci.tif"

    edges = gpd.read_parquet(edges_path)
    walk = edges.loc[edges["type"].astype(str).str.lower() == "walk"].copy()
    if walk.empty:
        return
    with rasterio.open(utci_path) as src:
        walk_metric = walk.to_crs(src.crs)
        walk["utci_factor"] = walk_metric.geometry.map(lambda geom: _sample_factor(geom, src))
    pair_factor = walk.groupby(["u", "v"])["utci_factor"].mean().to_dict()
    edges["utci_factor"] = 1.0
    edges.loc[walk.index, "utci_factor"] = walk["utci_factor"].to_numpy()
    edges.loc[walk.index, "length_meter"] = (
        pd.to_numeric(edges.loc[walk.index, "length_meter"], errors="coerce").fillna(0.0)
        * edges.loc[walk.index, "utci_factor"]
    )
    edges.to_parquet(edges_path)

    with graph_path.open("rb") as fh:
        graph: nx.MultiDiGraph = pickle.load(fh)
    for u, v, key, data in graph.edges(keys=True, data=True):
        if str(data.get("type", "")).lower() != "walk":
            continue
        factor = float(pair_factor.get((u, v), 1.0))
        data["utci_factor"] = factor
        data["time_min"] = float(data.get("time_min", 0.0) or 0.0) * factor
    with graph_path.open("wb") as fh:
        pickle.dump(graph, fh)


def _run_command(args: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    subprocess.run(args, cwd=REPO_ROOT, env=env, check=True)


def _bundle_root(scenario: str) -> Path:
    return OUT_ROOT / f"joint_inputs_{scenario}"


def _city_dir(scenario: str) -> Path:
    return _bundle_root(scenario) / CITY


def build_bundles() -> None:
    spec = MadridSpec()
    city_dir = _city_dir("baseline")
    _write_boundary(spec, city_dir / "blocksnet" / "boundary.parquet")
    _write_boundary(spec, city_dir / "analysis_territory" / "buffer.parquet")
    _write_boundary(spec, city_dir / "analysis_territory" / "buffer_collection.parquet")
    _clip_buildings(spec, city_dir / "derived_layers" / "buildings_floor_enriched.parquet")
    _write_services_raw(spec, city_dir / "pipeline_2" / "services_raw")
    _build_graph_bundle(city_dir)
    heat_city = _city_dir("heat")
    if heat_city.exists():
        shutil.rmtree(heat_city)
    shutil.copytree(city_dir, heat_city)
    _build_heat_graph(_city_dir("heat"))


def run_existing_scripts() -> None:
    empty_pt_root = OUT_ROOT / "empty_pt_root"
    empty_pt_root.mkdir(parents=True, exist_ok=True)
    for scenario in ["baseline", "heat"]:
        bundle_root = _bundle_root(scenario)
        walk_root = OUT_ROOT / f"walk_{scenario}"
        pt_root = OUT_ROOT / f"pt_{scenario}"
        diag_root = OUT_ROOT / f"diag_{scenario}"
        _run_command(
            [
                str(COMPUTE_PYTHON),
                "scripts/run_residential_to_services_top1.py",
                "--joint-inputs-root",
                str(bundle_root),
                "--out-root",
                str(walk_root),
                "--cities",
                CITY,
                "--services",
                *SERVICES,
            ]
        )
        _run_command(
            [
                str(COMPUTE_PYTHON),
                "scripts/run_residential_to_services_pt_top1.py",
                "--joint-inputs-root",
                str(bundle_root),
                "--walk-root",
                str(walk_root),
                "--out-root",
                str(pt_root),
                "--cities",
                CITY,
                "--services",
                *SERVICES,
                "--min-walk-min",
                "15",
            ]
        )
        _run_command(
            [
                str(COMPUTE_PYTHON),
                "scripts/classify_service_access_failures.py",
                "--walk-root",
                str(walk_root),
                "--pt-walk-lt-root",
                str(empty_pt_root),
                "--pt-walk-ge-root",
                str(pt_root),
                "--joint-inputs-root",
                str(bundle_root),
                "--out-root",
                str(diag_root),
                "--threshold-min",
                "15",
                "--cities",
                CITY,
            ]
        )


def _plot_boundary(ax, boundary: gpd.GeoDataFrame) -> None:
    boundary.plot(ax=ax, facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=0.95)
    minx, miny, maxx, maxy = boundary.to_crs(boundary.estimate_utm_crs()).total_bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def _render_service_map(diag_path: Path, scenario: str, service: str, out_path: Path) -> None:
    diag = pd.read_parquet(diag_path)
    sub = diag.loc[diag["service_name"] == service].copy()
    buildings = gpd.read_parquet(_city_dir(scenario) / "derived_layers" / "buildings_floor_enriched.parquet")
    if "is_living" in buildings.columns:
        mask = pd.to_numeric(buildings["is_living"], errors="coerce").fillna(0).astype(float) > 0
        buildings = buildings.loc[mask].copy()
    buildings = buildings.reset_index(drop=False).rename(columns={"index": "building_idx"})
    buildings["geometry"] = buildings.geometry.representative_point()
    points = gpd.GeoDataFrame(
        buildings[["building_idx", "geometry"]].merge(sub[["building_idx", "access_diagnosis_label"]], on="building_idx", how="inner"),
        geometry="geometry",
        crs=buildings.crs,
    ).to_crs("EPSG:25830")
    boundary = gpd.read_parquet(_city_dir(scenario) / "blocksnet" / "boundary.parquet").to_crs("EPSG:25830")

    fig, ax = plt.subplots(figsize=(10, 10), dpi=220)
    _plot_boundary(ax, boundary)
    for label in PLOT_ORDER:
        pts = points.loc[points["access_diagnosis_label"] == label]
        if pts.empty:
            continue
        ax.scatter(pts.geometry.x, pts.geometry.y, s=5.0, c=LABEL_COLORS[label], alpha=0.72, linewidths=0, rasterized=True)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=LABEL_COLORS[label],
            markeredgecolor="none",
            markersize=9,
            label=LABEL_RU[label],
        )
        for label in LABEL_ORDER
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=11)
    ax.set_title(f"Madrid 2×2 км — {SERVICE_RU[service]} — {'heat-aware' if scenario == 'heat' else 'baseline'}", fontsize=16, pad=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_maps() -> None:
    maps_dir = OUT_ROOT / "maps"
    for scenario in ["baseline", "heat"]:
        diag_path = OUT_ROOT / f"diag_{scenario}" / CITY / "home_to_service_access_diagnostics.parquet"
        for service in SERVICES:
            _render_service_map(diag_path, scenario, service, maps_dir / f"{service}_{scenario}.png")


def compare_results() -> None:
    compare_dir = OUT_ROOT / "comparison"
    compare_dir.mkdir(parents=True, exist_ok=True)
    base = pd.read_parquet(OUT_ROOT / "diag_baseline" / CITY / "home_to_service_access_diagnostics.parquet")
    heat = pd.read_parquet(OUT_ROOT / "diag_heat" / CITY / "home_to_service_access_diagnostics.parquet")
    merged = base.merge(
        heat,
        on=["building_idx", "service_name"],
        suffixes=("_baseline", "_heat"),
    )
    label_summary_rows = []
    metric_rows = []
    for service in SERVICES:
        sub = merged.loc[merged["service_name"] == service].copy()
        labels = pd.DataFrame({"label": LABEL_ORDER})
        base_counts = sub["access_diagnosis_label_baseline"].value_counts().rename_axis("label").reset_index(name="count_baseline")
        heat_counts = sub["access_diagnosis_label_heat"].value_counts().rename_axis("label").reset_index(name="count_heat")
        summary = labels.merge(base_counts, on="label", how="left").merge(heat_counts, on="label", how="left").fillna(0)
        summary["service"] = service
        summary["delta_count"] = summary["count_heat"] - summary["count_baseline"]
        summary["share_baseline"] = summary["count_baseline"] / len(sub)
        summary["share_heat"] = summary["count_heat"] / len(sub)
        summary["delta_share"] = summary["share_heat"] - summary["share_baseline"]
        label_summary_rows.append(summary)

        metric_rows.append(
            {
                "service": service,
                "n_buildings": int(len(sub)),
                "label_changed_share": float((sub["access_diagnosis_label_baseline"] != sub["access_diagnosis_label_heat"]).mean()),
                "median_walk_time_baseline": float(pd.to_numeric(sub["walk_time_min_baseline"], errors="coerce").median()),
                "median_walk_time_heat": float(pd.to_numeric(sub["walk_time_min_heat"], errors="coerce").median()),
                "median_walk_delta": float((pd.to_numeric(sub["walk_time_min_heat"], errors="coerce") - pd.to_numeric(sub["walk_time_min_baseline"], errors="coerce")).median()),
                "median_pt_total_baseline": float(pd.to_numeric(sub["effective_pt_total_min_baseline"], errors="coerce").replace([np.inf, -np.inf], np.nan).median()),
                "median_pt_total_heat": float(pd.to_numeric(sub["effective_pt_total_min_heat"], errors="coerce").replace([np.inf, -np.inf], np.nan).median()),
                "median_pt_total_delta": float((pd.to_numeric(sub["effective_pt_total_min_heat"], errors="coerce").replace([np.inf, -np.inf], np.nan) - pd.to_numeric(sub["effective_pt_total_min_baseline"], errors="coerce").replace([np.inf, -np.inf], np.nan)).median()),
                "ok_walk_share_baseline": float((sub["access_diagnosis_label_baseline"] == "ok_walk").mean()),
                "ok_walk_share_heat": float((sub["access_diagnosis_label_heat"] == "ok_walk").mean()),
                "ok_pt_only_share_baseline": float((sub["access_diagnosis_label_baseline"] == "ok_pt_only").mean()),
                "ok_pt_only_share_heat": float((sub["access_diagnosis_label_heat"] == "ok_pt_only").mean()),
            }
        )

        transition = (
            sub.groupby(["access_diagnosis_label_baseline", "access_diagnosis_label_heat"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        transition["service"] = service
        transition.to_csv(compare_dir / f"{service}_label_transitions.csv", index=False)

    pd.concat(label_summary_rows, ignore_index=True).to_csv(compare_dir / "label_summary.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(compare_dir / "metric_summary.csv", index=False)
    (compare_dir / "metric_summary.json").write_text(
        pd.DataFrame(metric_rows).to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    build_bundles()
    run_existing_scripts()
    render_maps()
    compare_results()


if __name__ == "__main__":
    main()
