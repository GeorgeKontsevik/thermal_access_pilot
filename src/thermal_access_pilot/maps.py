from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
from PIL import Image
from rasterio.plot import show

from .config import PilotConfig
from .local_inputs import LocalInputs


REQUIRED_MAPS = [
    "01_inputs.png",
    "02_thermal_fields.png",
    "03_routes_examples.png",
    "04_building_exposure.png",
    "05_time_change.png",
    "06_sensitivity.png",
]


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_raster(ax, path: Path, title: str, cmap: str = "inferno"):
    with rasterio.open(path) as src:
        show(src.read(1), transform=src.transform, ax=ax, cmap=cmap)
    ax.set_title(title)
    ax.set_axis_off()


def render_maps(cfg: PilotConfig, local: LocalInputs, paths: dict[str, Path], results: pd.DataFrame, routes: gpd.GeoDataFrame) -> None:
    maps_dir = cfg.output_dir / "maps"
    buildings = local.buildings.loc[local.buildings.intersects(local.core)]

    fig, ax = plt.subplots(figsize=(10, 8))
    gpd.GeoSeries([local.model_area], crs=cfg.crs).boundary.plot(ax=ax, color="black", linewidth=1)
    gpd.GeoSeries([local.core], crs=cfg.crs).boundary.plot(ax=ax, color="red", linewidth=1)
    buildings.plot(ax=ax, color="#777777", markersize=1, alpha=0.6)
    local.stops.plot(ax=ax, color="blue", markersize=12)
    ax.set_title("Inputs: buildings, core, model area, PT stops")
    ax.set_axis_off()
    _save(fig, maps_dir / "01_inputs.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    _plot_raster(axes[0], paths["utci"], "UTCI °C", "magma")
    _plot_raster(axes[1], paths["tmrt"], "Tmrt °C", "inferno")
    _plot_raster(axes[2], paths["hot_mask"], f"Hot mask > {cfg.hot_threshold_c:g} °C", "Reds")
    _save(fig, maps_dir / "02_thermal_fields.png")

    base_routes = routes.loc[routes["penalty"] == 0].head(40)
    heat_routes = routes.loc[routes["penalty"] == 0.5].head(40)
    fig, ax = plt.subplots(figsize=(10, 8))
    _plot_raster(ax, paths["utci"], "Example routes: base gray, heat-aware cyan", "magma")
    base_routes.plot(ax=ax, color="white", linewidth=0.8, alpha=0.6)
    heat_routes.plot(ax=ax, color="cyan", linewidth=0.8, alpha=0.8)
    local.stops.plot(ax=ax, color="blue", markersize=8)
    _save(fig, maps_dir / "03_routes_examples.png")

    joined = buildings[["building_id", "geometry"]].merge(results.loc[results["penalty"] == 0, ["building_id", "hot_fraction"]], on="building_id", how="left")
    fig, ax = plt.subplots(figsize=(10, 8))
    joined.plot(ax=ax, column="hot_fraction", cmap="Reds", legend=True, missing_kwds={"color": "lightgray"})
    ax.set_title("Building origins by baseline route hot fraction")
    ax.set_axis_off()
    _save(fig, maps_dir / "04_building_exposure.png")

    penalty = 0.5 if 0.5 in set(results["penalty"]) else max(results["penalty"])
    base = results.loc[results["penalty"] == 0, ["building_id", "time_min"]].rename(columns={"time_min": "base_time_min"})
    changed = results.loc[results["penalty"] == penalty, ["building_id", "generalized_time_min"]].merge(base, on="building_id")
    changed["delta_min"] = changed["generalized_time_min"] - changed["base_time_min"]
    joined = buildings[["building_id", "geometry"]].merge(changed[["building_id", "delta_min"]], on="building_id", how="left")
    fig, ax = plt.subplots(figsize=(10, 8))
    joined.plot(ax=ax, column="delta_min", cmap="viridis", legend=True, missing_kwds={"color": "lightgray"})
    ax.set_title(f"Generalized walking-time increase, heat penalty={penalty:g}")
    ax.set_axis_off()
    _save(fig, maps_dir / "05_time_change.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, column, title in zip(axes, ["time_min", "hot_fraction", "generalized_time_min"], ["Walk time", "Hot share", "Generalized time"]):
        results.boxplot(column=column, by="penalty", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Penalty")
    fig.suptitle("Sensitivity by heat penalty")
    _save(fig, maps_dir / "06_sensitivity.png")


def validate_maps(output_dir: Path) -> None:
    for name in REQUIRED_MAPS:
        path = output_dir / "maps" / name
        if not path.exists():
            raise FileNotFoundError(path)
        image = Image.open(path).convert("RGB")
        if image.size[0] < 1000 or image.size[1] < 700:
            raise ValueError(f"{name} too small: {image.size}")
        if len(image.getcolors(maxcolors=1_000_000) or []) < 32:
            raise ValueError(f"{name} has too few colors")
