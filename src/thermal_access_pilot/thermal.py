from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from .config import PilotConfig
from .surfaces import SurfacePaths
from .weather import fallback_hot_weather, fetch_open_meteo_archive, select_extreme_hour


def _copy_profile(src_path: Path) -> dict:
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
    profile.update(dtype="float32", compress="deflate")
    return profile


def _write_like(path: Path, values: np.ndarray, like: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **_copy_profile(like)) as dst:
        dst.write(values.astype("float32"), 1)


def run_thermal(cfg: PilotConfig, surfaces: SurfacePaths, force: bool = False) -> dict[str, Path | str | float]:
    out_dir = cfg.output_dir / "thermal"
    headline_utci = out_dir / "headline_utci.tif"
    headline_tmrt = out_dir / "headline_tmrt.tif"
    headline_shadow = out_dir / "headline_shadow.tif"
    hot_mask = out_dir / "hot_mask_utci_gt_threshold.tif"
    metadata = out_dir / "thermal_metadata.json"
    if headline_utci.exists() and not force:
        return {
            "utci": headline_utci,
            "tmrt": headline_tmrt,
            "shadow": headline_shadow,
            "hot_mask": hot_mask,
            "metadata": metadata,
            "engine": json.loads(metadata.read_text(encoding="utf-8")).get("engine", "unknown"),
        }

    weather_source = "open_meteo_archive_hourly_2025_summer"
    try:
        weather = fetch_open_meteo_archive(cfg.center_lat, cfg.center_lon)
    except Exception as exc:  # pragma: no cover - network fallback is reported in metadata.
        weather_source = f"synthetic_hot_day_after_weather_fetch_error:{exc!r}"
        weather = fallback_hot_weather()
    weather_path = out_dir / "selected_day_weather.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(weather_path)
    selected = select_extreme_hour(weather)

    engine = "solweig"
    error = None
    try:
        import solweig

        surface = solweig.SurfaceData.prepare(
            dsm=surfaces.dsm,
            dem=surfaces.dem,
            cdsm=surfaces.cdsm,
            land_cover=surfaces.land_cover,
            working_dir=out_dir / "solweig_work",
            cdsm_relative=True,
        )
        location = solweig.Location(latitude=cfg.center_lat, longitude=cfg.center_lon, utc_offset=2)
        record = solweig.Weather(
            datetime=selected["datetime_local"].to_pydatetime(),
            ta=float(selected["ta_c"]),
            rh=float(selected["rh"]),
            global_rad=float(selected["global_rad_w_m2"]),
            ws=float(selected["ws"]),
            pressure=float(selected["pressure"]),
            timestep_minutes=60,
        )
        result = solweig.calculate(
            surface=surface,
            weather=[record],
            location=location,
            output_dir=out_dir / "solweig",
            outputs=["tmrt", "utci", "shadow"],
        )
        del result
        # Official package writes per-timestep GeoTIFFs under named subdirectories.
        utci_candidates = sorted((out_dir / "solweig").glob("**/utci*.tif"))
        tmrt_candidates = sorted((out_dir / "solweig").glob("**/tmrt*.tif"))
        shadow_candidates = sorted((out_dir / "solweig").glob("**/shadow*.tif"))
        for src, dst in [
            (utci_candidates[-1], headline_utci),
            (tmrt_candidates[-1], headline_tmrt),
            (shadow_candidates[-1], headline_shadow),
        ]:
            with rasterio.open(src) as dataset:
                values = dataset.read(1)
                profile = dataset.profile.copy()
            dst.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(dst, "w", **profile) as target:
                target.write(values, 1)
    except Exception as exc:  # pragma: no cover - integration fallback is inspected in metadata.
        # ponytail: fallback is a transparent diagnostic surface, remove when SOLWEIG is stable on macOS wheels.
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
        shadow = shade
        _write_like(headline_utci, utci, surfaces.dsm)
        _write_like(headline_tmrt, tmrt, surfaces.dsm)
        _write_like(headline_shadow, shadow, surfaces.dsm)

    with rasterio.open(headline_utci) as src:
        utci_values = src.read(1)
        profile = src.profile.copy()
    mask = (utci_values > cfg.hot_threshold_c).astype("float32")
    with rasterio.open(hot_mask, "w", **profile) as dst:
        dst.write(mask, 1)

    meta = {
        "engine": engine,
        "error": error,
        "selected_hour": str(selected["datetime_local"]),
        "weather_source": weather_source,
        "utci_min": float(np.nanmin(utci_values)),
        "utci_mean": float(np.nanmean(utci_values)),
        "utci_max": float(np.nanmax(utci_values)),
        "hot_threshold_c": cfg.hot_threshold_c,
        "share_gt_hot_threshold": float(np.nanmean(utci_values > cfg.hot_threshold_c)),
        "share_gt_32c": float(np.nanmean(utci_values > 32)),
    }
    metadata.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"utci": headline_utci, "tmrt": headline_tmrt, "shadow": headline_shadow, "hot_mask": hot_mask, "metadata": metadata, "engine": engine}
