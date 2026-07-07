from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject

from .config import PilotConfig
from .local_inputs import LocalInputs


@dataclass(frozen=True)
class SurfacePaths:
    dem: Path
    dsm: Path
    cdsm: Path
    land_cover: Path
    core_mask: Path


def _profile(width: int, height: int, transform, crs: str, dtype: str) -> dict:
    return {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
    }


def _write(path: Path, array: np.ndarray, transform, crs: str, dtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **_profile(array.shape[1], array.shape[0], transform, crs, dtype)) as dst:
        dst.write(array.astype(dtype), 1)


def _aligned_grid(cfg: PilotConfig, local: LocalInputs):
    minx, miny, maxx, maxy = local.model_area.bounds
    width = int(np.ceil((maxx - minx) / cfg.pixel_size_m))
    height = int(np.ceil((maxy - miny) / cfg.pixel_size_m))
    transform = from_origin(minx, maxy, cfg.pixel_size_m, cfg.pixel_size_m)
    return width, height, transform


def _reproject_one(path: Path, width: int, height: int, transform, crs: str, resampling=Resampling.bilinear) -> np.ndarray:
    with rasterio.open(path) as src:
        out = np.zeros((height, width), dtype="float32")
        reproject(
            rasterio.band(src, 1),
            out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return out


def _worldcover_to_umep(values: np.ndarray) -> np.ndarray:
    out = np.zeros(values.shape, dtype="uint8")
    out[np.isin(values, [10, 20, 30, 40, 90, 95, 100])] = 5
    out[values == 50] = 1
    out[np.isin(values, [60, 70])] = 6
    out[values == 80] = 7
    return out


def build_surfaces(cfg: PilotConfig, local: LocalInputs, tiles: dict[str, Path]) -> SurfacePaths:
    width, height, transform = _aligned_grid(cfg, local)
    out_dir = cfg.output_dir / "inputs" / "surfaces"

    dem = _reproject_one(tiles["srtm"], width, height, transform, cfg.crs)
    dem = np.where(np.isfinite(dem), dem, np.nanmedian(dem))

    canopy = _reproject_one(tiles["canopy"], width, height, transform, cfg.crs)
    canopy = np.where(np.isfinite(canopy), np.maximum(canopy, 0), 0)

    wc = _reproject_one(tiles["worldcover"], width, height, transform, cfg.crs, Resampling.nearest).astype("uint8")
    land_cover = _worldcover_to_umep(wc)

    buildings = local.buildings.loc[local.buildings.intersects(local.model_area)].copy()
    burned_height = rasterize(
        [(geom, float(height_m)) for geom, height_m in zip(buildings.geometry, buildings["height_m"])],
        out_shape=(height, width),
        transform=transform,
        fill=0.0,
        dtype="float32",
    )
    building_mask = burned_height > 0
    land_cover[building_mask] = 2
    dsm = dem + np.maximum(burned_height, canopy)
    cdsm = canopy.astype("float32")
    cdsm[building_mask] = 0

    core_mask = rasterize([(local.core, 1)], out_shape=(height, width), transform=transform, fill=0, dtype="uint8")

    paths = SurfacePaths(
        dem=out_dir / "dem_2m.tif",
        dsm=out_dir / "dsm_2m.tif",
        cdsm=out_dir / "cdsm_2m.tif",
        land_cover=out_dir / "land_cover_umep_2m.tif",
        core_mask=out_dir / "core_mask_2m.tif",
    )
    _write(paths.dem, dem, transform, cfg.crs, "float32")
    _write(paths.dsm, dsm, transform, cfg.crs, "float32")
    _write(paths.cdsm, cdsm, transform, cfg.crs, "float32")
    _write(paths.land_cover, land_cover, transform, cfg.crs, "uint8")
    _write(paths.core_mask, core_mask, transform, cfg.crs, "uint8")
    return paths

