from __future__ import annotations

import gzip
import shutil
import urllib.request
from pathlib import Path

from .config import PilotConfig


def download(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target
    tmp = target.with_suffix(target.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as fh:
        shutil.copyfileobj(response, fh)
    tmp.replace(target)
    return target


def fetch_external_tiles(cfg: PilotConfig) -> dict[str, Path]:
    raw = cfg.output_dir / "inputs" / "external" / "raw"
    srtm_gz = download("https://s3.amazonaws.com/elevation-tiles-prod/skadi/N54/N54E020.hgt.gz", raw / "srtm/N54E020.hgt.gz")
    srtm = srtm_gz.with_suffix("")
    if not srtm.exists():
        with gzip.open(srtm_gz, "rb") as src, srtm.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    worldcover = download(
        "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N54E018_Map.tif",
        raw / "worldcover/ESA_WorldCover_10m_2021_v200_N54E018_Map.tif",
    )
    canopy = download(
        "https://libdrive.ethz.ch/index.php/s/cO8or7iOe5dT2Rt/download?path=%2F3deg_cogs&files=ETH_GlobalCanopyHeight_10m_2020_N54E018_Map.tif",
        raw / "canopy/ETH_GlobalCanopyHeight_10m_2020_N54E018_Map.tif",
    )
    return {"srtm": srtm, "worldcover": worldcover, "canopy": canopy}

