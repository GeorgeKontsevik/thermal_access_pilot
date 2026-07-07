from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PilotConfig:
    repo_root: Path
    city_bundle: Path
    output_dir: Path
    center_lon: float
    center_lat: float
    core_radius_m: float
    model_halo_m: float
    crs: str
    pixel_size_m: float
    walk_speed_m_s: float
    hot_threshold_c: float
    max_snap_distance_m: float
    penalties: tuple[float, ...]

    @property
    def model_radius_m(self) -> float:
        return self.core_radius_m + self.model_halo_m


def load_config(path: Path, repo_root: Path | None = None) -> PilotConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    root = (repo_root or path.resolve().parents[2]).resolve()
    study = raw["study"]
    routing = raw["routing"]
    paths = raw["paths"]
    return PilotConfig(
        repo_root=root,
        city_bundle=root / paths["city_bundle"],
        output_dir=root / paths["output_dir"],
        center_lon=float(study["center_lon"]),
        center_lat=float(study["center_lat"]),
        core_radius_m=float(study["core_radius_m"]),
        model_halo_m=float(study["model_halo_m"]),
        crs=str(study["crs"]),
        pixel_size_m=float(study["pixel_size_m"]),
        walk_speed_m_s=float(routing["walk_speed_m_s"]),
        hot_threshold_c=float(routing["hot_threshold_c"]),
        max_snap_distance_m=float(routing.get("max_snap_distance_m", 100)),
        penalties=tuple(float(value) for value in routing["penalties"]),
    )

