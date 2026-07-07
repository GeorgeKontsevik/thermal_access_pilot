from pathlib import Path

from thermal_access_pilot.config import load_config


def test_load_config_resolves_repo_paths_and_scenarios(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    cfg_dir = repo / "thermal_access_pilot" / "configs"
    cfg_dir.mkdir(parents=True)
    path = cfg_dir / "pilot.toml"
    path.write_text(
        """
[study]
center_lon = 20.4531
center_lat = 54.7003
core_radius_m = 1250
model_halo_m = 250
crs = "EPSG:32634"
pixel_size_m = 2
[routing]
walk_speed_m_s = 1.4
hot_threshold_c = 32
penalties = [0.25, 0.5, 1.0]
[paths]
city_bundle = "aggregated_spatial_pipeline/data/city"
output_dir = "thermal_access_pilot/outputs/kaliningrad"
""".strip(),
        encoding="utf-8",
    )

    cfg = load_config(path, repo_root=repo)

    assert cfg.model_radius_m == 1500
    assert cfg.city_bundle == repo / "aggregated_spatial_pipeline/data/city"
    assert cfg.output_dir == repo / "thermal_access_pilot/outputs/kaliningrad"
    assert cfg.penalties == (0.25, 0.5, 1.0)

