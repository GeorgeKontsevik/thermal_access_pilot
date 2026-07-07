from pathlib import Path

from PIL import Image


def test_real_run_outputs_are_inspectable(real_output_dir: Path) -> None:
    summary = real_output_dir / "summary.json"
    assert summary.exists()

    for name in [
        "01_inputs.png",
        "02_thermal_fields.png",
        "03_routes_examples.png",
        "04_building_exposure.png",
        "05_time_change.png",
        "06_sensitivity.png",
    ]:
        image = Image.open(real_output_dir / "maps" / name)
        assert image.size[0] >= 1000
        assert image.size[1] >= 700

