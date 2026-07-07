import numpy as np
import pandas as pd

from thermal_access_pilot.capa_dc import dc_center_square_bbox, fahrenheit_to_celsius, noaa_capa_export_url, select_before_after_origin, temperature_display_limits


def test_noaa_capa_export_url_locks_washington_raster() -> None:
    url = noaa_capa_export_url((-77.16, 38.78, -76.88, 39.02), size=800)

    assert "Afternoon_Air_Temperature_in_Cities/ImageServer/exportImage" in url
    assert "lockRasterIds" in url
    assert "%5B1%5D" in url
    assert "format=tiff" in url


def test_fahrenheit_to_celsius() -> None:
    assert round(fahrenheit_to_celsius(94), 2) == 34.44


def test_capa_dc_uses_all_building_points_contract() -> None:
    from thermal_access_pilot.capa_dc import BUILDING_LIMIT

    assert BUILDING_LIMIT is None


def test_dc_center_square_bbox_is_about_1500m() -> None:
    bbox = dc_center_square_bbox(-77.04, 38.90, side_m=1500)

    assert bbox[0] < -77.04 < bbox[2]
    assert bbox[1] < 38.90 < bbox[3]
    assert round((bbox[2] - bbox[0]), 3) == 0.017
    assert round((bbox[3] - bbox[1]), 3) == 0.013


def test_select_before_after_origin_picks_largest_delta() -> None:
    results = pd.DataFrame(
        {
            "building_id": [1, 1, 2, 2],
            "origin_node": [10, 10, 20, 20],
            "penalty": [0.0, 1.0, 0.0, 1.0],
            "time_min": [1.0, 1.1, 2.0, 2.2],
            "generalized_time_min": [1.0, 3.0, 2.0, 2.5],
        }
    )

    assert select_before_after_origin(results) == (1, 10)


def test_temperature_display_limits_use_local_percentiles() -> None:
    arr = np.array([[0, 90, 91], [92, 93, 200]], dtype="float32")

    assert temperature_display_limits(arr) == (90.15, 92.85)
