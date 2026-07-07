import pandas as pd

from thermal_access_pilot.weather import relative_humidity, select_extreme_hour, solar_w_m2


def test_weather_conversions_and_hot_sunny_hour_selection() -> None:
    assert round(relative_humidity(30, 20), 1) == 55.1
    assert solar_w_m2(3_600_000) == 1000.0

    weather = pd.DataFrame(
        {
            "datetime_local": pd.to_datetime(["2025-07-01 10:00", "2025-07-02 13:00", "2025-07-03 13:00"]),
            "ta_c": [31, 32, 32],
            "global_rad_w_m2": [0, 500, 600],
        }
    )

    picked = select_extreme_hour(weather)

    assert picked["datetime_local"] == pd.Timestamp("2025-07-02 13:00")

