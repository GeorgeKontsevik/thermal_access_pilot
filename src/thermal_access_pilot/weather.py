from __future__ import annotations

import math
import json
import urllib.parse
import urllib.request
from datetime import date

import pandas as pd


def relative_humidity(ta_c: float, dewpoint_c: float) -> float:
    a, b = 17.625, 243.04
    es = math.exp((a * ta_c) / (b + ta_c))
    e = math.exp((a * dewpoint_c) / (b + dewpoint_c))
    return max(0.0, min(100.0, 100.0 * e / es))


def solar_w_m2(ssrd_j_m2: float, seconds: float = 3600.0) -> float:
    return float(ssrd_j_m2) / seconds


def select_extreme_hour(weather: pd.DataFrame) -> pd.Series:
    sunny = weather.loc[weather["global_rad_w_m2"] > 20].copy()
    if sunny.empty:
        sunny = weather.copy()
    sunny = sunny.sort_values(["ta_c", "datetime_local"], ascending=[False, True])
    return sunny.iloc[0]


def fallback_hot_weather(day: date = date(2025, 7, 2)) -> pd.DataFrame:
    hours = pd.date_range(f"{day.isoformat()} 00:00", periods=24, freq="h")
    rows = []
    for ts in hours:
        hour = ts.hour
        ta = 22 + 10 * max(0, math.sin((hour - 6) / 15 * math.pi))
        rad = max(0, 850 * math.sin((hour - 5) / 14 * math.pi))
        rows.append(
            {
                "datetime_local": ts,
                "ta_c": round(ta, 2),
                "rh": 55.0,
                "global_rad_w_m2": round(rad, 2),
                "ws": 2.0,
                "pressure": 1013.25,
            }
        )
    return pd.DataFrame(rows)


def fetch_open_meteo_archive(latitude: float, longitude: float, start_date: str = "2025-06-01", end_date: str = "2025-08-31") -> pd.DataFrame:
    params = urllib.parse.urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relative_humidity_2m,dew_point_2m,shortwave_radiation,wind_speed_10m,surface_pressure",
            "timezone": "Europe/Kaliningrad",
        }
    )
    url = f"https://archive-api.open-meteo.com/v1/archive?{params}"
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.load(response)
    hourly = payload["hourly"]
    frame = pd.DataFrame(
        {
            "datetime_local": pd.to_datetime(hourly["time"]),
            "ta_c": hourly["temperature_2m"],
            "rh": hourly["relative_humidity_2m"],
            "dewpoint_c": hourly["dew_point_2m"],
            "global_rad_w_m2": hourly["shortwave_radiation"],
            "ws": hourly["wind_speed_10m"],
            "pressure": hourly["surface_pressure"],
        }
    )
    return frame.dropna(subset=["ta_c", "rh", "global_rad_w_m2", "ws", "pressure"]).copy()
