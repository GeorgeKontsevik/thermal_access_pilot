import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon

from thermal_access_pilot.local_inputs import resolve_heights, select_building_origins


def test_resolve_heights_uses_height_then_storey_then_default() -> None:
    buildings = gpd.GeoDataFrame(
        {
            "height": ["12 m", None, None, "bad"],
            "storey": [9, 4, None, None],
            "geometry": [Point(0, 0)] * 4,
        },
        crs="EPSG:32634",
    )

    out = resolve_heights(buildings)

    assert out["height_m"].tolist() == [12.0, 12.0, 3.0, 3.0]
    assert out["height_rule"].tolist() == ["height", "storey_x3m", "default_3m", "default_3m"]


def test_select_building_origins_keeps_buildings_not_blocks() -> None:
    buildings = gpd.GeoDataFrame(
        {"building_id": [10, 11], "height_m": [3.0, 6.0]},
        geometry=[
            Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
            Polygon([(30, 30), (34, 30), (34, 34), (30, 34)]),
        ],
        crs="EPSG:32634",
    )

    origins = select_building_origins(buildings, Point(0, 0).buffer(10))

    assert origins["building_id"].tolist() == [10]
    assert origins.geometry.iloc[0].equals(Point(2, 2))

