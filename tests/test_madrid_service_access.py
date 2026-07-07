from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

import thermal_access_pilot.madrid_service_access as access


class HeatGraphTest(unittest.TestCase):
    def test_heat_changes_only_walk_edges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph_dir = root / "intermodal_graph_iduedu"
            graph_dir.mkdir(parents=True)
            raster_path = root / "utci.tif"

            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                height=2,
                width=2,
                count=1,
                dtype="float32",
                crs="EPSG:25830",
                transform=from_origin(0, 20, 10, 10),
            ) as dst:
                dst.write(np.full((2, 2), 47.0, dtype="float32"), 1)

            edges = gpd.GeoDataFrame(
                {
                    "u": [0, 1],
                    "v": [1, 2],
                    "type": ["walk", "bus"],
                    "length_meter": [10.0, 10.0],
                },
                geometry=[LineString([(1, 1), (9, 9)]), LineString([(1, 9), (9, 1)])],
                crs="EPSG:25830",
            )
            edges.to_parquet(graph_dir / "graph_edges.parquet")

            graph = nx.MultiDiGraph()
            graph.add_edge(0, 1, type="walk", time_min=2.0)
            graph.add_edge(1, 2, type="bus", time_min=2.0)
            with (graph_dir / "graph.pkl").open("wb") as fh:
                pickle.dump(graph, fh)

            access._build_heat_graph(root, raster_path=raster_path)

            result_edges = gpd.read_parquet(graph_dir / "graph_edges.parquet")
            self.assertAlmostEqual(result_edges.loc[0, "length_meter"], 18.0)
            self.assertAlmostEqual(result_edges.loc[1, "length_meter"], 10.0)
            with (graph_dir / "graph.pkl").open("rb") as fh:
                result_graph = pickle.load(fh)
            self.assertAlmostEqual(result_graph[0][1][0]["time_min"], 3.6)
            self.assertAlmostEqual(result_graph[1][2][0]["time_min"], 2.0)


if __name__ == "__main__":
    unittest.main()
