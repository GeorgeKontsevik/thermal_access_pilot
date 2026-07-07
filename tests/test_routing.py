import networkx as nx
import numpy as np
from affine import Affine
from shapely.geometry import LineString

from thermal_access_pilot.routing import route_all_origins, sample_edge_exposure


def test_sample_edge_exposure_counts_hot_fraction() -> None:
    raster = np.array([[35.0, 35.0, 20.0, 20.0]], dtype="float32")
    transform = Affine.translation(0, 1) * Affine.scale(1, -1)
    exposure = sample_edge_exposure(LineString([(0.5, 0.5), (3.5, 0.5)]), raster, transform, 32)

    assert round(exposure.hot_fraction, 2) == 0.5
    assert round(exposure.mean_utci_c, 2) == 27.5


def test_heat_penalty_can_choose_longer_cooler_route() -> None:
    graph = nx.Graph()
    graph.add_edge("origin", "hot", length_m=10, hot_fraction=1, geometry=LineString([(0, 0), (10, 0)]))
    graph.add_edge("hot", "stop", length_m=10, hot_fraction=1, geometry=LineString([(10, 0), (20, 0)]))
    graph.add_edge("origin", "cool", length_m=13, hot_fraction=0, geometry=LineString([(0, 0), (0, 13)]))
    graph.add_edge("cool", "stop", length_m=13, hot_fraction=0, geometry=LineString([(0, 13), (13, 13)]))

    base = route_all_origins(graph, ["origin"], ["stop"], penalty=0, walk_speed_m_s=1)
    hot_averse = route_all_origins(graph, ["origin"], ["stop"], penalty=1, walk_speed_m_s=1)

    assert base["origin"]["length_m"] == 20
    assert hot_averse["origin"]["length_m"] == 26
    assert hot_averse["origin"]["hot_fraction"] == 0

