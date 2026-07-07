from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from affine import Affine
from rasterio.transform import rowcol
from shapely.geometry import LineString


@dataclass(frozen=True)
class EdgeExposure:
    mean_utci_c: float
    max_utci_c: float
    hot_fraction: float
    sampled_fraction: float


def sample_edge_exposure(
    geometry: LineString,
    raster: np.ndarray,
    transform: Affine,
    hot_threshold_c: float,
    step_m: float = 2.0,
) -> EdgeExposure:
    length = max(float(geometry.length), step_m)
    distances = np.arange(0, length + 0.001, step_m)
    values: list[float] = []
    for distance in distances:
        point = geometry.interpolate(min(float(distance), geometry.length))
        row, col = rowcol(transform, point.x, point.y)
        if 0 <= row < raster.shape[0] and 0 <= col < raster.shape[1]:
            value = float(raster[row, col])
            if np.isfinite(value):
                values.append(value)
    if not values:
        return EdgeExposure(float("nan"), float("nan"), 0.0, 0.0)
    array = np.asarray(values)
    return EdgeExposure(
        mean_utci_c=float(array.mean()),
        max_utci_c=float(array.max()),
        hot_fraction=float((array > hot_threshold_c).mean()),
        sampled_fraction=float(len(values) / len(distances)),
    )


def attach_edge_exposure(graph: nx.Graph, raster: np.ndarray, transform: Affine, hot_threshold_c: float) -> nx.Graph:
    out = graph.copy()
    for u, v, data in out.edges(data=True):
        exposure = sample_edge_exposure(data["geometry"], raster, transform, hot_threshold_c)
        data["mean_utci_c"] = exposure.mean_utci_c
        data["max_utci_c"] = exposure.max_utci_c
        data["hot_fraction"] = exposure.hot_fraction
        data["sampled_fraction"] = exposure.sampled_fraction
        data["hot_length_m"] = data["length_m"] * exposure.hot_fraction
    return out


def _edge_weight(walk_speed_m_s: float, penalty: float):
    def weight(_, __, data) -> float:
        hot_fraction = float(data.get("hot_fraction", 0))
        return float(data["length_m"]) * (1 + penalty * hot_fraction) / walk_speed_m_s

    return weight


def route_all_origins(
    graph: nx.Graph,
    origins: list,
    stops: list,
    penalty: float,
    walk_speed_m_s: float,
) -> dict:
    weighted = _edge_weight(walk_speed_m_s, penalty)
    distances, paths = nx.multi_source_dijkstra(graph, set(stops), weight=lambda u, v, d: weighted(u, v, d))
    out = {}
    for origin in origins:
        if origin not in paths:
            out[origin] = {"status": "unreachable"}
            continue
        path = list(reversed(paths[origin]))
        length = hot_length = generalized_s = 0.0
        for u, v in zip(path, path[1:]):
            data = graph[u][v]
            length += float(data["length_m"])
            hot_length += float(data.get("hot_length_m", float(data["length_m"]) * float(data.get("hot_fraction", 0))))
            generalized_s += weighted(u, v, data)
        out[origin] = {
            "status": "ok",
            "stop_node": path[-1],
            "path": path,
            "length_m": length,
            "time_min": length / walk_speed_m_s / 60,
            "hot_length_m": hot_length,
            "hot_fraction": hot_length / length if length else 0,
            "generalized_time_min": generalized_s / 60,
        }
    return out
