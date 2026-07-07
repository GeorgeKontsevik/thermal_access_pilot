from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import networkx as nx
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point, box

from .config import PilotConfig


@dataclass(frozen=True)
class LocalInputs:
    center: Point
    core: object
    model_area: object
    buildings: gpd.GeoDataFrame
    origins: gpd.GeoDataFrame
    nodes: gpd.GeoDataFrame
    stops: gpd.GeoDataFrame
    graph: nx.Graph


def _first_number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", str(value))
    if not match:
        return None
    number = float(match.group(0).replace(",", "."))
    return number if number > 0 else None


def resolve_heights(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = buildings.copy()
    heights: list[float] = []
    rules: list[str] = []
    for _, row in out.iterrows():
        height = _first_number(row.get("height"))
        if height is not None:
            heights.append(height)
            rules.append("height")
            continue
        storey = _first_number(row.get("storey"))
        if storey is not None:
            heights.append(storey * 3.0)
            rules.append("storey_x3m")
            continue
        heights.append(3.0)
        rules.append("default_3m")
    out["height_m"] = heights
    out["height_rule"] = rules
    return out


def select_building_origins(buildings: gpd.GeoDataFrame, core) -> gpd.GeoDataFrame:
    selected = buildings.loc[buildings.intersects(core)].copy()
    selected["geometry"] = selected.geometry.representative_point()
    return selected


def load_local_inputs(cfg: PilotConfig) -> LocalInputs:
    transformer = Transformer.from_crs("EPSG:4326", cfg.crs, always_xy=True)
    x, y = transformer.transform(cfg.center_lon, cfg.center_lat)
    center = Point(x, y)
    core = center.buffer(cfg.core_radius_m)
    model_area = box(x - cfg.model_radius_m, y - cfg.model_radius_m, x + cfg.model_radius_m, y + cfg.model_radius_m)

    buildings_path = cfg.city_bundle / "derived_layers/buildings_floor_enriched.parquet"
    nodes_path = cfg.city_bundle / "intermodal_graph_iduedu/graph_nodes.parquet"
    edges_path = cfg.city_bundle / "intermodal_graph_iduedu/graph_edges.parquet"
    if not buildings_path.exists() or not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError(f"Missing Kaliningrad bundle inputs under {cfg.city_bundle}")

    buildings = gpd.read_parquet(buildings_path).to_crs(cfg.crs)
    buildings = buildings.reset_index(drop=True)
    buildings["building_id"] = buildings.index.astype(int)
    buildings = resolve_heights(buildings.loc[buildings.intersects(model_area)].copy())
    origins = select_building_origins(buildings, core)

    nodes = gpd.read_parquet(nodes_path).to_crs(cfg.crs)
    nodes = nodes.loc[nodes.intersects(model_area)].copy()
    nodes["node_id"] = nodes["index"].astype(int)

    edges = gpd.read_parquet(edges_path).to_crs(cfg.crs)
    edges = edges.loc[(edges["type"] == "walk") & edges.intersects(model_area)].copy()

    valid_nodes = set(nodes["node_id"].tolist())
    graph = nx.Graph()
    for row in nodes.itertuples(index=False):
        graph.add_node(int(row.node_id), geometry=row.geometry, type=getattr(row, "type", None))
    for row in edges.itertuples(index=False):
        u, v = int(row.u), int(row.v)
        if u not in valid_nodes or v not in valid_nodes:
            continue
        length = float(row.length_meter or row.geometry.length)
        if graph.has_edge(u, v):
            if length >= graph[u][v]["length_m"]:
                continue
        graph.add_edge(u, v, length_m=length, geometry=row.geometry)

    stop_types = {"platform", "bus", "tram", "trolleybus"}
    stops = nodes.loc[nodes["type"].isin(stop_types) & nodes["node_id"].isin(graph.nodes)].copy()
    reachable = set()
    for component in nx.connected_components(graph):
        if any(node in set(stops["node_id"]) for node in component):
            reachable.update(component)
    graph = graph.subgraph(reachable).copy()
    stops = stops.loc[stops["node_id"].isin(graph.nodes)].copy()

    return LocalInputs(center, core, model_area, buildings, origins, nodes, stops, graph)


def snap_origins_to_graph(origins: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, max_distance_m: float) -> gpd.GeoDataFrame:
    graph_nodes = nodes[["node_id", "geometry"]].copy()
    snapped = gpd.sjoin_nearest(
        origins[["building_id", "height_m", "height_rule", "geometry"]],
        graph_nodes,
        how="left",
        distance_col="snap_distance_m",
        max_distance=max_distance_m,
    )
    return snapped.dropna(subset=["node_id"]).drop(columns=["index_right"]).astype({"node_id": int})

