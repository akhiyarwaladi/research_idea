"""
Data Preparation Module
- Downloads Yogyakarta road network from OpenStreetMap via OSMnx
- Loads tourist POI coordinates
- Computes real road distance matrix using Dijkstra shortest paths
"""

import os
import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
GRAPH_FILE = os.path.join(DATA_DIR, "yogyakarta_road_network.graphml")
POI_FILE = os.path.join(DATA_DIR, "tourist_pois.csv")
DIST_MATRIX_FILE = os.path.join(DATA_DIR, "distance_matrix.csv")
EUCLIDEAN_MATRIX_FILE = os.path.join(DATA_DIR, "euclidean_matrix.csv")


def download_road_network(force=False):
    """Download drivable road network for Kota Yogyakarta from OSM."""
    if os.path.exists(GRAPH_FILE) and not force:
        print(f"Loading cached road network from {GRAPH_FILE}")
        G = ox.load_graphml(GRAPH_FILE)
    else:
        print("Downloading DIY (Yogyakarta Special Region) road network from OpenStreetMap...")
        # Use the entire DIY province to cover all POIs
        # (Borobudur, Prambanan, Parangtritis, Indrayanti, etc.)
        G = ox.graph_from_place(
            "Daerah Istimewa Yogyakarta, Indonesia", network_type="drive"
        )
        ox.save_graphml(G, filepath=GRAPH_FILE)
        print(f"Saved road network to {GRAPH_FILE}")

    # Convert to undirected to handle one-way streets.
    # For symmetric TSP we need bidirectional traversal.
    # Uses shorter edge weight when parallel edges exist.
    G = G.to_undirected()

    print(f"Road network: {len(G.nodes)} nodes, {len(G.edges)} edges (undirected)")
    return G


def load_pois():
    """Load tourist POI data from CSV."""
    df = pd.read_csv(POI_FILE)
    print(f"Loaded {len(df)} tourist attractions")
    return df


def map_pois_to_network(G, pois_df):
    """Find nearest road network node for each POI."""
    node_ids = []
    for _, row in pois_df.iterrows():
        node_id = ox.distance.nearest_nodes(G, row["longitude"], row["latitude"])
        node_ids.append(node_id)
    pois_df = pois_df.copy()
    pois_df["network_node"] = node_ids
    return pois_df


def compute_road_distance_matrix(G, pois_df):
    """
    Compute n x n distance matrix using shortest path (Dijkstra)
    on the real road network. Distances in meters.
    """
    names = pois_df["name"].tolist()
    nodes = pois_df["network_node"].tolist()
    n = len(nodes)
    dist_matrix = np.zeros((n, n))

    print("Computing real road distance matrix...")
    for i in tqdm(range(n), desc="Rows"):
        for j in range(n):
            if i != j:
                try:
                    dist_matrix[i][j] = nx.shortest_path_length(
                        G, nodes[i], nodes[j], weight="length"
                    )
                except nx.NetworkXNoPath:
                    print(f"  WARNING: No path from {names[i]} to {names[j]}")
                    dist_matrix[i][j] = 1e9  # large penalty

    df = pd.DataFrame(dist_matrix, index=names, columns=names)
    df.to_csv(DIST_MATRIX_FILE)
    print(f"Saved distance matrix to {DIST_MATRIX_FILE}")
    return dist_matrix, names


def compute_euclidean_distance_matrix(pois_df):
    """Compute Euclidean (haversine) distance matrix for comparison."""
    from math import radians, sin, cos, sqrt, atan2

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000  # Earth radius in meters
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        return R * 2 * atan2(sqrt(a), sqrt(1 - a))

    names = pois_df["name"].tolist()
    n = len(names)
    coords = list(zip(pois_df["latitude"], pois_df["longitude"]))
    euc_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                euc_matrix[i][j] = haversine(
                    coords[i][0], coords[i][1], coords[j][0], coords[j][1]
                )

    df = pd.DataFrame(euc_matrix, index=names, columns=names)
    df.to_csv(EUCLIDEAN_MATRIX_FILE)
    print(f"Saved Euclidean distance matrix to {EUCLIDEAN_MATRIX_FILE}")
    return euc_matrix


def load_distance_matrix():
    """Load pre-computed distance matrix from CSV."""
    df = pd.read_csv(DIST_MATRIX_FILE, index_col=0)
    return df.values, df.index.tolist()


def prepare_all_data(force_download=False):
    """Full data preparation pipeline."""
    G = download_road_network(force=force_download)
    pois_df = load_pois()
    pois_df = map_pois_to_network(G, pois_df)

    dist_matrix, names = compute_road_distance_matrix(G, pois_df)
    euc_matrix = compute_euclidean_distance_matrix(pois_df)

    print("\n--- Distance Matrix Summary ---")
    print(f"Shape: {dist_matrix.shape}")
    print(f"Min non-zero: {dist_matrix[dist_matrix > 0].min():.0f} m")
    print(f"Max: {dist_matrix.max():.0f} m")
    print(f"Mean: {dist_matrix[dist_matrix > 0].mean():.0f} m")

    ratio = dist_matrix[euc_matrix > 0] / euc_matrix[euc_matrix > 0]
    print(f"\nRoad/Euclidean ratio — Mean: {ratio.mean():.2f}, Max: {ratio.max():.2f}")

    return G, pois_df, dist_matrix, names


if __name__ == "__main__":
    prepare_all_data()
