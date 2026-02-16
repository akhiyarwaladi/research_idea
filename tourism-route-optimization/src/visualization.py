"""
Visualization Module — Publication Quality

Generates all figures for the journal paper:
- Figure 1: Tourist attraction map with satellite/OpenStreetMap tiles
- Figure 2: Best route overlaid on real map (interactive + static)
- Figure 3: Convergence graph (all algorithms)
- Figure 4: Box plot + violin plot comparison
- Figure 5: Parameter sensitivity heatmap
- Figure 6: Side-by-side route comparison (multiple algorithms)
- Figure 7: Euclidean vs road distance scatter + ratio
- Figure 8: Scalability analysis chart
- Figure 9: Distance matrix heatmap
- Figure 10: Bar chart summary (best/mean/worst per algorithm)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

ALGO_COLORS = {
    "MMAS": "#e74c3c",
    "ACS": "#3498db",
    "GA": "#2ecc71",
    "SA": "#9b59b6",
    "NN": "#f39c12",
    "BruteForce": "#1abc9c",
}

ALGO_MARKERS = {
    "MMAS": "o",
    "ACS": "s",
    "GA": "D",
    "SA": "^",
    "NN": "v",
}


def ensure_figures_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# FIGURE 1: Tourist Attraction Map (Folium interactive)
# ============================================================

def fig1_attraction_map(pois_df):
    """Map of all tourist attractions with satellite tiles and categories."""
    import folium
    from folium.plugins import MarkerCluster

    ensure_figures_dir()

    center_lat = pois_df["latitude"].mean()
    center_lon = pois_df["longitude"].mean()

    # Interactive map with OpenStreetMap tiles
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11,
                   tiles="OpenStreetMap")

    # Add satellite tile layer
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite",
    ).add_to(m)
    folium.LayerControl().add_to(m)

    # Category -> color mapping
    cat_colors = {
        "Cultural Heritage": "red", "Historical": "darkred",
        "Museum": "blue", "Monument": "purple", "Monument/Museum": "purple",
        "Shopping/Cultural": "orange", "Traditional Market": "orange",
        "Public Square": "green", "Educational": "lightblue",
        "Zoo": "darkgreen", "Art Museum": "cadetblue",
        "Heritage District": "darkred", "Cultural Performance": "pink",
        "Temple": "red", "Archaeological Site": "darkred",
        "Nature/Cliff": "darkgreen", "Beach": "lightblue",
        "Nature": "green", "Cave/Adventure": "gray",
        "Viewpoint": "lightgreen",
    }

    for _, row in pois_df.iterrows():
        color = cat_colors.get(row["category"], "blue")
        popup_html = f"""
        <b>{row['id']}. {row['name']}</b><br>
        Category: {row['category']}<br>
        Lat: {row['latitude']:.4f}, Lon: {row['longitude']:.4f}
        """
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{row['id']}. {row['name']}",
            icon=folium.Icon(color=color, icon="star", prefix="fa"),
        ).add_to(m)

    # Add title
    title_html = """
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
    z-index:1000;background:white;padding:10px;border-radius:5px;
    box-shadow:0 2px 6px rgba(0,0,0,0.3);font-size:16px;font-weight:bold;">
    Tourist Attractions in DIY Yogyakarta (25 Locations)
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    path = os.path.join(FIGURES_DIR, "fig1_attraction_map.html")
    m.save(path)
    print(f"Saved Figure 1 (interactive) to {path}")

    # Static matplotlib version with OSM-style background
    _fig1_static(pois_df)


def _fig1_static(pois_df):
    """Static version of attraction map for PDF paper."""
    fig, ax = plt.subplots(figsize=(12, 10))

    cat_colors = {
        "Cultural Heritage": "#e74c3c", "Historical": "#c0392b",
        "Museum": "#3498db", "Monument": "#9b59b6", "Monument/Museum": "#9b59b6",
        "Shopping/Cultural": "#f39c12", "Traditional Market": "#e67e22",
        "Public Square": "#2ecc71", "Educational": "#1abc9c",
        "Zoo": "#27ae60", "Art Museum": "#2980b9",
        "Heritage District": "#c0392b", "Cultural Performance": "#e91e63",
        "Temple": "#e74c3c", "Archaeological Site": "#c0392b",
        "Nature/Cliff": "#27ae60", "Beach": "#00bcd4",
        "Nature": "#4caf50", "Cave/Adventure": "#607d8b",
        "Viewpoint": "#8bc34a",
    }

    for cat in pois_df["category"].unique():
        sub = pois_df[pois_df["category"] == cat]
        color = cat_colors.get(cat, "#95a5a6")
        ax.scatter(sub["longitude"], sub["latitude"], c=color, s=120,
                   zorder=5, edgecolors="black", linewidths=0.8, label=cat)

    # Add labels with offset to avoid overlap
    for _, row in pois_df.iterrows():
        ax.annotate(
            f" {row['id']}",
            (row["longitude"], row["latitude"]),
            fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Tourist Attraction Locations in Yogyakarta Special Region (DIY)")
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    path = os.path.join(FIGURES_DIR, "fig1_attraction_map.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved Figure 1 (static) to {path}")


# ============================================================
# FIGURE 2: Best Route on Real Map (Interactive + Static)
# ============================================================

def fig2_best_route_map(G, tour, pois_df, names, algorithm_name="MMAS",
                         total_distance=None):
    """Best route plotted on actual road network with real map tiles."""
    import folium
    import networkx as nx

    ensure_figures_dir()

    center_lat = pois_df["latitude"].mean()
    center_lon = pois_df["longitude"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11,
                   tiles="OpenStreetMap")

    # Satellite layer
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite",
    ).add_to(m)
    folium.LayerControl().add_to(m)

    node_ids = pois_df["network_node"].tolist()
    n_steps = len(tour)

    # Use a gradient colormap for the route segments
    cmap = plt.cm.plasma
    colors_arr = [matplotlib.colors.to_hex(cmap(i / n_steps)) for i in range(n_steps)]

    # Draw route segments on real roads
    for step in range(n_steps):
        i = tour[step]
        j = tour[(step + 1) % n_steps]
        try:
            route_nodes = nx.shortest_path(G, node_ids[i], node_ids[j], weight="length")
            route_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route_nodes]

            # Compute segment distance
            seg_dist = nx.shortest_path_length(G, node_ids[i], node_ids[j], weight="length")

            folium.PolyLine(
                route_coords,
                color=colors_arr[step],
                weight=5,
                opacity=0.85,
                tooltip=f"Leg {step+1}: {names[i]} → {names[j]} ({seg_dist/1000:.1f} km)",
            ).add_to(m)

            # Add direction arrows at midpoints
            mid_idx = len(route_coords) // 2
            if mid_idx < len(route_coords):
                folium.RegularPolygonMarker(
                    location=route_coords[mid_idx],
                    number_of_sides=3,
                    radius=6,
                    color=colors_arr[step],
                    fill=True,
                    fill_color=colors_arr[step],
                    fill_opacity=0.8,
                ).add_to(m)

        except Exception as e:
            print(f"  Warning: Could not route {names[i]} → {names[j]}: {e}")

    # Add numbered markers for each stop
    for step, poi_idx in enumerate(tour):
        row = pois_df.iloc[poi_idx]
        is_start = (step == 0)

        popup_html = f"""
        <b>Stop {step+1}: {row['name']}</b><br>
        Category: {row['category']}<br>
        {'<i>START/END POINT</i>' if is_start else ''}
        """

        icon_html = f"""
        <div style="background-color:{'#e74c3c' if is_start else '#3498db'};
        color:white;border-radius:50%;width:28px;height:28px;
        display:flex;align-items:center;justify-content:center;
        font-weight:bold;font-size:12px;border:2px solid white;
        box-shadow:0 2px 4px rgba(0,0,0,0.3);">
        {step+1}
        </div>
        """

        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"Stop {step+1}: {row['name']}",
            icon=folium.DivIcon(html=icon_html, icon_size=(28, 28), icon_anchor=(14, 14)),
        ).add_to(m)

    # Title with distance
    dist_text = f" — Total: {total_distance/1000:.1f} km" if total_distance else ""
    title_html = f"""
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
    z-index:1000;background:white;padding:10px 20px;border-radius:5px;
    box-shadow:0 2px 6px rgba(0,0,0,0.3);font-size:14px;font-weight:bold;">
    Optimized Tourism Route — {algorithm_name}{dist_text}
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    path = os.path.join(FIGURES_DIR, f"fig2_best_route_{algorithm_name}.html")
    m.save(path)
    print(f"Saved Figure 2 ({algorithm_name}) to {path}")


# ============================================================
# FIGURE 3: Convergence Graph
# ============================================================

def fig3_convergence_graph(convergence_data=None):
    """Convergence curves for all algorithms — publication quality."""
    ensure_figures_dir()

    if convergence_data is None:
        convergence_data = {}
        tables_dir = os.path.join(RESULTS_DIR, "tables")
        for algo in ["MMAS", "ACS", "GA", "SA"]:
            path = os.path.join(tables_dir, f"convergence_{algo}.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                convergence_data[algo] = df["avg_best_distance"].values

    if not convergence_data:
        print("No convergence data available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Full convergence
    ax = axes[0]
    for algo, conv in convergence_data.items():
        color = ALGO_COLORS.get(algo, "gray")
        ax.plot(range(1, len(conv) + 1), np.array(conv) / 1000,
                label=algo, color=color, linewidth=1.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Distance (km)")
    ax.set_title("(a) Full Convergence")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Right: Zoomed in (last 50% of iterations)
    ax = axes[1]
    for algo, conv in convergence_data.items():
        color = ALGO_COLORS.get(algo, "gray")
        start = len(conv) // 2
        ax.plot(range(start + 1, len(conv) + 1), np.array(conv[start:]) / 1000,
                label=algo, color=color, linewidth=1.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Distance (km)")
    ax.set_title("(b) Convergence Detail (last 50%)")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Convergence Comparison of Optimization Algorithms", fontsize=14, y=1.02)
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "fig3_convergence.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved Figure 3 to {path}")


# ============================================================
# FIGURE 4: Box Plot + Violin Plot
# ============================================================

def fig4_boxplot(df=None):
    """Box plot and violin plot of distances across runs."""
    ensure_figures_dir()

    if df is None:
        df = pd.read_csv(os.path.join(RESULTS_DIR, "tables", "raw_results.csv"))

    algo_counts = df.groupby("algorithm").size()
    multi_run = algo_counts[algo_counts > 1].index.tolist()
    df_filtered = df[df["algorithm"].isin(multi_run)].copy()
    df_filtered["distance_km"] = df_filtered["best_distance_m"] / 1000

    algo_order = [a for a in ["MMAS", "ACS", "GA", "SA"] if a in df_filtered["algorithm"].values]

    if len(algo_order) == 0:
        print("Not enough multi-run data for boxplot")
        return

    palette = [ALGO_COLORS.get(a, "gray") for a in algo_order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Box plot
    ax = axes[0]
    sns.boxplot(data=df_filtered, x="algorithm", y="distance_km",
                order=algo_order, palette=palette, ax=ax, hue="algorithm",
                legend=False)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Total Route Distance (km)")
    ax.set_title("(a) Box Plot")
    ax.grid(True, alpha=0.3, axis="y")

    # Violin plot
    ax = axes[1]
    sns.violinplot(data=df_filtered, x="algorithm", y="distance_km",
                   order=algo_order, palette=palette, ax=ax, hue="algorithm",
                   legend=False, inner="quartile")
    sns.stripplot(data=df_filtered, x="algorithm", y="distance_km",
                  order=algo_order, color="black", size=3, alpha=0.5, ax=ax)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Total Route Distance (km)")
    ax.set_title("(b) Violin Plot with Individual Runs")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Distribution of Solution Quality Across Independent Runs", fontsize=14, y=1.02)
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "fig4_boxplot.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved Figure 4 to {path}")


# ============================================================
# FIGURE 5: Parameter Sensitivity Heatmap
# ============================================================

def fig5_parameter_heatmap():
    """Parameter sensitivity heatmaps for all rho values in one figure."""
    ensure_figures_dir()

    path_csv = os.path.join(RESULTS_DIR, "tables", "parameter_sensitivity.csv")
    if not os.path.exists(path_csv):
        print("Parameter sensitivity data not found.")
        return

    df = pd.read_csv(path_csv)
    rho_vals = sorted(df["rho"].unique())

    fig, axes = plt.subplots(1, len(rho_vals), figsize=(6 * len(rho_vals), 5))
    if len(rho_vals) == 1:
        axes = [axes]

    for idx, rho_val in enumerate(rho_vals):
        ax = axes[idx]
        sub = df[df["rho"] == rho_val]
        pivot = sub.pivot_table(
            values="mean_distance", index="alpha", columns="beta", aggfunc="mean"
        )
        pivot_km = pivot / 1000

        sns.heatmap(
            pivot_km, annot=True, fmt=".1f", cmap="YlOrRd_r",
            ax=ax, cbar_kws={"label": "Mean Distance (km)"},
            linewidths=0.5,
        )
        ax.set_title(f"ρ = {rho_val}")
        ax.set_xlabel("β (heuristic importance)")
        ax.set_ylabel("α (pheromone importance)")

    fig.suptitle("MMAS Parameter Sensitivity Analysis", fontsize=14, y=1.02)
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "fig5_parameter_sensitivity.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved Figure 5 to {path}")


# ============================================================
# FIGURE 6: Multi-Algorithm Route Comparison on Map
# ============================================================

def fig6_route_comparison(G, all_results, pois_df, names):
    """Side-by-side folium maps comparing routes from different algorithms."""
    import folium
    import networkx as nx

    ensure_figures_dir()

    # Get best tour per algorithm
    algo_best = {}
    for r in all_results:
        algo = r["algorithm"]
        if algo not in algo_best or r["best_distance"] < algo_best[algo]["best_distance"]:
            algo_best[algo] = r

    # Create individual route maps for each algorithm
    for algo, result in algo_best.items():
        if algo in ("NN", "BruteForce"):
            continue
        fig2_best_route_map(G, result["tour"], pois_df, names,
                           algorithm_name=algo, total_distance=result["best_distance"])

    print(f"Saved Figure 6 (individual algorithm routes)")


# ============================================================
# FIGURE 7: Euclidean vs Road Distance
# ============================================================

def fig7_euclidean_vs_road(road_matrix, euclidean_matrix, names):
    """Scatter + histogram of Euclidean vs real road distances."""
    ensure_figures_dir()

    n = len(names)
    road_flat, euc_flat = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if road_matrix[i][j] > 0 and euclidean_matrix[i][j] > 0:
                road_flat.append(road_matrix[i][j])
                euc_flat.append(euclidean_matrix[i][j])

    road_flat = np.array(road_flat) / 1000  # km
    euc_flat = np.array(euc_flat) / 1000
    ratios = (road_flat / euc_flat)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot with regression line
    ax = axes[0]
    ax.scatter(euc_flat, road_flat, alpha=0.5, edgecolors="black",
               linewidths=0.5, s=40, c="#3498db")
    max_val = max(road_flat.max(), euc_flat.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5,
            label="1:1 line (Road = Euclidean)")

    # Fit trend line
    z = np.polyfit(euc_flat, road_flat, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, euc_flat.max(), 100)
    ax.plot(x_line, p(x_line), "g-", linewidth=1.5,
            label=f"Trend: y = {z[0]:.2f}x + {z[1]:.2f}")

    ax.set_xlabel("Euclidean (Haversine) Distance (km)")
    ax.set_ylabel("Real Road Distance (km)")
    ax.set_title("(a) Euclidean vs Real Road Distance")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    # Ratio histogram
    ax = axes[1]
    ax.hist(ratios, bins=25, edgecolor="black", alpha=0.7, color="#3498db")
    ax.axvline(ratios.mean(), color="red", linestyle="--", linewidth=2,
               label=f"Mean: {ratios.mean():.2f}")
    ax.axvline(np.median(ratios), color="orange", linestyle=":",
               linewidth=2, label=f"Median: {np.median(ratios):.2f}")
    ax.set_xlabel("Road / Euclidean Distance Ratio")
    ax.set_ylabel("Frequency")
    ax.set_title("(b) Distribution of Distance Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Comparison of Euclidean and Real Road Distances", fontsize=14, y=1.02)
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "fig7_euclidean_vs_road.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved Figure 7 to {path}")


# ============================================================
# FIGURE 8: Scalability Analysis
# ============================================================

def fig8_scalability():
    """Scalability analysis — distance and time vs number of nodes."""
    ensure_figures_dir()

    path_csv = os.path.join(RESULTS_DIR, "tables", "scalability.csv")
    if not os.path.exists(path_csv):
        print("Scalability data not found.")
        return

    df = pd.read_csv(path_csv)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Distance vs nodes
    ax = axes[0]
    for algo in df["algorithm"].unique():
        sub = df[df["algorithm"] == algo]
        color = ALGO_COLORS.get(algo, "gray")
        marker = ALGO_MARKERS.get(algo, "o")
        ax.plot(sub["n_nodes"], sub["mean_distance"] / 1000, marker=marker,
                color=color, label=algo, linewidth=2, markersize=8)
    ax.set_xlabel("Number of Tourist Attractions (n)")
    ax.set_ylabel("Mean Total Distance (km)")
    ax.set_title("(a) Solution Quality vs Problem Size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Time vs nodes
    ax = axes[1]
    for algo in df["algorithm"].unique():
        sub = df[df["algorithm"] == algo]
        color = ALGO_COLORS.get(algo, "gray")
        marker = ALGO_MARKERS.get(algo, "o")
        ax.plot(sub["n_nodes"], sub["mean_time"], marker=marker,
                color=color, label=algo, linewidth=2, markersize=8)
    ax.set_xlabel("Number of Tourist Attractions (n)")
    ax.set_ylabel("Mean Computation Time (s)")
    ax.set_title("(b) Computation Time vs Problem Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    fig.suptitle("Scalability Analysis", fontsize=14, y=1.02)
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "fig8_scalability.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved Figure 8 to {path}")


# ============================================================
# FIGURE 9: Distance Matrix Heatmap
# ============================================================

def fig9_distance_matrix_heatmap(road_matrix, names):
    """Heatmap visualization of the distance matrix."""
    ensure_figures_dir()

    n = len(names)
    # Shorten names for display
    short_names = []
    for name in names:
        if len(name) > 18:
            short_names.append(name[:16] + "...")
        else:
            short_names.append(name)

    matrix_km = road_matrix / 1000

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.eye(n, dtype=bool)

    sns.heatmap(
        matrix_km, annot=True, fmt=".1f", cmap="YlOrBr",
        xticklabels=short_names, yticklabels=short_names,
        ax=ax, mask=mask, linewidths=0.3,
        cbar_kws={"label": "Distance (km)"},
        annot_kws={"size": 7},
    )
    ax.set_title("Real Road Distance Matrix Between Tourist Attractions (km)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    path = os.path.join(FIGURES_DIR, "fig9_distance_matrix.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved Figure 9 to {path}")


# ============================================================
# FIGURE 10: Bar Chart Summary
# ============================================================

def fig10_bar_chart_summary(df=None):
    """Grouped bar chart: Best/Mean/Worst per algorithm."""
    ensure_figures_dir()

    if df is None:
        df = pd.read_csv(os.path.join(RESULTS_DIR, "tables", "raw_results.csv"))

    summary = []
    for algo in df["algorithm"].unique():
        sub = df[df["algorithm"] == algo]
        dists = sub["best_distance_m"].values / 1000
        times = sub["time_seconds"].values
        summary.append({
            "Algorithm": algo,
            "Best": dists.min(),
            "Mean": dists.mean(),
            "Worst": dists.max(),
            "Time": times.mean(),
        })
    sdf = pd.DataFrame(summary)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Distance comparison
    ax = axes[0]
    x = np.arange(len(sdf))
    width = 0.25
    colors_bar = [ALGO_COLORS.get(a, "gray") for a in sdf["Algorithm"]]

    bars1 = ax.bar(x - width, sdf["Best"], width, label="Best", alpha=0.9,
                   color=colors_bar, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x, sdf["Mean"], width, label="Mean", alpha=0.6,
                   color=colors_bar, edgecolor="black", linewidth=0.5)
    bars3 = ax.bar(x + width, sdf["Worst"], width, label="Worst", alpha=0.3,
                   color=colors_bar, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(sdf["Algorithm"])
    ax.set_ylabel("Total Route Distance (km)")
    ax.set_title("(a) Solution Quality Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    # Time comparison
    ax = axes[1]
    bars = ax.bar(sdf["Algorithm"], sdf["Time"],
                  color=colors_bar, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Mean Computation Time (s)")
    ax.set_title("(b) Computation Time Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.2f}s", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Algorithm Performance Summary", fontsize=14, y=1.02)
    fig.tight_layout()

    path = os.path.join(FIGURES_DIR, "fig10_bar_summary.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved Figure 10 to {path}")


# ============================================================
# MASTER FUNCTION
# ============================================================

def generate_all_figures(pois_df=None, G=None, names=None,
                         road_matrix=None, euclidean_matrix=None,
                         best_tour=None, convergence_data=None,
                         all_results=None):
    """Generate all publication-quality figures."""
    print("\n" + "=" * 60)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("=" * 60)

    if pois_df is not None:
        fig1_attraction_map(pois_df)

    if G is not None and best_tour is not None and pois_df is not None:
        # Find best distance
        best_dist = None
        if all_results:
            mmas = [r for r in all_results if r["algorithm"] == "MMAS"]
            if mmas:
                best_dist = min(r["best_distance"] for r in mmas)
        fig2_best_route_map(G, best_tour, pois_df, names,
                           total_distance=best_dist)

    fig3_convergence_graph(convergence_data)
    fig4_boxplot()
    fig5_parameter_heatmap()

    if G is not None and all_results is not None and pois_df is not None:
        fig6_route_comparison(G, all_results, pois_df, names)

    if road_matrix is not None and euclidean_matrix is not None:
        fig7_euclidean_vs_road(road_matrix, euclidean_matrix, names)

    fig8_scalability()

    if road_matrix is not None and names is not None:
        fig9_distance_matrix_heatmap(road_matrix, names)

    fig10_bar_chart_summary()

    print("\nAll publication-quality figures generated!")
    print(f"Location: {FIGURES_DIR}")
