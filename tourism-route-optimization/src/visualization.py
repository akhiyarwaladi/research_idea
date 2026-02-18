"""
Visualization Module — International Journal Standard

All static figures formatted for reputable journal submission:
- Font: Times New Roman (serif), minimum 10 pt
- Primary output: SVG (vector) for infinite scalability
- Secondary output: PNG (300 DPI) for preview/compatibility
- Format: Single-column (~3.5 in) or double-column (~7.0 in) widths
- No suptitles (captions provided separately in the manuscript)
- Subplot labels: (a), (b) as text annotations
- Grayscale-friendly color palette with distinct markers
- Minimal grid, clean axis formatting
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns

# Optional imports — used only when specific figures are generated
try:
    from scipy.cluster.hierarchy import linkage, leaves_list
except ImportError:
    linkage = leaves_list = None

try:
    from scipy.stats import gaussian_kde
except ImportError:
    gaussian_kde = None

# ── Journal-quality global settings ──────────────────────────
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.edgecolor": "0.8",
    "legend.fancybox": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.grid": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",       # embed actual fonts in SVG
})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def _save_fig(fig, basename):
    """Save figure in both SVG (vector) and PNG (raster) formats."""
    svg_path = os.path.join(FIGURES_DIR, f"{basename}.svg")
    png_path = os.path.join(FIGURES_DIR, f"{basename}.png")
    fig.savefig(svg_path, format="svg")
    fig.savefig(png_path, format="png")
    print(f"  Saved {basename}.svg + .png")

# Grayscale-friendly palette with distinct patterns
ALGO_COLORS = {
    "MMAS": "#D62728",   # red
    "ACS":  "#1F77B4",   # blue
    "GA":   "#2CA02C",   # green
    "SA":   "#7F7F7F",   # gray
    "NN":   "#FF7F0E",   # orange
    "BruteForce": "#17BECF",
}

ALGO_MARKERS = {"MMAS": "o", "ACS": "s", "GA": "D", "SA": "^", "NN": "v"}
ALGO_LINESTYLES = {"MMAS": "-", "ACS": "--", "GA": "-.", "SA": ":", "NN": "-"}
ALGO_ORDER = ["MMAS", "ACS", "GA", "SA", "NN"]

ALGO_HATCHES = {"Best": "", "Mean": "//", "Worst": ".."}

CATEGORY_ICONS = {
    "Cultural Heritage": ("university", "#D62728"),
    "Historical": ("landmark", "#8B0000"),
    "Heritage District": ("archway", "#8B0000"),
    "Cultural Performance": ("music", "#FF69B4"),
    "Museum": ("building-columns", "#1F77B4"),
    "Art Museum": ("palette", "#5F9EA0"),
    "Monument": ("monument", "#9467BD"),
    "Monument/Museum": ("monument", "#9467BD"),
    "Shopping/Cultural": ("store", "#FF7F0E"),
    "Traditional Market": ("cart-shopping", "#FF7F0E"),
    "Public Square": ("tree-city", "#2CA02C"),
    "Educational": ("graduation-cap", "#87CEEB"),
    "Zoo": ("hippo", "#006400"),
    "Temple": ("place-of-worship", "#D62728"),
    "Archaeological Site": ("gopuram", "#8B0000"),
    "Nature/Cliff": ("mountain", "#006400"),
    "Beach": ("umbrella-beach", "#87CEEB"),
    "Nature": ("leaf", "#2CA02C"),
    "Cave/Adventure": ("dungeon", "#808080"),
    "Viewpoint": ("binoculars", "#90EE90"),
}


def ensure_figures_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ==============================================================
# FIGURE 1: Tourist Attraction Map (interactive + static)
# ==============================================================
def fig1_attraction_map(pois_df):
    """Interactive Folium map + static scatter for journal PDF."""
    import folium
    from folium.plugins import MiniMap

    ensure_figures_dir()
    center_lat = pois_df["latitude"].mean()
    center_lon = pois_df["longitude"].mean()

    # CartoDB Positron — cleaner base tile
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12,
                   tiles="CartoDB Positron")
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite",
    ).add_to(m)
    folium.LayerControl().add_to(m)

    # MiniMap for geographic context
    MiniMap(toggle_display=True, position="bottomleft").add_to(m)

    for _, row in pois_df.iterrows():
        icon_name, icon_color = CATEGORY_ICONS.get(row["category"], ("info", "#1F77B4"))
        popup_html = (
            f'<div style="font-family:Arial,sans-serif;min-width:200px;">'
            f'<div style="display:flex;align-items:center;margin-bottom:6px;">'
            f'<span style="background:#1F77B4;color:white;border-radius:50%;'
            f'width:24px;height:24px;display:inline-flex;align-items:center;'
            f'justify-content:center;font-weight:bold;font-size:12px;margin-right:8px;">'
            f'{row["id"]}</span>'
            f'<b style="font-size:13px;">{row["name"]}</b></div>'
            f'<div style="color:#666;font-size:11px;margin-bottom:4px;">'
            f'<i class="fa fa-tag"></i> {row["category"]}</div>'
            f'<div style="color:#888;font-size:10px;">'
            f'<i class="fa fa-map-pin"></i> {row["latitude"]:.4f}, {row["longitude"]:.4f}</div>'
            f'</div>'
        )
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f'{row["id"]}. {row["name"]}',
            icon=folium.Icon(color="white", icon_color=icon_color,
                             icon=icon_name, prefix="fa"),
        ).add_to(m)

    # Auto-zoom to fit all markers tightly
    bounds = [[pois_df["latitude"].min(), pois_df["longitude"].min()],
              [pois_df["latitude"].max(), pois_df["longitude"].max()]]
    m.fit_bounds(bounds, padding=(20, 20))

    # Title bar with subtitle
    title_html = """
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
    z-index:1000;background:white;padding:12px 20px;border-radius:8px;
    box-shadow:0 2px 8px rgba(0,0,0,0.25);text-align:center;">
    <div style="font-size:16px;font-weight:bold;color:#333;">
    Tourist Attractions in DIY Yogyakarta</div>
    <div style="font-size:11px;color:#888;margin-top:2px;">
    25 Locations — Categorized by Type</div></div>"""
    m.get_root().html.add_child(folium.Element(title_html))

    # HTML legend panel
    categories_used = pois_df["category"].unique()
    legend_items = ""
    for cat in sorted(categories_used):
        icon_name, icon_color = CATEGORY_ICONS.get(cat, ("info", "#1F77B4"))
        legend_items += (
            f'<div style="display:flex;align-items:center;margin:3px 0;">'
            f'<i class="fa fa-{icon_name}" style="color:{icon_color};width:18px;'
            f'text-align:center;margin-right:6px;"></i>'
            f'<span style="font-size:11px;">{cat}</span></div>'
        )
    legend_html = (
        f'<div style="position:fixed;bottom:30px;right:10px;z-index:1000;'
        f'background:white;padding:10px 14px;border-radius:8px;'
        f'box-shadow:0 2px 8px rgba(0,0,0,0.25);max-height:300px;overflow-y:auto;">'
        f'<div style="font-weight:bold;font-size:12px;margin-bottom:6px;'
        f'border-bottom:1px solid #eee;padding-bottom:4px;">Categories</div>'
        f'{legend_items}</div>'
    )
    m.get_root().html.add_child(folium.Element(legend_html))

    path = os.path.join(FIGURES_DIR, "fig1_attraction_map.html")
    m.save(path)
    print(f"  Saved Figure 1 (interactive): {path}")

    _fig1_static(pois_df)


def _fig1_static(pois_df):
    """Static scatter for journal PDF — double-column width."""
    # Consolidate categories into broader groups for cleaner legend
    cat_map = {
        "Cultural Heritage": "Cultural",  "Historical": "Cultural",
        "Heritage District": "Cultural",  "Cultural Performance": "Cultural",
        "Museum": "Museum",               "Art Museum": "Museum",
        "Monument": "Monument",           "Monument/Museum": "Monument",
        "Shopping/Cultural": "Urban",      "Traditional Market": "Urban",
        "Public Square": "Urban",          "Educational": "Urban",
        "Zoo": "Urban",
        "Temple": "Temple/Archaeological", "Archaeological Site": "Temple/Archaeological",
        "Nature/Cliff": "Nature",          "Beach": "Nature",
        "Nature": "Nature",                "Cave/Adventure": "Nature",
        "Viewpoint": "Nature",
    }
    cat_styles = {
        "Cultural":            {"c": "#D62728", "marker": "o"},
        "Museum":              {"c": "#1F77B4", "marker": "s"},
        "Monument":            {"c": "#9467BD", "marker": "D"},
        "Urban":               {"c": "#FF7F0E", "marker": "^"},
        "Temple/Archaeological": {"c": "#8C564B", "marker": "P"},
        "Nature":              {"c": "#2CA02C", "marker": "v"},
    }

    pois_df = pois_df.copy()
    pois_df["cat_group"] = pois_df["category"].map(cat_map).fillna("Other")

    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    for grp, style in cat_styles.items():
        sub = pois_df[pois_df["cat_group"] == grp]
        if len(sub) == 0:
            continue
        ax.scatter(sub["longitude"], sub["latitude"],
                   c=style["c"], marker=style["marker"], s=120,
                   zorder=5, edgecolors="black", linewidths=0.6, label=grp,
                   path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    for _, row in pois_df.iterrows():
        ax.annotate(f" {row['id']}", (row["longitude"], row["latitude"]),
                    fontsize=8, fontweight="bold", xytext=(4, 4),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.8, lw=0.3),
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])

    # Coordinate grid overlay
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    ax.set_xlabel("Longitude (\u00b0E)")
    ax.set_ylabel("Latitude (\u00b0S)")
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.95,
              handletextpad=0.3, columnspacing=0.8)
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.set_aspect("equal", adjustable="datalim")

    # Tight axis padding: 0.02 degrees on each side
    lon_min, lon_max = pois_df["longitude"].min(), pois_df["longitude"].max()
    lat_min, lat_max = pois_df["latitude"].min(), pois_df["latitude"].max()
    ax.set_xlim(lon_min - 0.02, lon_max + 0.02)
    ax.set_ylim(lat_min - 0.02, lat_max + 0.02)

    # Scale bar (approximate 10 km at Yogyakarta latitude ~-7.8)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # 1 degree longitude ≈ 111 km * cos(lat), at lat=-7.8 ≈ 109.9 km
    deg_per_10km = 10.0 / 109.9
    bar_x = xlim[0] + (xlim[1] - xlim[0]) * 0.05
    bar_y = ylim[0] + (ylim[1] - ylim[0]) * 0.05
    ax.plot([bar_x, bar_x + deg_per_10km], [bar_y, bar_y], "k-", linewidth=2.5)
    ax.text(bar_x + deg_per_10km / 2, bar_y + (ylim[1] - ylim[0]) * 0.015,
            "10 km", ha="center", fontsize=7, fontweight="bold")

    # North arrow
    arrow_x = xlim[1] - (xlim[1] - xlim[0]) * 0.06
    arrow_y = ylim[1] - (ylim[1] - ylim[0]) * 0.12
    arrow_len = (ylim[1] - ylim[0]) * 0.07
    ax.annotate("N", xy=(arrow_x, arrow_y + arrow_len), xytext=(arrow_x, arrow_y),
                fontsize=9, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))

    _save_fig(fig, "fig1_attraction_map")
    plt.close(fig)


# ==============================================================
# FIGURE 2: Best Route on Real Map (interactive Folium)
# ==============================================================
def fig2_best_route_map(G, tour, pois_df, names, algorithm_name="MMAS",
                         total_distance=None):
    """Best route plotted on actual road network."""
    import folium
    import networkx as nx
    from folium.plugins import MiniMap, AntPath

    ensure_figures_dir()
    center_lat = pois_df["latitude"].mean()
    center_lon = pois_df["longitude"].mean()

    # CartoDB Positron base
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12,
                   tiles="CartoDB Positron")
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite",
    ).add_to(m)
    folium.LayerControl().add_to(m)

    # MiniMap
    MiniMap(toggle_display=True, position="bottomleft").add_to(m)

    node_ids = pois_df["network_node"].tolist()
    n_steps = len(tour)
    cmap = plt.cm.plasma
    colors_arr = [matplotlib.colors.to_hex(cmap(i / n_steps)) for i in range(n_steps)]

    leg_distances = []
    for step in range(n_steps):
        i = tour[step]
        j = tour[(step + 1) % n_steps]
        try:
            route_nodes = nx.shortest_path(G, node_ids[i], node_ids[j], weight="length")
            route_coords = [(G.nodes[nd]["y"], G.nodes[nd]["x"]) for nd in route_nodes]
            seg_dist = nx.shortest_path_length(G, node_ids[i], node_ids[j], weight="length")
            leg_distances.append(seg_dist)

            # AntPath for animated directional route
            AntPath(
                route_coords, color=colors_arr[step], weight=5, opacity=0.85,
                delay=1000, dash_array=[10, 20],
                tooltip=f"Leg {step+1}: {names[i]} \u2192 {names[j]} ({seg_dist/1000:.1f} km)",
            ).add_to(m)
        except Exception as e:
            leg_distances.append(0)
            print(f"  Warning: Could not route {names[i]} \u2192 {names[j]}: {e}")

    # Markers: green=start, red=end, blue=middle
    for step, poi_idx in enumerate(tour):
        row = pois_df.iloc[poi_idx]
        is_start = (step == 0)
        is_end = (step == len(tour) - 1)
        if is_start:
            bg = "#2CA02C"  # green
        elif is_end:
            bg = "#D62728"  # red
        else:
            bg = "#1F77B4"  # blue
        icon_html = (f'<div style="background:{bg};color:white;border-radius:50%;'
                     f'width:26px;height:26px;display:flex;align-items:center;'
                     f'justify-content:center;font-weight:bold;font-size:11px;'
                     f'border:2px solid white;box-shadow:0 2px 4px rgba(0,0,0,0.3);">'
                     f'{step+1}</div>')
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            tooltip=f"Stop {step+1}: {row['name']}",
            icon=folium.DivIcon(html=icon_html, icon_size=(26, 26), icon_anchor=(13, 13)),
        ).add_to(m)

    # Auto-zoom to fit all route points tightly
    bounds = [[pois_df["latitude"].min(), pois_df["longitude"].min()],
              [pois_df["latitude"].max(), pois_df["longitude"].max()]]
    m.fit_bounds(bounds, padding=(20, 20))

    # Title with distance
    dist_text = f" \u2014 {total_distance/1000:.1f} km" if total_distance else ""
    title_html = (f'<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);'
                  f'z-index:1000;background:white;padding:12px 20px;border-radius:8px;'
                  f'box-shadow:0 2px 8px rgba(0,0,0,0.25);text-align:center;">'
                  f'<div style="font-size:15px;font-weight:bold;color:#333;">'
                  f'Optimized Route \u2014 {algorithm_name}{dist_text}</div></div>')
    m.get_root().html.add_child(folium.Element(title_html))

    # Route statistics info panel
    avg_leg = np.mean(leg_distances) / 1000 if leg_distances else 0
    stats_html = (
        f'<div style="position:fixed;bottom:30px;right:10px;z-index:1000;'
        f'background:white;padding:10px 14px;border-radius:8px;'
        f'box-shadow:0 2px 8px rgba(0,0,0,0.25);font-size:11px;">'
        f'<div style="font-weight:bold;margin-bottom:6px;border-bottom:1px solid #eee;'
        f'padding-bottom:4px;">Route Statistics</div>'
        f'<div><b>Total:</b> {total_distance/1000:.1f} km</div>' if total_distance else ''
    )
    stats_html += (
        f'<div><b>Stops:</b> {n_steps}</div>'
        f'<div><b>Avg Leg:</b> {avg_leg:.1f} km</div>'
        f'</div>'
    )
    m.get_root().html.add_child(folium.Element(stats_html))

    # Color gradient legend (plasma start→end)
    gradient_html = (
        '<div style="position:fixed;bottom:30px;left:50%;transform:translateX(-50%);'
        'z-index:1000;background:white;padding:8px 14px;border-radius:8px;'
        'box-shadow:0 2px 8px rgba(0,0,0,0.25);text-align:center;">'
        '<div style="font-size:10px;color:#666;margin-bottom:4px;">Route Progression</div>'
        '<div style="display:flex;align-items:center;gap:6px;">'
        '<span style="font-size:10px;">Start</span>'
        '<div style="width:120px;height:10px;border-radius:3px;'
        'background:linear-gradient(to right, #0d0887, #9c179e, #ed7953, #f0f921);"></div>'
        '<span style="font-size:10px;">End</span></div></div>'
    )
    m.get_root().html.add_child(folium.Element(gradient_html))

    path = os.path.join(FIGURES_DIR, f"fig2_best_route_{algorithm_name}.html")
    m.save(path)
    print(f"  Saved Figure 2 ({algorithm_name}): {path}")


# ==============================================================
# FIGURE 3: Convergence Graph
# ==============================================================
def fig3_convergence_graph(convergence_data=None):
    """Convergence curves — double-column width, two sub-panels."""
    ensure_figures_dir()

    # Try loading from CSV with min/max/std columns
    tables_dir = os.path.join(RESULTS_DIR, "tables")
    conv_full = {}  # algo -> DataFrame with avg, min, max, std
    if convergence_data is None:
        convergence_data = {}
        for algo in ["MMAS", "ACS", "GA", "SA"]:
            p = os.path.join(tables_dir, f"convergence_{algo}.csv")
            if os.path.exists(p):
                df_c = pd.read_csv(p)
                convergence_data[algo] = df_c["avg_best_distance"].values
                conv_full[algo] = df_c
    else:
        for algo in ["MMAS", "ACS", "GA", "SA"]:
            p = os.path.join(tables_dir, f"convergence_{algo}.csv")
            if os.path.exists(p):
                conv_full[algo] = pd.read_csv(p)

    if not convergence_data:
        print("  No convergence data available"); return

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    for ax_idx, (ax, title_lbl, start_frac) in enumerate(
        zip(axes, ["(a)", "(b)"], [0, 0.5])
    ):
        for algo in ["MMAS", "ACS", "GA", "SA"]:
            if algo not in convergence_data:
                continue
            conv = convergence_data[algo]
            start = int(len(conv) * start_frac)
            x = np.arange(start + 1, len(conv) + 1)
            avg_km = np.array(conv[start:]) / 1000

            ax.plot(x, avg_km,
                    label=algo, color=ALGO_COLORS[algo],
                    linestyle=ALGO_LINESTYLES[algo], linewidth=1.2)

            # Shaded min-max confidence bands
            if algo in conv_full and "min_best_distance" in conv_full[algo].columns:
                df_c = conv_full[algo]
                min_km = df_c["min_best_distance"].values[start:] / 1000
                max_km = df_c["max_best_distance"].values[start:] / 1000
                ax.fill_between(x, min_km, max_km,
                                color=ALGO_COLORS[algo], alpha=0.1)

            # Final value annotation on panel (b) right edge
            if ax_idx == 1:
                ax.annotate(f"{avg_km[-1]:.1f}",
                            xy=(x[-1], avg_km[-1]),
                            xytext=(5, 0), textcoords="offset points",
                            fontsize=7, color=ALGO_COLORS[algo], va="center")

        # Convergence point markers (improvement < 0.01% for 20 consecutive iterations)
        if ax_idx == 0:
            for algo in ["MMAS", "ACS", "GA", "SA"]:
                if algo not in convergence_data:
                    continue
                conv = convergence_data[algo]
                conv_arr = np.array(conv) / 1000
                for k in range(20, len(conv_arr)):
                    window = conv_arr[k-20:k]
                    if len(window) >= 20 and window[0] > 0:
                        pct_change = abs(window[-1] - window[0]) / window[0] * 100
                        if pct_change < 0.01:
                            ax.plot(k + 1, conv_arr[k], marker="*", markersize=8,
                                    color=ALGO_COLORS[algo], zorder=6)
                            break

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Distance (km)")
        ax.legend(fontsize=8, loc="upper right", handlelength=2.0)
        ax.tick_params(direction="in", top=True, right=True)
        ax.text(0.02, 0.95, title_lbl, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top")
        # Light grid overlay
        ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.4)

    fig.tight_layout(w_pad=2.0)
    _save_fig(fig, "fig3_convergence")
    plt.close(fig)


# ==============================================================
# FIGURE 4: Box Plot + Violin Plot
# ==============================================================
def fig4_boxplot(df=None):
    """Box and violin plots — double-column width."""
    ensure_figures_dir()
    if df is None:
        df = pd.read_csv(os.path.join(RESULTS_DIR, "tables", "raw_results.csv"))

    multi = df.groupby("algorithm").size()
    multi = multi[multi > 1].index.tolist()
    df_f = df[df["algorithm"].isin(multi)].copy()
    df_f["distance_km"] = df_f["best_distance_m"] / 1000
    order = [a for a in ["MMAS", "ACS", "GA", "SA"] if a in df_f["algorithm"].values]
    if not order:
        print("  Not enough data for boxplot"); return
    palette = {a: ALGO_COLORS[a] for a in order}

    # Load Wilcoxon results for significance brackets
    wilcoxon_path = os.path.join(RESULTS_DIR, "tables", "wilcoxon_all_pairwise.csv")
    sig_pairs = []
    if os.path.exists(wilcoxon_path):
        wdf = pd.read_csv(wilcoxon_path)
        for _, wrow in wdf.iterrows():
            if wrow.get("Significant (p<0.05)") == "Yes":
                comp = wrow["Comparison"]
                parts = comp.split(" vs ")
                if len(parts) == 2:
                    a1, a2 = parts[0].strip(), parts[1].strip()
                    if a1 in order and a2 in order:
                        pval = wrow["p-value"]
                        sig_pairs.append((a1, a2, pval))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

    # Box plot
    ax = axes[0]
    sns.boxplot(data=df_f, x="algorithm", y="distance_km", order=order,
                palette=palette, ax=ax, hue="algorithm", legend=False,
                linewidth=0.8, fliersize=3, width=0.6)
    # Diamond mean markers
    means = df_f.groupby("algorithm")["distance_km"].mean()
    for i, algo in enumerate(order):
        if algo in means:
            ax.plot(i, means[algo], marker="D", color="white",
                    markeredgecolor="black", markersize=5, zorder=6)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Total Route Distance (km)")
    ax.tick_params(direction="in", top=True, right=True)
    ax.text(0.02, 0.95, "(a)", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top")
    ax.grid(True, axis="y", alpha=0.2, linestyle="--", linewidth=0.4)

    # Add significance brackets (top 3 most significant)
    if sig_pairs:
        sig_pairs_sorted = sorted(sig_pairs, key=lambda x: x[2])[:3]
        y_max = df_f["distance_km"].max()
        y_range = df_f["distance_km"].max() - df_f["distance_km"].min()
        bracket_offset = y_range * 0.06
        for bi, (a1, a2, pval) in enumerate(sig_pairs_sorted):
            if a1 in order and a2 in order:
                x1, x2 = order.index(a1), order.index(a2)
                y_bracket = y_max + bracket_offset * (bi + 1)
                ax.plot([x1, x1, x2, x2],
                        [y_bracket - bracket_offset * 0.3, y_bracket,
                         y_bracket, y_bracket - bracket_offset * 0.3],
                        "k-", linewidth=0.7)
                stars = "***" if pval < 0.001 else ("**" if pval < 0.01 else "*")
                ax.text((x1 + x2) / 2, y_bracket + bracket_offset * 0.1,
                        stars, ha="center", fontsize=8)

    # Violin plot
    ax = axes[1]
    sns.violinplot(data=df_f, x="algorithm", y="distance_km", order=order,
                   palette=palette, ax=ax, hue="algorithm", legend=False,
                   inner="quartile", linewidth=0.8, width=0.7)
    # Swarmplot instead of stripplot
    try:
        sns.swarmplot(data=df_f, x="algorithm", y="distance_km", order=order,
                      color="black", size=2, alpha=0.5, ax=ax)
    except Exception:
        # Fallback to stripplot if swarmplot fails (too many points overlap)
        sns.stripplot(data=df_f, x="algorithm", y="distance_km", order=order,
                      color="black", size=2, alpha=0.4, ax=ax, jitter=True)
    # Diamond mean markers
    for i, algo in enumerate(order):
        if algo in means:
            ax.plot(i, means[algo], marker="D", color="white",
                    markeredgecolor="black", markersize=5, zorder=6)
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Total Route Distance (km)")
    ax.tick_params(direction="in", top=True, right=True)
    ax.text(0.02, 0.95, "(b)", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top")
    ax.grid(True, axis="y", alpha=0.2, linestyle="--", linewidth=0.4)

    fig.tight_layout(w_pad=2.0)
    _save_fig(fig, "fig4_boxplot")
    plt.close(fig)


# ==============================================================
# FIGURE 5: Parameter Sensitivity Heatmap
# ==============================================================
def fig5_parameter_heatmap():
    """MMAS parameter sensitivity heatmaps — double-column width."""
    ensure_figures_dir()
    path_csv = os.path.join(RESULTS_DIR, "tables", "parameter_sensitivity.csv")
    if not os.path.exists(path_csv):
        print("  Parameter sensitivity data not found."); return

    df = pd.read_csv(path_csv)
    rho_vals = sorted(df["rho"].unique())

    fig, axes = plt.subplots(1, len(rho_vals), figsize=(7.0, 2.8))
    if len(rho_vals) == 1:
        axes = [axes]

    # Find global minimum cell
    global_min_row = df.loc[df["mean_distance"].idxmin()]

    labels = [f"({chr(97 + i)})" for i in range(len(rho_vals))]
    for idx, rho_val in enumerate(rho_vals):
        ax = axes[idx]
        sub = df[df["rho"] == rho_val]
        pivot = sub.pivot_table(values="mean_distance", index="alpha",
                                columns="beta", aggfunc="mean") / 1000
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu_r", ax=ax,
                    cbar=idx == len(rho_vals) - 1,
                    cbar_kws={"label": "Mean Distance (km)"} if idx == len(rho_vals) - 1 else {},
                    linewidths=0.5, annot_kws={"size": 8})

        # Bold rectangle around optimal cell (global minimum)
        if rho_val == global_min_row["rho"]:
            opt_alpha = global_min_row["alpha"]
            opt_beta = global_min_row["beta"]
            alpha_vals = sorted(pivot.index.tolist())
            beta_vals = sorted(pivot.columns.tolist())
            if opt_alpha in alpha_vals and opt_beta in beta_vals:
                row_idx = alpha_vals.index(opt_alpha)
                col_idx = beta_vals.index(opt_beta)
                ax.add_patch(plt.Rectangle(
                    (col_idx, row_idx), 1, 1,
                    fill=False, edgecolor="red", linewidth=2.5, zorder=10))

        # LaTeX Greek letters for axis labels
        ax.set_title(r"$\rho$ = " + str(rho_val), fontsize=10)
        ax.set_xlabel(r"$\beta$" if idx == 1 else "")
        ax.set_ylabel(r"$\alpha$" if idx == 0 else "")
        if idx > 0:
            ax.set_yticklabels([])
        ax.text(0.02, 0.98, labels[idx], transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top", color="black")

    fig.tight_layout(w_pad=0.5)
    _save_fig(fig, "fig5_parameter_sensitivity")
    plt.close(fig)


# ==============================================================
# FIGURE 6: Multi-Algorithm Route Comparison
# ==============================================================
def fig6_route_comparison(G, all_results, pois_df, names):
    """Generate individual route maps per algorithm + combined grid."""
    ensure_figures_dir()
    algo_best = {}
    for r in all_results:
        algo = r["algorithm"]
        if algo not in algo_best or r["best_distance"] < algo_best[algo]["best_distance"]:
            algo_best[algo] = r
    for algo, result in algo_best.items():
        if algo in ("NN", "BruteForce"):
            continue
        fig2_best_route_map(G, result["tour"], pois_df, names,
                           algorithm_name=algo, total_distance=result["best_distance"])

    # Generate combined 2x2 grid comparison
    fig6_route_comparison_grid(algo_best)


def fig6_route_comparison_grid(algo_best):
    """Create a 2x2 HTML grid with iframes for each algorithm's route map."""
    ensure_figures_dir()
    algos = ["MMAS", "ACS", "GA", "SA"]

    # Build iframe panels
    panels = []
    for algo in algos:
        html_file = f"fig2_best_route_{algo}.html"
        html_path = os.path.join(FIGURES_DIR, html_file)
        if not os.path.exists(html_path):
            continue
        dist_km = ""
        if algo in algo_best and algo_best[algo].get("best_distance"):
            dist_km = f" — {algo_best[algo]['best_distance']/1000:.1f} km"
        panels.append((algo, html_file, dist_km))

    if len(panels) < 2:
        print("  Not enough route maps for comparison grid.")
        return

    panels_html = ""
    for algo, html_file, dist_km in panels:
        color = ALGO_COLORS.get(algo, "#333")
        panels_html += f"""
        <div class="panel">
            <div class="panel-label" style="border-left:4px solid {color};">
                <b>{algo}</b>{dist_km}
            </div>
            <iframe src="{html_file}" frameborder="0"></iframe>
        </div>
        """

    grid_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Route Comparison — All Algorithms</title>
    <style>
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        body {{ font-family: Arial, sans-serif; background: #f5f5f5; }}
        .title {{
            text-align:center; padding:16px;
            font-size:20px; font-weight:bold; color:#333;
            background:white; box-shadow:0 2px 4px rgba(0,0,0,0.1);
        }}
        .title .subtitle {{ font-size:12px; color:#888; margin-top:4px; font-weight:normal; }}
        .grid {{
            display:grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 4px;
            padding: 4px;
            height: calc(100vh - 60px);
        }}
        .panel {{
            position:relative;
            background:white;
            border-radius:4px;
            overflow:hidden;
        }}
        .panel-label {{
            position:absolute; top:8px; left:8px;
            z-index:1000; background:rgba(255,255,255,0.95);
            padding:6px 12px; border-radius:4px;
            font-size:13px; color:#333;
            box-shadow:0 1px 4px rgba(0,0,0,0.15);
        }}
        .panel iframe {{
            width:100%; height:100%; border:none;
        }}
    </style>
</head>
<body>
    <div class="title">
        Side-by-Side Route Comparison
        <div class="subtitle">Optimized routes for 25 tourist attractions in Yogyakarta</div>
    </div>
    <div class="grid">
        {panels_html}
    </div>
</body>
</html>"""

    path = os.path.join(FIGURES_DIR, "fig6_route_comparison.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(grid_html)
    print(f"  Saved Figure 6 (comparison grid): {path}")


# ==============================================================
# FIGURE 7: Euclidean vs Road Distance
# ==============================================================
def fig7_euclidean_vs_road(road_matrix, euclidean_matrix, names):
    """Scatter + histogram — double-column width."""
    ensure_figures_dir()
    n = len(names)
    road_flat, euc_flat = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if road_matrix[i][j] > 0 and euclidean_matrix[i][j] > 0:
                road_flat.append(road_matrix[i][j])
                euc_flat.append(euclidean_matrix[i][j])
    road_km = np.array(road_flat) / 1000
    euc_km = np.array(euc_flat) / 1000
    ratios = road_km / euc_km

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # Scatter with density coloring
    ax = axes[0]
    if gaussian_kde is not None and len(euc_km) > 5:
        xy = np.vstack([euc_km, road_km])
        density = gaussian_kde(xy)(xy)
        idx_sort = density.argsort()
        euc_sorted, road_sorted, density_sorted = euc_km[idx_sort], road_km[idx_sort], density[idx_sort]
        sc = ax.scatter(euc_sorted, road_sorted, c=density_sorted, cmap="viridis",
                        alpha=0.7, edgecolors="0.3", linewidths=0.3, s=25, zorder=3)
        plt.colorbar(sc, ax=ax, shrink=0.7, label="Density", pad=0.02)
    else:
        ax.scatter(euc_km, road_km, alpha=0.5, edgecolors="0.3", linewidths=0.3,
                   s=25, c="#1F77B4", zorder=3)

    max_val = max(road_km.max(), euc_km.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], "k--", lw=0.8, label="1:1 line")
    z = np.polyfit(euc_km, road_km, 1)
    x_line = np.linspace(0, euc_km.max(), 100)
    # Compute R²
    y_pred = np.poly1d(z)(euc_km)
    ss_res = np.sum((road_km - y_pred) ** 2)
    ss_tot = np.sum((road_km - road_km.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    intercept_str = f"+{z[1]:.1f}" if z[1] >= 0 else f"{z[1]:.1f}"
    ax.plot(x_line, np.poly1d(z)(x_line), color="#D62728", lw=1.0,
            label=f"Fit: y={z[0]:.2f}x{intercept_str} (R\u00b2={r_squared:.3f})")
    ax.set_xlabel("Euclidean Distance (km)")
    ax.set_ylabel("Road Distance (km)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(0, max_val); ax.set_ylim(0, max_val)
    ax.set_aspect("equal")
    ax.tick_params(direction="in", top=True, right=True)
    ax.text(0.02, 0.95, "(a)", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top")
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.4)

    # Histogram with KDE overlay
    ax = axes[1]
    ax.hist(ratios, bins=20, edgecolor="black", linewidth=0.5,
            alpha=0.7, color="#1F77B4", density=True)
    # KDE curve overlay
    if gaussian_kde is not None and len(ratios) > 5:
        kde = gaussian_kde(ratios)
        x_kde = np.linspace(ratios.min(), ratios.max(), 200)
        ax.plot(x_kde, kde(x_kde), color="#333", lw=1.5, label="KDE")
    ax.axvline(ratios.mean(), color="#D62728", ls="--", lw=1.0,
               label=f"Mean = {ratios.mean():.2f}")
    ax.axvline(np.median(ratios), color="#FF7F0E", ls=":", lw=1.0,
               label=f"Median = {np.median(ratios):.2f}")
    ax.set_xlabel("Road / Euclidean Ratio")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    ax.tick_params(direction="in", top=True, right=True)
    ax.text(0.02, 0.95, "(b)", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top")
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.4)

    fig.tight_layout(w_pad=2.0)
    _save_fig(fig, "fig7_euclidean_vs_road")
    plt.close(fig)


# ==============================================================
# FIGURE 8: Scalability Analysis
# ==============================================================
def fig8_scalability():
    """Scalability: distance and time vs n — double-column width."""
    ensure_figures_dir()
    path_csv = os.path.join(RESULTS_DIR, "tables", "scalability.csv")
    if not os.path.exists(path_csv):
        print("  Scalability data not found."); return
    df = pd.read_csv(path_csv)

    # Append n=25 data from main results if available
    raw_path = os.path.join(RESULTS_DIR, "tables", "raw_results.csv")
    if os.path.exists(raw_path):
        raw = pd.read_csv(raw_path)
        for algo in raw["algorithm"].unique():
            sub = raw[raw["algorithm"] == algo]
            row_25 = {
                "n_nodes": 25,
                "algorithm": algo,
                "mean_distance": sub["best_distance_m"].mean(),
                "std_distance": sub["best_distance_m"].std(),
                "best_distance": sub["best_distance_m"].min(),
                "mean_time": sub["time_seconds"].mean(),
            }
            df = pd.concat([df, pd.DataFrame([row_25])], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    for ax_idx, (ax, y_col, y_label, lbl) in enumerate(zip(
        axes,
        ["mean_distance", "mean_time"],
        ["Mean Distance (km)", "Mean Time (s)"],
        ["(a)", "(b)"],
    )):
        for algo in ALGO_ORDER:
            sub = df[df["algorithm"] == algo].sort_values("n_nodes")
            if len(sub) == 0:
                continue
            y_data = sub[y_col] / 1000 if y_col == "mean_distance" else sub[y_col]
            std_data = sub.get("std_distance")

            if y_col == "mean_time":
                mask = y_data > 1e-4
                ax.plot(sub["n_nodes"][mask], y_data[mask],
                        marker=ALGO_MARKERS.get(algo, "o"),
                        color=ALGO_COLORS.get(algo, "gray"),
                        linestyle=ALGO_LINESTYLES.get(algo, "-"),
                        label=algo, linewidth=1.2, markersize=5)
            else:
                ax.plot(sub["n_nodes"], y_data,
                        marker=ALGO_MARKERS.get(algo, "o"),
                        color=ALGO_COLORS.get(algo, "gray"),
                        linestyle=ALGO_LINESTYLES.get(algo, "-"),
                        label=algo, linewidth=1.2, markersize=5)

                # Error bars with caps + shaded ±1σ confidence band
                if std_data is not None and "std_distance" in sub.columns:
                    std_km = sub["std_distance"].values / 1000
                    y_arr = y_data.values
                    x_arr = sub["n_nodes"].values
                    ax.errorbar(x_arr, y_arr, yerr=std_km,
                                fmt="none", color=ALGO_COLORS.get(algo, "gray"),
                                capsize=3, capthick=0.8, linewidth=0.8, alpha=0.6)
                    ax.fill_between(x_arr, y_arr - std_km, y_arr + std_km,
                                    color=ALGO_COLORS.get(algo, "gray"), alpha=0.08)

        ax.set_xlabel("Number of Nodes (n)")
        ax.set_ylabel(y_label)
        ax.legend(fontsize=7, loc="upper left", handlelength=2.0)
        ax.tick_params(direction="in", top=True, right=True)
        ax.text(0.02, 0.95, lbl, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top")
        ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.4)
        if y_col == "mean_time":
            ax.set_yscale("log")
        # Integer x-ticks at actual node counts
        xticks = sorted(df["n_nodes"].unique())
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(int(x)) for x in xticks])

    fig.tight_layout(w_pad=2.0)
    _save_fig(fig, "fig8_scalability")
    plt.close(fig)


# ==============================================================
# FIGURE 9: Distance Matrix Heatmap
# ==============================================================
def fig9_distance_matrix_heatmap(road_matrix, names):
    """Distance matrix heatmap — full page width."""
    ensure_figures_dir()
    n = len(names)
    short = [nm[:15] + ".." if len(nm) > 17 else nm for nm in names]
    matrix_km = road_matrix / 1000

    # Hierarchical clustering reorder (Ward linkage)
    reorder_idx = list(range(n))
    if linkage is not None and leaves_list is not None and n > 2:
        try:
            condensed = matrix_km[np.triu_indices(n, k=1)]
            Z = linkage(condensed, method="ward")
            reorder_idx = leaves_list(Z).tolist()
            matrix_km = matrix_km[np.ix_(reorder_idx, reorder_idx)]
            short = [short[i] for i in reorder_idx]
        except Exception:
            pass  # Fall back to original order

    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    mask = np.eye(n, dtype=bool)

    # magma_r colormap (perceptually uniform)
    sns.heatmap(matrix_km, annot=True, fmt=".1f", cmap="magma_r",
                xticklabels=short, yticklabels=short,
                ax=ax, mask=mask, linewidths=0.3, linecolor="white",
                cbar_kws={"label": "Distance (km)", "shrink": 0.8},
                annot_kws={"size": 6})

    # White separator lines between major clusters
    if linkage is not None and n > 4:
        try:
            from scipy.cluster.hierarchy import fcluster
            clusters = fcluster(Z, t=3, criterion="maxclust")
            reordered_clusters = [clusters[i] for i in reorder_idx]
            for i in range(1, n):
                if reordered_clusters[i] != reordered_clusters[i - 1]:
                    ax.axhline(i, color="white", linewidth=2)
                    ax.axvline(i, color="white", linewidth=2)
        except Exception:
            pass

    ax.tick_params(axis="x", rotation=60, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)

    _save_fig(fig, "fig9_distance_matrix")
    plt.close(fig)


# ==============================================================
# FIGURE 10: Bar Chart Summary
# ==============================================================
def fig10_bar_chart_summary(df=None):
    """Summary grouped bar chart — double-column width."""
    ensure_figures_dir()
    if df is None:
        df = pd.read_csv(os.path.join(RESULTS_DIR, "tables", "raw_results.csv"))

    summary = []
    for algo in df["algorithm"].unique():
        sub = df[df["algorithm"] == algo]
        d = sub["best_distance_m"].values / 1000
        t = sub["time_seconds"].values
        summary.append({"Algorithm": algo, "Best": d.min(), "Mean": d.mean(),
                        "Worst": d.max(), "Time": t.mean()})
    sdf = pd.DataFrame(summary)
    # Enforce order
    sdf["_order"] = sdf["Algorithm"].map({a: i for i, a in enumerate(ALGO_ORDER)})
    sdf = sdf.sort_values("_order").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

    # Distance bars — zoomed y-axis to show differences
    ax = axes[0]
    x = np.arange(len(sdf))
    w = 0.25
    hatches = {"Best": "", "Mean": "//", "Worst": ".."}
    for i, (col, alpha_val, lbl) in enumerate(
        [("Best", 1.0, "Best"), ("Mean", 0.6, "Mean"), ("Worst", 0.3, "Worst")]
    ):
        bars = ax.bar(x + (i - 1) * w, sdf[col], w, alpha=alpha_val,
                      color=[ALGO_COLORS.get(a, "gray") for a in sdf["Algorithm"]],
                      edgecolor="black", linewidth=0.4, label=lbl,
                      hatch=hatches[col])
        # Value labels on distance bars
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}", ha="center", va="bottom",
                    fontsize=5.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(sdf["Algorithm"])
    ax.set_ylabel("Distance (km)")
    # Zoom y-axis to show meaningful differences
    y_min = sdf["Best"].min() - 3
    y_max = sdf["Worst"].max() + 8
    ax.set_ylim(y_min, y_max)
    ax.legend(fontsize=8, loc="upper right")
    ax.tick_params(direction="in", top=True, right=True)
    ax.text(0.02, 0.95, "(a)", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top")
    ax.grid(True, axis="y", alpha=0.2, linestyle="--", linewidth=0.4)

    # Time bars
    ax = axes[1]
    bars = ax.bar(sdf["Algorithm"], sdf["Time"],
                  color=[ALGO_COLORS.get(a, "gray") for a in sdf["Algorithm"]],
                  edgecolor="black", linewidth=0.4, width=0.5)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + height * 0.03,
                f"{height:.2f}s", ha="center", va="bottom", fontsize=7,
                fontweight="bold")
    ax.set_ylabel("Mean Time (s)")
    ax.tick_params(direction="in", top=True, right=True)
    ax.text(0.02, 0.95, "(b)", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top")
    ax.grid(True, axis="y", alpha=0.2, linestyle="--", linewidth=0.4)

    fig.tight_layout(w_pad=2.0)
    _save_fig(fig, "fig10_bar_summary")
    plt.close(fig)


# ==============================================================
# MASTER FUNCTION
# ==============================================================
def generate_all_figures(pois_df=None, G=None, names=None,
                         road_matrix=None, euclidean_matrix=None,
                         best_tour=None, convergence_data=None,
                         all_results=None):
    """Generate all journal-quality figures."""
    print("\n" + "=" * 60)
    print("GENERATING JOURNAL-QUALITY FIGURES (SVG + PNG)")
    print("=" * 60)

    if pois_df is not None:
        fig1_attraction_map(pois_df)

    if G is not None and best_tour is not None and pois_df is not None:
        best_dist = None
        if all_results:
            mmas = [r for r in all_results if r["algorithm"] == "MMAS"]
            if mmas:
                best_dist = min(r["best_distance"] for r in mmas)
        fig2_best_route_map(G, best_tour, pois_df, names, total_distance=best_dist)

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

    print(f"\nAll figures saved to: {FIGURES_DIR}")
