"""Capture Folium HTML maps as high-resolution PNG using Selenium + Chrome.

Generates journal-quality map screenshots with OpenStreetMap base tiles.
"""
import os
import time

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# JS to clean up maps for journal screenshots
CLEANUP_JS = """
    // Hide satellite/extra tile layers — keep only the first (base) layer
    var allLayers = document.querySelectorAll('.leaflet-tile-pane .leaflet-layer');
    if (allLayers.length > 1) {
        for (var i = 1; i < allLayers.length; i++) {
            allLayers[i].style.display = 'none';
        }
    }

    // Hide zoom control
    document.querySelectorAll('.leaflet-control-zoom').forEach(function(z) {
        z.style.display = 'none';
    });

    // Hide attribution
    document.querySelectorAll('.leaflet-control-attribution').forEach(function(a) {
        a.style.display = 'none';
    });

    // Hide layer control
    document.querySelectorAll('.leaflet-control-layers').forEach(function(c) {
        c.style.display = 'none';
    });

    // Hide MiniMap
    document.querySelectorAll('.leaflet-control-minimap').forEach(function(m) {
        m.style.display = 'none';
    });
"""


def capture_html_to_png(html_path, png_path, width=1600, height=1100, wait=5):
    """Open an HTML file in headless Chrome and save a screenshot.

    Forces OpenStreetMap tiles (removes satellite overlay) for clean journal output.
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    opts.add_argument(f"--window-size={width},{height}")
    opts.add_argument("--force-device-scale-factor=2")  # 2x for high-DPI
    opts.add_argument("--hide-scrollbars")

    driver = webdriver.Chrome(options=opts)
    try:
        file_uri = "file:///" + os.path.abspath(html_path).replace("\\", "/")
        driver.get(file_uri)
        time.sleep(2)  # wait for initial load

        driver.execute_script(CLEANUP_JS)

        time.sleep(wait)  # wait for OSM tiles to fully load
        driver.save_screenshot(png_path)
        print(f"  Captured: {os.path.basename(png_path)} ({width}x{height} @2x)")
    finally:
        driver.quit()


def capture_all_maps():
    """Capture all Folium HTML maps as publication-quality PNGs."""
    print("=" * 60)
    print("CAPTURING HTML MAPS AS HIGH-RES PNG (OpenStreetMap)")
    print("=" * 60)

    # Figure 1: Attraction map
    html1 = os.path.join(FIGURES_DIR, "fig1_attraction_map.html")
    if os.path.exists(html1):
        print("\nFig 1: Tourist attraction map...")
        capture_html_to_png(
            html1,
            os.path.join(FIGURES_DIR, "fig1_attraction_map.png"),
            width=1400, height=1000, wait=6,
        )

    # Figure 2: Best route maps
    for algo in ["SA", "MMAS", "ACS", "GA"]:
        html2 = os.path.join(FIGURES_DIR, f"fig2_best_route_{algo}.html")
        if os.path.exists(html2):
            print(f"\nFig 2: Best route ({algo})...")
            capture_html_to_png(
                html2,
                os.path.join(FIGURES_DIR, f"fig2_best_route_{algo}.png"),
                width=1400, height=1000, wait=6,
            )

    # Figure 6: Route comparison grid (2x2)
    html6 = os.path.join(FIGURES_DIR, "fig6_route_comparison.html")
    if os.path.exists(html6):
        print("\nFig 6: Route comparison grid (2x2)...")
        capture_html_to_png(
            html6,
            os.path.join(FIGURES_DIR, "fig6_route_comparison.png"),
            width=2400, height=1800, wait=8,
        )

    print("\nAll map screenshots saved to:", FIGURES_DIR)


if __name__ == "__main__":
    capture_all_maps()
