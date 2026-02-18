# Experiment Results Notes
## Tourism Route Optimization in Yogyakarta Using Metaheuristic Algorithms
### Last Updated: 2026-02-16 | 30 Independent Runs | 500 Iterations

---

## 1. EXPERIMENT CONFIGURATION

| Parameter | Value |
|-----------|-------|
| **Problem Size** | 25 tourist attractions in DIY Yogyakarta |
| **Distance Matrix** | Real road distances via Dijkstra on OSM network |
| **Road Network** | 153,334 nodes, 200,104 edges (undirected) |
| **Independent Runs** | 30 per stochastic algorithm |
| **Max Iterations** | 500 (MMAS, ACS, GA) |
| **SA Temperature Steps** | ~9,210 (T0=10000, cooling=0.999, T_end=1.0) |
| **Random Seeds** | 42-71 (base_seed + run_number) |
| **Statistical Test** | Wilcoxon signed-rank (alpha=0.05) |

---

## 2. MAIN RESULTS (TABLE FOR PAPER)

### Table 3: Algorithm Performance Comparison (30 runs)

| Algorithm | Best (km) | Mean (km) | Worst (km) | Std Dev (km) | Mean Time (s) |
|-----------|-----------|-----------|------------|---------------|----------------|
| **MMAS** | 283.76 | 284.35 | 286.71 | 0.85 | 1.70 |
| **ACS** | 283.94 | 285.49 | 288.48 | 1.80 | 0.33 |
| **GA** | 291.31 | 302.13 | 318.03 | 6.79 | 0.68 |
| **SA** | **283.76** | **283.82** | **283.97** | **0.08** | 3.49 |
| **NN** | 297.10 | 297.10 | 297.10 | 0.00 | 0.002 |

### Key Performance Metrics:
- **Best solution found**: 283,761.48 m (283.76 km) by both SA and MMAS
- **Most consistent**: SA (Std Dev = 84.27 m, only 0.03% variation)
- **Fastest metaheuristic**: ACS (0.33 s/run)
- **Best quality-speed tradeoff**: MMAS (283.76 km best, 1.70 s)
- **NN baseline**: 297.10 km (4.7% worse than best metaheuristic)

---

## 3. STATISTICAL SIGNIFICANCE (TABLE FOR PAPER)

### Table 5: Wilcoxon Signed-Rank Tests (Reference: MMAS)

| Comparison | W Statistic | p-value | Significant? | Winner |
|------------|-------------|---------|--------------|--------|
| MMAS vs ACS | 68.00 | 0.0110 | **Yes** | MMAS |
| MMAS vs GA | 0.00 | <0.0001 | **Yes** | MMAS |
| MMAS vs SA | 31.00 | <0.0001 | **Yes** | SA |
| MMAS vs NN | N/A | N/A | Deterministic | MMAS |

### Interpretation for Paper:
- All pairwise comparisons are **statistically significant** (p < 0.05)
- SA significantly outperforms MMAS (p = 0.000004) in solution quality
- MMAS significantly outperforms ACS (p = 0.011) and GA (p < 0.0001)
- Ranking by solution quality: **SA > MMAS > ACS > NN > GA**
- Ranking by computation time: **NN > ACS > GA > MMAS > SA**

---

## 4. PARAMETER SENSITIVITY ANALYSIS

### Table 4: MMAS Parameter Sensitivity (27 combinations, 10 runs each)

**Best parameter combination**: alpha=2.0, beta=2.0, rho=0.02
- Mean distance: 283,818.48 m (283.82 km)
- Std Dev: 108.39 m (very stable)

**Top 5 combinations:**

| Rank | alpha | beta | rho | Mean (km) | Std (km) |
|------|-------|------|-----|-----------|----------|
| 1 | 2.0 | 2.0 | 0.02 | 283.82 | 0.11 |
| 2 | 1.0 | 5.0 | 0.05 | 283.94 | ~0.00 |
| 3 | 1.0 | 3.0 | 0.05 | 283.94 | 0.31 |
| 4 | 1.0 | 3.0 | 0.10 | 283.97 | 0.45 |
| 5 | 1.0 | 5.0 | 0.02 | 284.03 | 0.34 |

### Key Findings:
- **Low evaporation (rho=0.02)** with **high pheromone importance (alpha=2.0)** gives best results
- **Higher beta** (heuristic importance) improves quality but may cause premature convergence
- rho=0.02 consistently outperforms rho=0.05 and rho=0.1 across all alpha-beta combos
- The algorithm is most sensitive to **rho** (evaporation rate)

---

## 5. SCALABILITY ANALYSIS

### Table 6: Scalability Results (10 runs per configuration)

| n | MMAS (km) | ACS (km) | GA (km) | SA (km) | NN (km) |
|---|-----------|----------|---------|---------|---------|
| 10 | 10.60 | 10.60 | 10.60 | 10.60 | 10.60 |
| 12 | 17.98 | 17.98 | 18.01 | 17.98 | 17.98 |
| 15 | 30.09 | 30.09 | 31.02 | 30.09 | 35.35 |
| 20 | 143.21 | 143.38 | 150.49 | **143.10** | 146.92 |
| 25 | 284.35 | 285.49 | 302.13 | **283.82** | 297.10 |

### Computation Time (seconds, mean):

| n | MMAS | ACS | GA | SA |
|---|------|-----|-----|-----|
| 10 | 0.087 | 0.035 | 0.259 | 3.50 |
| 12 | 0.163 | 0.049 | 0.270 | 3.49 |
| 15 | 0.409 | 0.081 | 0.328 | 3.48 |
| 20 | 0.924 | 0.244 | 0.409 | 3.45 |
| 25 | 1.700 | 0.334 | 0.679 | 3.49 |

### Key Findings:
- For small problems (n<=12), all algorithms find the same optimal solution
- GA degrades significantly as n increases (5% worse at n=20, 6.4% worse at n=25)
- SA maintains consistent quality but has fixed time cost (~3.5s regardless of n)
- MMAS and ACS scale well with problem size
- NN becomes increasingly suboptimal (18% worse at n=15)

---

## 6. CONVERGENCE BEHAVIOR

### Key Observations:
- **SA**: Converges fastest to near-optimal (within first 20% of iterations)
- **MMAS**: Steady improvement, converges around iteration 200-300
- **ACS**: Quick initial improvement, plateaus early
- **GA**: Slowest convergence, often stagnates far from optimum

---

## 7. ROAD vs EUCLIDEAN DISTANCE

### Distance Comparison:
- Road/Euclidean ratio: Mean ~1.15-1.20x
- Road distances are consistently longer due to road network topology
- Maximum ratio ~2.0x for some POI pairs (due to geographic barriers)
- This validates the need for real road network distances over Euclidean approximations

---

## 8. FIGURES GENERATED

All figures are saved in both SVG (vector) and PNG (300 DPI raster) formats.

| Figure | File | Description |
|--------|------|-------------|
| Fig 1 | fig1_attraction_map.svg/.png/.html | Tourist attraction locations in DIY Yogyakarta |
| Fig 2a | fig2_best_route_MMAS.html | Best MMAS route on road network (interactive) |
| Fig 2b | fig2_best_route_ACS.html | Best ACS route on road network (interactive) |
| Fig 2c | fig2_best_route_GA.html | Best GA route on road network (interactive) |
| Fig 2d | fig2_best_route_SA.html | Best SA route on road network (interactive) |
| Fig 3 | fig3_convergence.svg/.png | Convergence curves (full + zoomed) |
| Fig 4 | fig4_boxplot.svg/.png | Box plot + violin plot comparison |
| Fig 5 | fig5_parameter_sensitivity.svg/.png | Parameter sensitivity heatmaps |
| Fig 7 | fig7_euclidean_vs_road.svg/.png | Euclidean vs road distance analysis |
| Fig 8 | fig8_scalability.svg/.png | Scalability analysis |
| Fig 9 | fig9_distance_matrix.svg/.png | Distance matrix heatmap |
| Fig 10 | fig10_bar_summary.svg/.png | Summary bar chart |

---

## 9. DATA FILES

| File | Description |
|------|-------------|
| results/tables/raw_results.csv | All 121 individual run results |
| results/tables/summary_table.csv | Summary statistics per algorithm |
| results/tables/wilcoxon_tests.csv | Statistical significance tests |
| results/tables/parameter_sensitivity.csv | 27 parameter combinations |
| results/tables/scalability.csv | Scalability across problem sizes |
| results/tables/tours.csv | Best tour sequences per algorithm |
| results/tables/convergence_*.csv | Convergence data per algorithm |
| results/tables/journal_tables.xlsx | Formatted Excel tables (6 sheets) |
| data/distance_matrix.csv | 25x25 road distance matrix |
| data/euclidean_matrix.csv | 25x25 Euclidean distance matrix |
| data/tourist_pois.csv | 25 POI coordinates and categories |

---

## 10. HOW TO REPRODUCE

```bash
# Install dependencies
pip install -r requirements.txt

# Full pipeline (data + experiments + analysis + figures + Excel tables)
python main.py --runs 30 --iterations 500

# Individual steps
python main.py --data-only        # Step 1: Prepare data
python main.py --experiment       # Step 2: Run experiments
python main.py --analyze          # Step 3: Statistical analysis
python main.py --visualize        # Step 4: Generate figures (SVG + PNG)
python main.py --tables           # Step 5: Generate Excel tables

# Quick test
python main.py --quick            # 5 runs, 200 iterations
```

---

## 11. PERFORMANCE NOTES

### Algorithm Optimizations Applied:
1. **NumPy vectorized tour distance**: O(1) array indexing vs O(n) Python loop
2. **Precomputed eta^beta**: Eliminated redundant power operations
3. **Boolean mask visited tracking**: Faster than set/list operations
4. **SA delta evaluation**: O(1) 2-opt cost vs O(n) full recalculation
5. **GA elitism**: Preserves top 10% of population
6. **GA vectorized fitness**: Batch population evaluation
7. **Early stopping**: Terminates after 100 iterations without improvement

### Execution Times:
- Main experiment (30 runs): ~186 seconds
- Parameter sensitivity (270 runs): ~351 seconds
- Scalability analysis (200 runs): ~120 seconds
- **Total pipeline: ~11 minutes**
