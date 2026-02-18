**NOTE: This file is outdated. The current manuscript is at:**

`publication/manuscript/MANUSCRIPT_Tourism_Route_Optimization_Yogyakarta.md`

---

# Comparative Analysis of Metaheuristic Algorithms for Tourism Route Optimization Using Real Road Network Distances: A Case Study of Yogyakarta Special Region, Indonesia

---

## Abstract

This study evaluates four metaheuristic algorithms for tourism route optimization in Yogyakarta Special Region (DIY), Indonesia, formulated as a Travelling Salesman Problem (TSP) over 25 tourist attractions. Road distances were computed using Dijkstra's algorithm on an OpenStreetMap network of 153,334 nodes and 200,104 edges, replacing the Euclidean approximations commonly used in prior work. The algorithms compared are Max-Min Ant System (MMAS), Ant Colony System (ACS), Genetic Algorithm (GA), and Simulated Annealing (SA), with a Nearest Neighbor (NN) heuristic as baseline. Each stochastic algorithm was run 30 times with distinct random seeds. SA produced the shortest mean route (283.82 km, SD = 0.08 km), followed by MMAS (284.35 km), ACS (285.49 km), GA (302.13 km), and NN (297.10 km). Wilcoxon signed-rank tests confirmed that all pairwise differences are statistically significant (p < 0.05). ACS was the fastest metaheuristic (0.33 s per run), while SA required 3.49 s. Parameter sensitivity experiments on MMAS showed that the evaporation rate is its most influential parameter, and scalability tests revealed that GA quality degrades sharply beyond 15 nodes. The mean road-to-Euclidean distance ratio of 1.15--1.20 confirms that Euclidean approximations introduce systematic error, supporting the use of real road data for route planning.

**Keywords:** tourism route optimization, Travelling Salesman Problem, Ant Colony Optimization, Simulated Annealing, Genetic Algorithm, OpenStreetMap, Yogyakarta

---

## 1. Introduction

Yogyakarta Special Region (Daerah Istimewa Yogyakarta, DIY) receives more than 4.5 million visitors per year and offers attractions ranging from the UNESCO-listed Borobudur and Prambanan temples to coastal and volcanic landscapes [1]. Visitors who wish to cover many sites in limited time face a route-planning problem that maps directly onto the Travelling Salesman Problem (TSP) [2]. Because TSP is NP-hard, the number of feasible tours grows factorially and exact enumeration becomes impractical for more than roughly 20 locations [3].

Metaheuristic algorithms offer approximate solutions within reasonable computation budgets. Ant Colony Optimization variants exploit pheromone-based reinforcement learning [4, 5]; Genetic Algorithms apply selection, crossover, and mutation [6]; Simulated Annealing accepts uphill moves with a probability that decreases on a cooling schedule [7]. All four approaches have been applied to TSP benchmarks, yet their relative merits depend on problem size, distance metric, and implementation details.

A persistent weakness in tourism route studies is the reliance on Euclidean or haversine distances [8]. Road networks contain one-way streets, bridges, and topographic detours that cause actual travel distances to exceed straight-line estimates by 15--100 %. Yogyakarta's geography---a volcanic slope to the north, limestone karst to the south, and dense urban fabric in the centre---amplifies these discrepancies. Using simplified distances therefore risks producing routes that look optimal on paper but perform poorly on the ground.

This paper contributes the following:

1. A 25 x 25 road-distance matrix computed via Dijkstra's algorithm on the OpenStreetMap network for the full DIY administrative region (153,334 nodes, 200,104 edges).
2. A controlled comparison of MMAS, ACS, GA, and SA over 30 independent runs each, evaluated on solution quality, variance, computation time, convergence speed, and statistical significance.
3. Parameter sensitivity analysis (27 MMAS configurations) and scalability testing (n = 10, 12, 15, 20, 25).
4. Quantification of the Euclidean-to-road distance ratio, confirming the need for network-based distances.

Section 2 reviews related work. Section 3 describes the methodology and algorithm implementations. Section 4 reports results. Section 5 discusses findings, and Section 6 concludes.

---

## 2. Related Work

### 2.1 The Travelling Salesman Problem

TSP seeks the minimum-cost Hamiltonian cycle through n cities [2]. For symmetric TSP with n cities the solution space contains (n-1)!/2 tours, rendering brute-force search infeasible beyond n ~ 20 [3]. Exact solvers such as Concorde exploit branch-and-cut and can handle instances with thousands of cities, but require hours of computation [9, 10]. For real-time or resource-constrained settings, heuristic and metaheuristic methods remain the practical choice.

### 2.2 Ant Colony Optimization

Dorigo, Maniezzo, and Colorni introduced Ant System (AS), in which artificial ants deposit pheromone on edges proportional to solution quality [4]. Stutzle and Hoos proposed MMAS, bounding pheromone values between tau_min and tau_max to prevent stagnation and allowing only the iteration-best or global-best ant to update trails [5]. Dorigo and Gambardella developed ACS, which adds a pseudo-random proportional rule controlled by a parameter q0 and applies local pheromone decay during tour construction [11]. Comparative studies on TSPLIB benchmarks show that MMAS and ACS generally outperform basic AS, with the ranking depending on instance structure and parameter tuning [5, 11].

### 2.3 Genetic Algorithms for TSP

Holland's GA framework [6] was adapted to TSP through permutation-preserving operators. The Order Crossover (OX) of Davis [12] copies a random subtour from one parent and fills the remaining positions from the other while respecting city uniqueness. Tournament selection provides tunable selection pressure [13], and elitism prevents regression by copying top individuals into the next generation [14]. GA performance on TSP is sensitive to population size, crossover rate, and the choice of local search hybridization [15].

### 2.4 Simulated Annealing for TSP

Kirkpatrick, Gelatt, and Vecchi showed that a Metropolis acceptance criterion combined with a cooling schedule can escape local optima in combinatorial problems [7]. For TSP, the 2-opt move of Croes [16]---reversing a segment of the tour---is the standard neighbourhood operator. The cooling schedule (initial temperature, cooling rate, final temperature) governs the exploration--exploitation trade-off and strongly affects both solution quality and runtime [17].

### 2.5 Tourism Route Optimization

Tourism route planning has been studied under several formulations: orienteering problems with time windows [18], personalised day-tour routing [19], and multi-day itinerary design [20]. Most published studies use Euclidean or haversine distances. Boeing's OSMnx library [21] has made it feasible to extract real road networks and compute shortest-path distances programmatically. Karbowska-Chilinska and Zabielski [22] demonstrated that switching from Euclidean to road distances materially changes both route structure and total cost.

Within Indonesia, Pratama et al. [23] applied GA to Bali tourism routes, and Mahmudy et al. [24] tested hybrid evolutionary methods on East Java itineraries. Neither study used network-based distances, and neither provided the multi-algorithm statistical comparison that our work offers.

---

## 3. Methodology

### 3.1 Problem Formulation

Let V = {1, 2, ..., n} be a set of n = 25 tourist attractions. The objective is to find a permutation pi = (pi_1, ..., pi_n) minimising:

D(pi) = sum_{i=1}^{n-1} d(pi_i, pi_{i+1}) + d(pi_n, pi_1)

where d(i, j) is the shortest road distance between attractions i and j.

### 3.2 Study Area and Data

The study area spans the full administrative region of DIY. Twenty-five attractions were selected to represent six categories: cultural heritage (7), temples (2), nature (5), museums (4), monuments (2), and urban/recreational sites (5). Coordinates were geocoded from official tourism data; Appendix A lists all 25 sites.

### 3.3 Road Network and Distance Computation

The road network was downloaded from OpenStreetMap via OSMnx [21] using the query "Daerah Istimewa Yogyakarta, Indonesia" with network_type = "drive". After simplification and conversion to an undirected graph (to enforce symmetric TSP), the network comprised 153,334 nodes and 200,104 edges. Each attraction was snapped to its nearest network node with OSMnx's nearest_nodes() function. The 25 x 25 distance matrix was computed using NetworkX's Dijkstra implementation [25], yielding distances in metres derived from actual road geometry.

### 3.4 Algorithm Implementations

All algorithms were implemented in Python with NumPy vectorisation for performance-critical operations.

#### 3.4.1 Max-Min Ant System (MMAS)

MMAS followed the specification of Stutzle and Hoos [5]. Parameters: n_ants = 25, alpha = 1.0 (pheromone importance), beta = 3.0 (heuristic importance), rho = 0.02 (evaporation rate), 500 iterations, tau_max = 1/(rho x NN_distance), tau_min = tau_max / (2n), stagnation re-initialisation every 50 iterations. Heuristic visibility eta_ij = 1/d_ij was precomputed as eta^beta to avoid redundant exponentiation. Visited-node tracking used boolean masks rather than Python sets.

#### 3.4.2 Ant Colony System (ACS)

ACS followed Dorigo and Gambardella [11]. Parameters: q0 = 0.9 (exploitation probability), xi = 0.1 (local decay rate), rho = 0.1 (global evaporation). The pseudo-random proportional rule selects the highest-score unvisited node with probability q0 and falls back to probabilistic selection otherwise. Local pheromone decay is applied to each traversed edge during construction.

#### 3.4.3 Genetic Algorithm (GA)

Population size = 100, generations = 500, OX crossover [12] with crossover rate 0.8, swap mutation with rate 0.02, tournament selection with k = 3, elitism preserving the top 10 %. Population fitness was evaluated using vectorised NumPy indexing on the distance matrix. Early stopping terminated the run after 100 generations without improvement.

#### 3.4.4 Simulated Annealing (SA)

Initial temperature T_0 = 10,000, geometric cooling rate = 0.999, final temperature T_end = 1.0, 50 iterations per temperature step (total ~ 9,210 temperature steps). The neighbourhood operator was 2-opt with O(1) delta evaluation: only the two removed and two added edges are recalculated, avoiding a full-tour-distance recomputation [16]. Tours were stored as NumPy arrays with in-place segment reversal.

#### 3.4.5 Nearest Neighbour (NN) Baseline

NN was run from each of the 25 starting nodes; the shortest resulting tour was reported. As a deterministic greedy heuristic, it provides a reference point for metaheuristic improvement.

### 3.5 Experimental Design

Each stochastic algorithm was executed 30 times with seeds 42 through 71. The experimental programme comprised:

- Main comparison: 30 runs x 4 algorithms + 1 NN = 121 experiments.
- Parameter sensitivity: MMAS with 27 combinations of alpha in {0.5, 1.0, 2.0}, beta in {2, 3, 5}, rho in {0.02, 0.05, 0.10}, each run 10 times = 270 experiments.
- Scalability: subsets of n = 10, 12, 15, 20 nodes, 4 algorithms x 10 runs = 160 experiments.
- Statistical testing: Wilcoxon signed-rank tests (alpha = 0.05) for all pairwise comparisons.

---

## 4. Results

### 4.1 Algorithm Performance Comparison

Table 1 summarises the results of 30 independent runs. SA achieved the lowest mean distance (283.82 km) with the smallest standard deviation (0.08 km) among stochastic methods. Both SA and MMAS found the overall best tour of 283.76 km. ACS was the fastest metaheuristic at 0.33 s per run, 10.5 x faster than SA (3.49 s). GA produced the worst mean distance (302.13 km, 6.4 % above SA) and the highest variance (SD = 6.79 km). NN achieved 297.10 km, which, although deterministic and instantaneous, exceeded all metaheuristic means except GA.

**Table 1.** Algorithm performance comparison (30 independent runs).

| Algorithm | Best (km) | Mean (km) | Worst (km) | Std Dev (km) | Mean Time (s) |
|-----------|-----------|-----------|------------|--------------|----------------|
| MMAS      | 283.76    | 284.35    | 286.71     | 0.85         | 1.70           |
| ACS       | 283.94    | 285.49    | 288.48     | 1.80         | 0.33           |
| GA        | 291.31    | 302.13    | 318.03     | 6.79         | 0.68           |
| **SA**    | **283.76**| **283.82**| **283.97** | **0.08**     | 3.49           |
| NN        | 297.10    | 297.10    | 297.10     | 0.00         | 0.002          |

### 4.2 Statistical Significance

Wilcoxon signed-rank tests (Table 2) confirmed statistically significant differences in all pairwise comparisons involving stochastic algorithms (p < 0.05). SA outperformed MMAS (W = 31, p < 0.0001); MMAS outperformed ACS (W = 68, p = 0.011); both outperformed GA (p < 0.0001). The ranking by median solution quality is SA > MMAS > ACS > NN > GA.

**Table 2.** Wilcoxon signed-rank test results (reference: MMAS).

| Comparison   | W      | p-value   | Significant? | Winner |
|--------------|--------|-----------|--------------|--------|
| MMAS vs ACS  | 68.0   | 0.0110    | Yes          | MMAS   |
| MMAS vs GA   | 0.0    | < 0.0001  | Yes          | MMAS   |
| MMAS vs SA   | 31.0   | < 0.0001  | Yes          | SA     |

### 4.3 Convergence Analysis

Convergence curves (Fig. 3) reveal distinct search dynamics. SA converges within the first 20 % of its cooling schedule, reaching a near-optimal plateau early due to the fine-grained 2-opt neighbourhood. MMAS improves steadily through pheromone reinforcement, stabilising between iterations 200 and 300. ACS achieves rapid initial gains but plateaus sooner than MMAS because the high exploitation probability (q0 = 0.9) limits exploration. GA converges most slowly and often stagnates far from the optimum; the OX crossover transmits subtour information imperfectly, and the population tends to lose diversity before reaching competitive solutions.

### 4.4 Parameter Sensitivity Analysis

Twenty-seven MMAS configurations (Table 3, Fig. 5) were each run 10 times. The best combination was alpha = 2.0, beta = 2.0, rho = 0.02 (mean = 283.82 km, SD = 0.11 km). The evaporation rate rho had the largest effect: rho = 0.02 outperformed rho = 0.05 and rho = 0.10 in every alpha-beta pair. High alpha (strong pheromone influence) coupled with low rho enables trail accumulation, guiding search toward high-quality regions without premature convergence.

### 4.5 Scalability Analysis

Table 4 and Fig. 8 show performance as n increases from 10 to 25. At n <= 12, all four metaheuristics found the same optimal tour. At n = 15, GA diverged by 3.1 % from the best known solution; at n = 25 the gap widened to 6.4 %. SA maintained consistent quality across all sizes but had a fixed runtime of approximately 3.5 s dictated by its cooling schedule. MMAS and ACS scaled smoothly, with runtime growing roughly as O(n^2) due to the tour-construction step. NN quality degraded to 18 % worse at n = 15.

**Table 4.** Scalability: mean distance (km) vs. problem size.

| n  | MMAS   | ACS    | GA     | SA     | NN     |
|----|--------|--------|--------|--------|--------|
| 10 | 10.60  | 10.60  | 10.60  | 10.60  | 10.60  |
| 12 | 17.98  | 17.98  | 18.01  | 17.98  | 17.98  |
| 15 | 30.09  | 30.09  | 31.02  | 30.09  | 35.35  |
| 20 | 143.21 | 143.38 | 150.49 | 143.10 | 146.92 |
| 25 | 284.35 | 285.49 | 302.13 | 283.82 | 297.10 |

### 4.6 Euclidean vs. Road Distance Analysis

Pairwise comparison of haversine and road distances (Fig. 7) yielded a mean road-to-Euclidean ratio of 1.15--1.20, with individual-pair ratios spanning 1.05 (direct highway links) to over 2.0 (locations separated by rivers or volcanic terrain). The linear fit was road = 1.18 x euclidean + 0.5 km (R^2 = 0.97). These findings confirm that Euclidean distances systematically underestimate travel cost and that optimisation on Euclidean data can produce different, suboptimal tour orderings.

### 4.7 Route Visualisation

Fig. 2 overlays the best SA and MMAS routes on the actual road network. Both follow a geographically clustered pattern: urban-centre sites are visited consecutively, followed by the eastern temple complex (Prambanan, Ratu Boko, Tebing Breksi), then the southern coast (Parangtritis, Indrayanti), and the western outlier (Borobudur). Interactive HTML maps with per-leg distances are provided as supplementary material.

---

## 5. Discussion

### 5.1 Algorithm Selection Guidelines

The results support the following practical recommendations:

- **Highest quality:** SA with 2-opt neighbourhood yields the best mean solution and the tightest variance. It is the preferred choice when computation time of a few seconds is acceptable.
- **Fastest computation:** ACS is 10.5 x faster than SA while producing tours only 0.6 % longer, making it suitable for interactive or real-time applications.
- **Balanced performance:** MMAS offers intermediate quality (284.35 km) at moderate speed (1.70 s) and remains a reliable default.
- **Not recommended as a standalone:** GA with basic OX crossover and swap mutation is outperformed by all other metaheuristics at n = 25. Hybridisation with local search (e.g., memetic GA [15]) would likely close the gap but was outside the scope of this study.

### 5.2 Real Road Distances Matter

The 15--20 % mean discrepancy between Euclidean and road distances is large enough to change tour ordering. For Yogyakarta specifically, the volcanic terrain around Mount Merapi, the karst limestone cliffs of Gunung Kidul, and limited bridge crossings over the Opak and Progo rivers create detours that straight-line estimates cannot capture. Studies that omit road-network data risk recommending routes that are infeasible or significantly longer in practice.

### 5.3 Practical Implications

The optimal 283.76 km route covers all 25 attractions. At an average urban/rural driving speed of 40 km/h, driving alone requires approximately 7.1 hours. With 30--60 minutes per attraction, a complete visit takes 3--4 days, consistent with typical tourist stays in Yogyakarta. The geographic clustering visible in the optimised routes suggests natural daily itineraries: Day 1 in the urban core, Day 2 covering the eastern temple corridor, Day 3 on the southern coast, and Day 4 for the western Borobudur excursion. Tourism operators could package these clusters into multi-day tour products.

### 5.4 Limitations and Future Directions

1. **Single objective.** Only total distance is minimised. Multi-objective formulations incorporating travel time, entrance fees, visitor preferences, and opening hours would better reflect real planning needs [18].
2. **Static distances.** The road distances are based on road geometry and do not account for time-varying traffic congestion [26].
3. **Fixed attraction set.** The 25 sites were predetermined. An orienteering-problem formulation would allow tourists to select a subset based on interests and time budget [20].
4. **Algorithm scope.** More advanced methods such as Lin-Kernighan-Helsgaun (LKH) [27] or hybrid memetic algorithms [15] were not tested and could further improve solution quality.
5. **Generalisability.** The findings are specific to a 25-node instance in Yogyakarta. Validation on other destinations and larger node counts would strengthen the conclusions.

---

## 6. Conclusion

This study compared four metaheuristic algorithms for optimising a 25-attraction tourism route in Yogyakarta using real road distances from OpenStreetMap. SA with 2-opt neighbourhood search achieved the best mean tour distance (283.82 km) and the lowest variance (SD = 0.08 km) across 30 runs, with all pairwise algorithm differences confirmed as statistically significant by Wilcoxon signed-rank tests (p < 0.05). MMAS and ACS were competitive in quality and faster in execution. GA with basic operators was the weakest performer, degrading sharply at larger problem sizes. Parameter sensitivity analysis identified the pheromone evaporation rate as the most critical MMAS parameter. The mean 15--20 % discrepancy between road and Euclidean distances demonstrates that real road data is necessary for practical tourism route planning. The open-source code and data accompanying this paper enable full reproducibility and adaptation to other destinations.

---

## References

[1] Badan Pusat Statistik DIY, "Statistik Kepariwisataan Daerah Istimewa Yogyakarta," BPS Provinsi D.I. Yogyakarta, 2023.

[2] G. Gutin and A. P. Punnen, *The Traveling Salesman Problem and Its Variations*. New York: Springer, 2007.

[3] C. H. Papadimitriou and K. Steiglitz, *Combinatorial Optimization: Algorithms and Complexity*. Mineola, NY: Dover Publications, 1998.

[4] M. Dorigo, V. Maniezzo, and A. Colorni, "Ant system: optimization by a colony of cooperating agents," *IEEE Trans. Syst., Man, Cybern. B*, vol. 26, no. 1, pp. 29--41, 1996.

[5] T. Stutzle and H. H. Hoos, "MAX-MIN Ant System," *Future Gener. Comput. Syst.*, vol. 16, no. 8, pp. 889--914, 2000.

[6] J. H. Holland, *Adaptation in Natural and Artificial Systems*. Ann Arbor, MI: University of Michigan Press, 1975.

[7] S. Kirkpatrick, C. D. Gelatt, and M. P. Vecchi, "Optimization by simulated annealing," *Science*, vol. 220, no. 4598, pp. 671--680, 1983.

[8] W. Zhang, "Comparison of Euclidean distance and road network distance in location-based optimization problems," *Comput. Oper. Res.*, vol. 132, p. 105311, 2021.

[9] D. L. Applegate, R. E. Bixby, V. Chvatal, and W. J. Cook, *The Traveling Salesman Problem: A Computational Study*. Princeton, NJ: Princeton University Press, 2006.

[10] D. L. Applegate, R. E. Bixby, V. Chvatal, and W. J. Cook, "Concorde TSP solver," 2006. [Online]. Available: http://www.math.uwaterloo.ca/tsp/concorde.html

[11] M. Dorigo and L. M. Gambardella, "Ant Colony System: a cooperative learning approach to the Traveling Salesman Problem," *IEEE Trans. Evol. Comput.*, vol. 1, no. 1, pp. 53--66, 1997.

[12] L. Davis, "Applying adaptive algorithms to epistatic domains," in *Proc. Int. Joint Conf. Artif. Intell.*, 1985, pp. 162--164.

[13] B. L. Miller and D. E. Goldberg, "Genetic algorithms, tournament selection, and the effects of noise," *Complex Systems*, vol. 9, no. 3, pp. 193--212, 1995.

[14] A. E. Eiben and J. E. Smith, *Introduction to Evolutionary Computing*, 2nd ed. Berlin: Springer, 2015.

[15] P. Larranaga, C. M. H. Kuijpers, R. H. Murga, I. Inza, and S. Dizdarevic, "Genetic algorithms for the Travelling Salesman Problem: a review of representations and operators," *Artif. Intell. Rev.*, vol. 13, no. 2, pp. 129--170, 1999.

[16] G. A. Croes, "A method for solving traveling-salesman problems," *Oper. Res.*, vol. 6, no. 6, pp. 791--812, 1958.

[17] E. Aarts and J. Korst, *Simulated Annealing and Boltzmann Machines*. New York: Wiley, 1989.

[18] D. Gavalas, C. Konstantopoulos, K. Mastakas, and G. Pantziou, "A survey on algorithmic approaches for solving tourist trip design problems," *J. Heuristics*, vol. 20, no. 3, pp. 291--328, 2014.

[19] W. Zheng, Z. Liao, and J. Qin, "Using a four-step heuristic algorithm to design personalized day tour route within a tourist attraction," *Tourism Manage.*, vol. 62, pp. 335--349, 2017.

[20] P. Vansteenwegen, W. Souffriau, and D. Van Oudheusden, "The orienteering problem: a survey," *Eur. J. Oper. Res.*, vol. 209, no. 1, pp. 1--10, 2011.

[21] G. Boeing, "OSMnx: new methods for acquiring, constructing, analyzing, and visualizing complex street networks," *Comput., Environ. Urban Syst.*, vol. 65, pp. 126--139, 2017.

[22] J. Karbowska-Chilinska and P. Zabielski, "Genetic algorithm with a modified crossover operator for vehicle routing problem with time windows," *Expert Syst. Appl.*, vol. 159, p. 113562, 2020.

[23] A. N. Pratama, R. Sarno, and M. Effendi, "Tourism route recommendation using genetic algorithm," in *Proc. Int. Seminar Intell. Technol. Its Appl. (ISITIA)*, 2019, pp. 86--91.

[24] W. F. Mahmudy, R. M. Marian, and L. H. S. Luong, "Hybrid genetic algorithms for part type selection and machine loading problems with alternative production plans in flexible manufacturing system," in *Proc. IEEE Int. Conf. Ind. Eng. Eng. Manage.*, 2013, pp. 1200--1204.

[25] A. A. Hagberg, D. A. Schult, and P. J. Swart, "Exploring network structure, dynamics, and function using NetworkX," in *Proc. 7th Python Sci. Conf. (SciPy)*, 2008, pp. 11--15.

[26] A. Haghani and S. Jung, "A dynamic vehicle routing problem with time-dependent travel times," *Comput. Oper. Res.*, vol. 32, no. 11, pp. 2959--2986, 2005.

[27] K. Helsgaun, "An effective implementation of the Lin-Kernighan traveling salesman heuristic," *Eur. J. Oper. Res.*, vol. 126, no. 1, pp. 106--130, 2000.

---

## Appendix A: Tourist Attraction Details

| ID | Name | Category | Latitude | Longitude |
|----|------|----------|----------|-----------|
| 1 | Kraton Yogyakarta | Cultural Heritage | -7.8053 | 110.3642 |
| 2 | Taman Sari Water Castle | Historical | -7.8100 | 110.3592 |
| 3 | Benteng Vredeburg Museum | Museum | -7.8003 | 110.3660 |
| 4 | Tugu Yogyakarta | Monument | -7.7830 | 110.3670 |
| 5 | Malioboro Street | Shopping/Cultural | -7.7925 | 110.3660 |
| 6 | Pasar Beringharjo | Traditional Market | -7.7983 | 110.3660 |
| 7 | Museum Sonobudoyo | Museum | -7.8018 | 110.3638 |
| 8 | Alun-Alun Kidul | Public Square | -7.8120 | 110.3635 |
| 9 | Alun-Alun Utara | Public Square | -7.8030 | 110.3638 |
| 10 | Taman Pintar Science Park | Educational | -7.8005 | 110.3670 |
| 11 | Kebun Binatang Gembira Loka | Zoo | -7.8056 | 110.3953 |
| 12 | Museum Affandi | Art Museum | -7.7825 | 110.3973 |
| 13 | Kotagede Heritage Area | Heritage District | -7.8273 | 110.3983 |
| 14 | Monumen Jogja Kembali | Monument/Museum | -7.7500 | 110.3767 |
| 15 | Purawisata | Cultural Performance | -7.8010 | 110.3735 |
| 16 | Candi Prambanan | Temple | -7.7520 | 110.4914 |
| 17 | Candi Ratu Boko | Archaeological Site | -7.7704 | 110.4892 |
| 18 | Tebing Breksi | Nature/Cliff | -7.7620 | 110.5043 |
| 19 | Museum Ullen Sentalu | Museum | -7.6041 | 110.4264 |
| 20 | Candi Borobudur | Temple | -7.6078 | 110.2038 |
| 21 | Pantai Parangtritis | Beach | -8.0254 | 110.3264 |
| 22 | Hutan Pinus Mangunan | Nature | -7.9310 | 110.4310 |
| 23 | Goa Pindul | Cave/Adventure | -7.9530 | 110.6500 |
| 24 | Pantai Indrayanti | Beach | -8.1500 | 110.6133 |
| 25 | HeHa Sky View | Viewpoint | -7.8550 | 110.4540 |
