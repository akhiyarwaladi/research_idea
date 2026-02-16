# CS-Oriented Research Topics for SINTA 4-6 Journals
## (No IS/Sentiment/Kaggle — Niche Data Only)

---

## NICHE INDONESIAN DATA SOURCES

| Source | URL | Data Available |
|--------|-----|----------------|
| **BMKG Open Data** | data.bmkg.go.id | Earthquake (M5.0+), weather, climate — XML & REST API |
| **BNPB Disaster Data** | data.bnpb.go.id | Disaster incidents 2005-2024 by district + geoportal |
| **BPS Web API** | webapi.bps.go.id | 453K+ statistical datasets; Python package `stadata` |
| **Badan Pangan** | panelharga.badanpangan.go.id | Real-time food prices across all regions |
| **KPU Election Data** | opendata.kpu.go.id | 156 datasets: voter data, vote counts 2024 |
| **SINTA Scraper** | github.com/rendicahya/sinta3-scraper | Researcher profiles, publications, citations |
| **PDDIKTI API** | github.com/IlhamriSKY/PDDIKTI-kemdikbud-API | University, lecturer, student program data |
| **OSM Indonesia** | download.geofabrik.de/asia/indonesia.html | 1.5M km roads, POIs, buildings (Shapefile/PBF) |
| **peraturan.go.id** | peraturan.go.id | Full-text Indonesian laws & regional regulations |
| **IndoNLU** | huggingface.co/datasets/indonlp/indonlu | 12 NLU tasks + IndoBERT (4B words) |
| **NERGRIT Corpus** | GitHub | 2,090 sentences, 8 entity types for Indonesian NER |
| **Wikipedia ID dump** | dumps.wikimedia.org | Full Indonesian Wikipedia text corpus |
| **id-nlp-resource** | github.com/kmkurn/id-nlp-resource | Curated Indonesian NLP dataset collection |

---

## TOP 10 RECOMMENDED RESEARCH PROJECTS

### #1: Co-authorship Network Analysis from SINTA Data
- **Data**: Scrape SINTA using `sinta-scraper` Python package
- **Method**: Build co-authorship graph → centrality metrics → community detection (Louvain)
- **Tools**: Python, NetworkX, matplotlib
- **Why**: Completely unique self-collected data, pure graph theory, zero GPU

### #2: Topic Modeling on Indonesian Government Regulations (LDA)
- **Data**: Scrape from peraturan.go.id
- **Method**: Preprocessing with Sastrawi stemmer → LDA → pyLDAvis visualization
- **Tools**: Python, Gensim, pyLDAvis
- **Why**: Niche legal text, NLP beyond sentiment, no GPU

### #3: TSP on Indonesian Tourism Routes Using ACO + OpenStreetMap
- **Data**: Geofabrik OSM download + tourist POIs
- **Method**: Implement ACO → compare with GA and brute-force
- **Tools**: Python, OSMnx, NetworkX
- **Why**: Real road data, real problem, pure algorithm research

### #4: WCAG Accessibility Audit of Indonesian University Websites
- **Data**: Automated scan of 50+ university websites
- **Method**: aXe-core scanning → statistical analysis of violations
- **Tools**: Node.js (aXe-core), Python
- **Why**: Timely topic, easy, genuinely useful findings

### #5: Steganography Comparison (LSB vs DCT) on Batik Images
- **Data**: Self-photographed batik patterns
- **Method**: Implement LSB and DCT → measure PSNR, MSE, SSIM, capacity
- **Tools**: Python, OpenCV, NumPy
- **Why**: Indonesian cultural element, classic CS topic

### #6: Earthquake Spatiotemporal Clustering Using BMKG Data
- **Data**: BMKG earthquake catalog (data.bmkg.go.id)
- **Method**: DBSCAN spatiotemporal clustering → seismic zone mapping
- **Tools**: Python, scikit-learn, folium
- **Why**: Unique geophysical data, genuinely interesting patterns

### #7: Lightweight Crypto Benchmarking on Arduino/RPi
- **Data**: Self-generated test vectors
- **Method**: Implement AES-128, PRESENT, SIMON, SPECK → benchmark speed/memory/energy
- **Tools**: C/Arduino IDE or Python on RPi
- **Why**: Practical IoT security, empirical benchmarking

### #8: NER for Indonesian Disaster News
- **Data**: Scrape from BNPB or Indonesian news portals
- **Method**: CRF-based NER → extract location/date/disaster type → evaluate P/R/F1
- **Tools**: Python, sklearn-crfsuite
- **Why**: Domain-specific NLP, unique corpus

### #9: Batik Pattern Classification Using GLCM + Traditional ML
- **Data**: Self-photographed batik samples
- **Method**: GLCM texture features → KNN/SVM classification
- **Tools**: Python, OpenCV, scikit-learn
- **Why**: Cultural heritage + classic image processing, no GPU

### #10: Food Price Forecasting Using Fuzzy Time Series + Badan Pangan Data
- **Data**: panelharga.badanpangan.go.id (daily food prices)
- **Method**: Chen's fuzzy time series → compare with ARIMA
- **Tools**: Python, NumPy, matplotlib
- **Why**: Real economic data, forecasting with lightweight math

---

## ALL 76 PAPER TITLE IDEAS BY CATEGORY

### A. Graph Theory & Network Analysis (10 ideas)

1. Analisis Jaringan Kolaborasi Peneliti Indonesia Menggunakan SNA pada Data SINTA
2. Perbandingan Algoritma Dijkstra dan A* pada Jaringan Jalan Kota Bandung dari OpenStreetMap
3. Analisis Keterhubungan Jaringan Kereta Api Jawa Menggunakan Graph Centrality Metrics
4. Deteksi Komunitas pada Jaringan Retweet Isu Politik Indonesia (Girvan-Newman)
5. Analisis Small-World dan Scale-Free Properties pada Jaringan Kolaborasi Dosen Jawa Timur
6. Optimasi Rute Distribusi Logistik Bencana BNPB Menggunakan Floyd-Warshall
7. Analisis Keterhubungan Rute Penerbangan Domestik Indonesia Menggunakan Teori Graf
8. Visualisasi dan Analisis Jaringan Sitasi antar Jurnal SINTA Menggunakan PageRank
9. Perbandingan BFS dan DFS untuk Pencarian Jalur Transportasi Publik Jakarta
10. Analisis Topologi Jaringan Internet Exchange Point Indonesia

### B. NLP Beyond Sentiment (10 ideas)

11. Topic Modeling Peraturan Daerah Se-Indonesia Menggunakan LDA
12. Ekstraksi Kata Kunci Otomatis dari Abstrak Jurnal SINTA (TextRank vs TF-IDF)
13. Named Entity Recognition untuk Berita Bencana Alam Indonesia Menggunakan CRF
14. Deteksi Plagiarisme Dokumen Bahasa Indonesia (Rabin-Karp vs Cosine Similarity)
15. Analisis Kemiripan Teks Undang-Undang Indonesia (TF-IDF + Jaccard Similarity)
16. Pemodelan Topik Keluhan Masyarakat pada Portal LAPOR! (NMF)
17. Peringkasan Teks Otomatis Berita Indonesia Menggunakan TextRank
18. Perbandingan Algoritma Stemming Bahasa Indonesia (Nazief-Adriani vs Sastrawi)
19. Klasifikasi Topik Skripsi Mahasiswa Informatika (Naive Bayes + N-Gram)
20. Deteksi Bahasa Daerah (Jawa, Sunda, Batak) pada Teks Media Sosial (Character N-Gram)

### C. Steganography & Watermarking (7 ideas)

21. Perbandingan LSB dan DCT untuk Steganografi pada Citra Batik Digital
22. Steganografi Audio pada File Gamelan Menggunakan Phase Coding
23. Watermarking Citra Wayang Kulit Menggunakan DWT
24. Analisis Ketahanan Steganografi LSB terhadap Kompresi JPEG pada Citra Tenun
25. Steganografi Video pada Konten Budaya Indonesia (Frame-Based LSB)
26. Perbandingan Kapasitas Embedding pada Berbagai Format Citra Motif Batik
27. Digital Watermarking pada Dokumen Ijazah Elektronik (Spread Spectrum)

### D. Cryptography & Security (8 ideas)

28. Perbandingan Performa AES-128, PRESENT, SIMON pada Arduino untuk IoT
29. Analisis Kerentanan Website Pemerintah Daerah Jawa (OWASP ZAP)
30. Analisis Kekuatan Password Pengguna Internet Indonesia dari Data Breach Publik
31. Implementasi SHA-3 untuk Verifikasi Integritas Dokumen Digital Pemerintah
32. Deteksi Website Phishing Indonesia (Fitur URL + Random Forest)
33. Perbandingan Enkripsi Simetris dan Asimetris pada Data Sensor IoT
34. Analisis Keamanan TLS pada Website E-Government Indonesia
35. Visual Cryptography untuk Pengamanan Citra Tanda Tangan Digital

### E. Image Processing Without Deep Learning (8 ideas)

36. Klasifikasi Motif Batik Menggunakan GLCM + KNN
37. Deteksi Penyakit Daun Padi (Segmentasi Warna HSV + Morfologi)
38. Pengenalan Aksara Jawa pada Citra Dokumen (Template Matching + Moment Invariant)
39. Analisis Kualitas Beras Indonesia (Histogram Warna + Shape Features)
40. Deteksi Kematangan Buah Kelapa Sawit (Segmentasi HSV + Decision Tree)
41. Penghitungan Koloni Bakteri pada Petri Dish (Watershed Segmentation)
42. Identifikasi Jenis Ikan Air Tawar Indonesia (Shape Descriptor + Naive Bayes)
43. Deteksi Cacat Kain Tenun NTT (Edge Detection Canny + Analisis Frekuensi)

### F. Metaheuristic Optimization (8 ideas)

44. Optimasi Penjadwalan Mata Kuliah Menggunakan Algoritma Genetika
45. Perbandingan PSO dan ACO untuk Vehicle Routing Problem Distribusi Sembako
46. TSP Rute Wisata Yogyakarta Menggunakan ACO + Data OpenStreetMap
47. Optimasi Penempatan BTS Daerah Rural (PSO + Data Populasi BPS)
48. Penjadwalan Ujian Akhir Semester (Genetic Algorithm + Constraint Handling)
49. Optimasi Rute Pengumpulan Sampah Kota (GA + Graf Jalan OSM)
50. Perbandingan Simulated Annealing dan Tabu Search untuk Tata Letak Fasilitas
51. Optimasi Penjadwalan Shift Perawat Menggunakan Bee Colony

### G. Web Performance & API Analysis (6 ideas)

52. Audit Aksesibilitas Website PTN Indonesia (WCAG 2.1 + aXe)
53. Analisis Performa Website E-Government 34 Provinsi (Google Lighthouse)
54. Perbandingan Performa REST API: Express.js vs FastAPI vs Gin
55. Analisis Kecepatan Website Marketplace Indonesia (WebPageTest)
56. Evaluasi SEO dan Aksesibilitas Website UMKM Indonesia
57. Perbandingan HTTP/2 vs HTTP/3 pada Latency Jaringan Indonesia

### H. Recommender Systems (5 ideas)

58. Collaborative Filtering Rekomendasi Buku pada Data Peminjaman Perpustakaan
59. Content-Based Filtering Produk UMKM Indonesia (Data Tokopedia)
60. Perbandingan User-Based vs Item-Based CF untuk Rekomendasi Mata Kuliah
61. Hybrid Filtering Destinasi Wisata Indonesia (Data OpenTripMap)
62. Content-Based Filtering Menu Makanan Khas Daerah (Berbasis Komposisi Bahan)

### I. Simulation & Cellular Automata (5 ideas)

63. Simulasi Evakuasi Gempa pada Gedung Kampus (Cellular Automata)
64. Model Penyebaran Kebakaran Hutan Kalimantan (CA + Data Hotspot BMKG)
65. Simulasi Arus Lalu Lintas Persimpangan Surabaya (Agent-Based Modeling)
66. Simulasi Penyebaran DBD di Jakarta (Model SIR Berbasis Agen + Data Kemenkes)
67. Model CA Pertumbuhan Urban Sprawl Jabodetabek

### J. Fuzzy Logic & Control (4 ideas)

68. Penilaian Risiko Banjir DAS di Jawa (Fuzzy Mamdani + Data BMKG)
69. Kontrol Lampu Lalu Lintas Adaptif (Fuzzy Controller — Simulasi)
70. Penentuan Kualitas Air Sungai (Fuzzy Logic + Parameter Fisika-Kimia)
71. Prediksi Produksi Padi Nasional (Fuzzy Time Series + Data BPS)

### K. Miscellaneous CS (5 ideas)

72. Perbandingan Algoritma Kompresi Huffman dan LZW pada Teks Bahasa Indonesia
73. Algoritma Rabin-Karp untuk Pencocokan String pada Pencarian Hadits Indonesia
74. DSL untuk Konfigurasi Smart Home Berbasis Raspberry Pi
75. Performa Algoritma Sorting pada Dataset Harga Pangan Indonesia
76. Sequence Alignment DNA Padi Varietas Indonesia (Needleman-Wunsch vs Smith-Waterman)

---

## TARGET CS-ORIENTED JOURNALS

### SINTA 4
| Journal | Focus | URL |
|---------|-------|-----|
| JISKA | NLP, security, crypto, algorithms | ejournal.uin-suka.ac.id |
| MALCOM | ML, computer science | journal.irpi.or.id |
| Mantik | Steganography, algorithms, NLP | iocscience.org |
| Sinkron | Optimization, web scraping, CS | jurnal.polgan.ac.id |
| JISA | Algorithm analysis, network | trilogi.ac.id |
| Khazanah Informatika | Image processing, batik | journals.ums.ac.id |
| JOIN | PSO, ACO, scraping | join.if.uinsgd.ac.id |
| Jurnal RESTI | Topic modeling, NLP | jurnal.iaii.or.id |
| KLIK | CS applications | djournals.com/klik |

### SINTA 5
| Journal | URL |
|---------|-----|
| JATISI | jurnal.mdp.ac.id |
| JITET | journal.eng.unila.ac.id |
| JISTI | SINTA portal |
| Jurnal Media Informatika | SINTA portal |

---

## WHAT MAKES IT "CS" NOT "IS"

- Focus on **algorithm comparison/analysis** (not just applying a tool)
- Include **complexity discussion** or **performance metrics** (time, accuracy, PSNR)
- Use **formal methods or math models** (graph theory, fuzzy logic, info theory)
- The contribution is a **technical finding** (not a system/application)

**Avoid**: rancang bangun, waterfall/SDLC, SPK with AHP/SAW, CRUD web apps, sentiment on Twitter, Kaggle/UCI data.

---

*Generated: February 14, 2026*
