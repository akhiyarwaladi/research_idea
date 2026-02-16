# Deep Research: SINTA 4, 5, 6 Computer Science Journals -- Topics, Quality Bar, and Actionable Research Ideas

---

## 1. UNDERSTANDING SINTA LEVELS 4, 5, 6

### What is SINTA?

SINTA (Science and Technology Index) is the Indonesian journal ranking system managed by the Ministry of Education, Culture, Research and Technology (Kemdikbudristek). It ranks national journals into six tiers (S1-S6) based on accreditation scores evaluated through the ARJUNA portal, governed by Permenristekdikti No. 9/2018.

### Score Ranges and Characteristics

| Level | Score Range | Key Characteristics |
|-------|-----------|---------------------|
| **S1** | >85 (or Scopus-indexed) | Highest quality, international standards |
| **S2** | 71-85 | High quality, strong peer review |
| **S3** | 61-70 | Good quality, nationally competitive |
| **S4** | 51-60 | Nationally recognized but needs improvement; limited scientific impact |
| **S5** | 41-50 | Under development/evaluation; requires further development |
| **S6** | 31-40 | Lowest tier; needs significant quality improvement |

### What S4-S6 Means in Practice

- **Review process**: Peer-reviewed but with shorter turnaround (weeks, not months)
- **Acceptance rate**: Generally higher than S1-S3
- **Language**: Mix of Bahasa Indonesia and English accepted
- **Novelty bar**: Lower novelty requirements; application/implementation papers are welcome
- **Methodology**: Simple, well-known methods applied to new domains/cases are acceptable
- **Impact**: Primarily used by Indonesian lecturers (dosen) for credit (kum) requirements
- **Recent finding (2026)**: A study published in *Information Research* found that statistically, there are at most 4 distinguishable groups (not 6) based on citation metrics, meaning S4-S6 journals have very similar citation distributions

### Assessment Criteria (8 Elements)

1. Journal title, aims and scope
2. Publisher identity
3. Editorial and journal management
4. Article quality
5. Writing style/quality
6. PDF and e-journal format
7. Regularity of publication
8. Online availability and indexing

---

## 2. COMMON CS TOPICS IN SINTA 4-6 JOURNALS

Based on analysis of actual published papers in journals like KLIK (S4), CO-SCIENCE (S4), JUTISI (S4), BITS (S4), Resolusi (S5), JISTI (S5), IJCS (S5), and others:

### Top 10 Most Frequently Published Topic Categories

1. **Decision Support Systems (SPK)** -- ~11% of all publications
2. **Web/Mobile Application Development (Rancang Bangun)** -- ~31% of publications
3. **Sentiment Analysis / Text Classification** -- Very frequent
4. **Expert Systems (Sistem Pakar)** -- Consistently popular
5. **Data Mining / Classification** -- Comparing ML algorithms
6. **Information System Design** -- Business/organizational systems
7. **UI/UX Evaluation** -- SUS, Heuristic Evaluation
8. **Network/QoS Analysis** -- Wireless, ISP comparison
9. **IoT Prototypes** -- Arduino/ESP32-based
10. **Image Processing / Simple Computer Vision** -- Often transfer learning

---

## 3. LIGHTWEIGHT RESEARCH APPROACHES (No GPU, Laptop-Level)

### Category A: Decision Support Systems (SPK) -- EASIEST TO PUBLISH

**Methods commonly used:**
- SAW (Simple Additive Weighting)
- WP (Weighted Product)
- TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
- AHP (Analytic Hierarchy Process)
- MOORA (Multi-Objective Optimization by Ratio Analysis)
- ORESTE
- Composite Performance Index (CPI)
- ELECTRE
- PROMETHEE
- Combinations: AHP-TOPSIS, AHP-SAW, WP+Entropy

**Specific actionable topic ideas:**
1. SPK Pemilihan Laptop Terbaik untuk Mahasiswa Menggunakan Metode TOPSIS dan Entropy
2. SPK Penentuan Lokasi Usaha Kuliner Menggunakan AHP dan WP
3. SPK Pemilihan Platform E-Learning Terbaik dengan Metode SAW dan AHP
4. SPK Seleksi Penerima Beasiswa Menggunakan Kombinasi MOORA-Entropy
5. SPK Pemilihan Smartphone Terbaik Berdasarkan Harga dan Spesifikasi dengan TOPSIS
6. SPK Penentuan Prioritas Perbaikan Infrastruktur Desa Menggunakan AHP-SAW
7. SPK Rekomendasi Destinasi Wisata Menggunakan Metode Weighted Product
8. Analisis Komparatif Metode AHP-TOPSIS dan AHP-SAW dalam Pemilihan Karyawan Terbaik
9. SPK Pemilihan Supplier Bahan Baku UMKM Menggunakan MOORA dan ROC
10. SPK Penentuan Dompet Digital Terbaik Menggunakan AHP

**Requirements:** Spreadsheet-level computation, 5-15 criteria, 10-30 alternatives, questionnaire data

---

### Category B: Expert Systems (Sistem Pakar) -- VERY POPULAR

**Methods commonly used:**
- Forward Chaining
- Backward Chaining
- Certainty Factor (CF)
- Dempster-Shafer
- Fuzzy Logic (Tsukamoto, Mamdani, Sugeno)
- Natural Language Processing (simple rule-based)
- Combinations: Forward Chaining + Certainty Factor

**Specific actionable topic ideas:**
1. Sistem Pakar Diagnosa Penyakit Tanaman Cabai Menggunakan Certainty Factor
2. Sistem Pakar Identifikasi Kerusakan Laptop/PC Berbasis Forward Chaining
3. Sistem Pakar Diagnosa Penyakit Kulit Menggunakan Forward Chaining dan Certainty Factor
4. Sistem Pakar Penentuan Jenis Perawatan Wajah Berdasarkan Tipe Kulit
5. Sistem Pakar Rekomendasi Diet Berdasarkan BMI dan Riwayat Kesehatan
6. Sistem Pakar Identifikasi Hama pada Tanaman Padi Berbasis Web
7. Sistem Pakar Diagnosa Kerusakan Sepeda Motor dengan Metode Backward Chaining
8. Sistem Pakar Deteksi Gangguan Kesehatan Mental Menggunakan Dempster-Shafer
9. Sistem Pakar Pemilihan Jenis Olahraga Berdasarkan Kondisi Fisik dan Usia
10. Sistem Pakar Identifikasi Kualitas Biji Kopi Menggunakan Fuzzy Logic

**Requirements:** Rule base from domain expert (5-30 rules), simple if-then logic, PHP/Python web app

---

### Category C: Sentiment Analysis & Text Classification -- HIGH DEMAND

**Methods commonly used:**
- Naive Bayes
- SVM (Support Vector Machine)
- K-NN
- Random Forest
- Logistic Regression
- TF-IDF for feature extraction
- TextBlob, VADER for comparison
- Orange Data Mining (no-code tool)

**Specific actionable topic ideas:**
1. Analisis Sentimen Ulasan Pengguna Aplikasi MyTelkomsel Menggunakan Naive Bayes dan SVM
2. Klasifikasi Sentimen Twitter terhadap Kebijakan Publik Menggunakan Random Forest
3. Perbandingan Algoritma Naive Bayes, KNN, dan SVM untuk Klasifikasi Sentimen Review Tokopedia
4. Analisis Sentimen Opini Masyarakat terhadap Transportasi Online Menggunakan Logistic Regression
5. Text Mining Ulasan Google Play Store untuk Analisis Sentimen Aplikasi Pendidikan
6. Klasifikasi Berita Hoax Menggunakan Naive Bayes dengan TF-IDF
7. Analisis Sentimen Media Sosial tentang Pariwisata Daerah Menggunakan VADER dan TextBlob
8. Perbandingan Performa Naive Bayes dan Random Forest dalam Klasifikasi Spam Email Bahasa Indonesia
9. Analisis Sentimen Ulasan Produk UMKM di Shopee Menggunakan SVM
10. Deteksi Cyberbullying pada Komentar Instagram Menggunakan KNN dan Naive Bayes

**Requirements:** Web scraping (Tweepy/Selenium), 500-5000 data points (enough for these journals), sklearn, small laptop-level dataset

---

### Category D: Web/Mobile Application Development (Rancang Bangun) -- LARGEST VOLUME

**Methodologies commonly used:**
- Waterfall
- Prototype
- Agile/Scrum (simplified)
- RAD (Rapid Application Development)
- User Centered Design (UCD)
- Design Thinking

**Specific actionable topic ideas:**
1. Rancang Bangun Sistem Informasi Inventaris Barang Berbasis Web Menggunakan Laravel
2. Pengembangan Aplikasi Pemesanan Makanan UMKM Berbasis Android dengan Metode Prototype
3. Sistem Informasi Pengelolaan Keuangan Masjid Berbasis Web
4. Aplikasi Pencatatan Kalori Harian Berbasis Mobile dengan Arsitektur MVVM
5. Rancang Bangun Aplikasi Presensi Berbasis Foto Selfie dengan Geolocation
6. Sistem Informasi Manajemen Perpustakaan Desa Berbasis Web
7. Pengembangan Aplikasi Monitoring Tugas Mahasiswa Menggunakan React Native
8. Rancang Bangun E-Commerce UMKM Batik dengan Payment Gateway
9. Aplikasi Reservasi Lapangan Olahraga Berbasis Web Menggunakan Metode Waterfall
10. Sistem Informasi Pengelolaan Donasi Sembako dengan Metode SAW

**Requirements:** PHP/Laravel or Android Studio, MySQL, basic web hosting

---

### Category E: UI/UX Evaluation & Usability Testing -- GROWING TREND

**Methods commonly used:**
- System Usability Scale (SUS) -- Most popular
- Heuristic Evaluation (Nielsen's 10 Heuristics)
- User Experience Questionnaire (UEQ)
- USE Questionnaire
- Design Sprint
- A/B Testing (simple)
- Likert scale surveys

**Specific actionable topic ideas:**
1. Evaluasi Usability Website Akademik Universitas X Menggunakan System Usability Scale
2. Analisis Komparatif UI/UX Aplikasi Ojek Online (Gojek vs Grab vs Maxim) dengan SUS
3. Evaluasi Heuristik Aplikasi E-Government Kabupaten X
4. Redesign UI/UX Website SIAKAD Menggunakan User Centered Design dan Evaluasi SUS
5. Analisis User Experience Aplikasi KAI Access Menggunakan UEQ
6. Usability Testing Aplikasi Mobile Banking BRI dan BCA Menggunakan SUS
7. Evaluasi Heuristik dan SUS pada Aplikasi Manajemen Keuangan Pribadi
8. Perbandingan Usability Platform LMS (Moodle vs Google Classroom) dengan SUS
9. Analisis UX Website Pariwisata Daerah Menggunakan Heuristic Evaluation
10. Evaluasi Usability Aplikasi PeduliLindungi/SatuSehat Menggunakan SUS dan UEQ

**Requirements:** Google Forms for surveys, 30-100 respondents, SUS scoring formula (spreadsheet)

---

### Category F: Systematic Literature Review (SLR) -- MINIMAL RESOURCES NEEDED

**Methods commonly used:**
- PRISMA protocol
- Kitchenham methodology
- VOSviewer for bibliometric visualization
- Publish or Perish for citation analysis

**Specific actionable topic ideas:**
1. Systematic Literature Review: Penerapan Machine Learning pada Pertanian di Indonesia (2019-2024)
2. SLR: Metode Pengembangan Aplikasi Mobile di Jurnal Nasional Indonesia
3. SLR: Tren Penggunaan Decision Support System di Sektor Pendidikan Indonesia
4. Bibliometric Analysis: Riset IoT di Indonesia Menggunakan VOSviewer
5. SLR: Penerapan Gamifikasi dalam Pembelajaran Berbasis Teknologi
6. SLR: Implementasi Chatbot sebagai Virtual Assistant - Tren dan Tantangan
7. Literature Review: Perbandingan Framework Pengembangan Web (Laravel, Django, Express.js)
8. SLR: Penerapan Augmented Reality dalam Pendidikan di Indonesia
9. Bibliometric Analysis: Tren Penelitian Sistem Informasi di Jurnal SINTA
10. SLR: Metode Evaluasi Usability pada Aplikasi Mobile di Indonesia

**Requirements:** Access to Google Scholar, Scopus (via institutional access), VOSviewer (free), Publish or Perish (free)

---

### Category G: Simple Data Mining / Machine Learning (Small Datasets) -- VERY POPULAR

**Methods commonly used:**
- C4.5 / Decision Tree
- K-Means Clustering
- Naive Bayes
- K-NN
- Random Forest
- Linear/Logistic Regression
- Apriori (association rules)
- FP-Growth
- PCA + Clustering combinations
- Fuzzy C-Means
- DBSCAN

**Specific actionable topic ideas:**
1. Klasifikasi Tingkat Kepuasan Pelanggan E-Commerce Menggunakan C4.5
2. Clustering Profil Mahasiswa Berdasarkan IPK dan Keaktifan Organisasi Menggunakan K-Means
3. Prediksi Kelulusan Tepat Waktu Mahasiswa Menggunakan Naive Bayes
4. Market Basket Analysis pada Transaksi Minimarket Menggunakan Algoritma Apriori
5. Segmentasi Pelanggan Toko Online Menggunakan RFM Analysis dan K-Means
6. Klasifikasi Kualitas Air Sungai Menggunakan Random Forest dan Decision Tree
7. Prediksi Harga Rumah di Kota X Menggunakan Regresi Linear
8. Analisis Pola Pembelian Konsumen Menggunakan FP-Growth
9. Clustering Wilayah Berdasarkan Indeks Pembangunan Manusia Menggunakan DBSCAN
10. Perbandingan Naive Bayes, KNN, dan Random Forest untuk Prediksi Risiko Diabetes

**Requirements:** sklearn, Pandas, publicly available datasets (Kaggle, BPS, UCI), 100-10000 rows is fine

---

### Category H: Network Analysis & QoS -- NICHE BUT STEADY

**Specific actionable topic ideas:**
1. Analisis QoS Jaringan Wi-Fi Kampus Menggunakan Metode Pengukuran ITU-T
2. Perbandingan Performa VPN WireGuard dan OpenVPN pada Jaringan Lokal
3. Social Network Analysis Kolaborasi Peneliti Indonesia Menggunakan Data SINTA
4. Analisis Kinerja Jaringan VLAN pada Infrastruktur Kampus
5. Perbandingan Throughput Protokol Routing OSPF dan EIGRP Menggunakan GNS3
6. Social Network Analysis Interaksi Pengguna Twitter pada Hashtag Tertentu
7. Monitoring Ketersediaan Server Web Menggunakan Zabbix dan Grafana
8. Analisis Performa Website E-Government Menggunakan GTmetrix dan PageSpeed Insights
9. Implementasi Firewall pada Jaringan UMKM Menggunakan Mikrotik
10. Perbandingan Kualitas Layanan ISP di Indonesia Menggunakan Speedtest dan SNA

**Requirements:** Wireshark, GNS3/Packet Tracer, free network tools, speedtest data

---

### Category I: GIS & Mapping -- GROWING AREA

**Specific actionable topic ideas:**
1. Pemetaan Daerah Rawan Banjir Menggunakan QGIS dan Data BNPB
2. Sistem Informasi Geografis Persebaran Fasilitas Kesehatan di Kabupaten X
3. Smart Mapping Lokasi UMKM Berbasis WebGIS Menggunakan Leaflet.js
4. SIG Pemetaan Potensi Wisata Daerah Berbasis Web
5. Analisis Spasial Kepadatan Penduduk Menggunakan QGIS

**Requirements:** QGIS (free), Leaflet.js, public geospatial data from BPS/BNPB

---

### Category J: IoT (Simple Prototypes) -- POPULAR BUT NEEDS MINIMAL HARDWARE

**Specific actionable topic ideas:**
1. Prototype Monitoring Suhu dan Kelembaban Ruangan Berbasis Arduino dan IoT
2. Sistem Pemantauan Ketinggian Air Tandon Berbasis ESP32 dan Blynk
3. Smart Home Sederhana Menggunakan NodeMCU dan MQTT Protocol
4. Monitoring Kualitas Udara Menggunakan Sensor MQ-135 dan ThingSpeak
5. Sistem Irigasi Otomatis Berbasis Arduino dengan Notifikasi Telegram

**Requirements:** Arduino/ESP32 (~Rp50-100k), sensors (~Rp20-50k each), Blynk/ThingSpeak (free tier)

---

## 4. TRENDING TOPICS IN SINTA CS JOURNALS (2024-2026)

Based on actual published papers analysis:

### Hot Topics (High Volume, Easy to Publish)
1. **Sentiment Analysis** -- Using Twitter/Instagram/Play Store data
2. **Decision Support Systems** -- New domain applications (SPK for anything)
3. **Chatbot Development** -- Rule-based or API-driven (not training from scratch)
4. **Simple ML Classification Comparisons** -- "Perbandingan Algoritma X vs Y vs Z"
5. **Web Application Development** -- Business process automation

### Emerging Topics (Growing Interest)
1. **AI in Education** -- Needs assessment, implementation studies
2. **Gamification** -- Educational apps, learning platforms
3. **Augmented Reality** -- Education, tourism (using AR.js, Vuforia)
4. **Blockchain Concepts** -- Surveys, feasibility studies (no actual implementation needed)
5. **Digital Transformation** -- UMKM/SME digitalization studies

### Evergreen Topics (Always Accepted)
1. Expert Systems with Certainty Factor
2. Waterfall/Prototype web apps
3. Information system design
4. Usability testing with SUS
5. Network QoS analysis

---

## 5. SPECIFIC EXAMPLES OF ACTUAL PUBLISHED PAPER TITLES (SINTA 4-6)

### From KLIK Journal (SINTA 4, 2024):
- "Perancangan dan Implementasi UI/UX Website Edukasi Kesehatan Balita Menggunakan Metode Design Thinking"
- "Sistem Pakar Deteksi Kerusakan Laptop Menggunakan Forward/Backward Chaining"
- "Analisis Sentimen Terhadap Rangka E-SAF Honda Menggunakan Naive Bayes"
- "Sistem Informasi Pengelolaan Donasi Sembako Menggunakan SAW"
- "Kombinasi Metode MOORA dan Rank Order Centroid untuk Pemilihan Supplier"
- "Penerapan Logika Fuzzy Untuk Peramalan Penjualan"
- "Sistem Pendukung Keputusan Menentukan Sales Terbaik Menggunakan SAW"
- "Rancang Bangun Sistem Informasi Akuntansi Menggunakan Metode Waterfall"
- "Analisa QoS Jaringan Wireless"
- "Aplikasi Pemesanan Tiket Di Wisata Alam"

### From CO-SCIENCE Journal (SINTA 4, 2024):
- "Analisis Sentimen Ulasan Pelanggan Menggunakan Algoritma Naive Bayes pada Aplikasi Gojek"
- "Customer Churn Prediction pada Sektor Perbankan dengan Logistic Regression dan Random Forest"
- "Aplikasi Pencatatan Kalori Harian Berbasis Android Dengan Arsitektur MVVM"

### From JUTISI Journal (SINTA 4, 2025):
- "Sistem Pakar Menggunakan NLP Untuk Diagnosa Penyakit Tanaman Cabai"
- "Implementasi Algoritma Naive Bayes Untuk Klasifikasi Sentimen Ulasan KAI Access"
- "Perencanaan Strategis Sistem Informasi Menggunakan Ward and Peppard"
- "Rancang Bangun Aplikasi Pemesanan Kolam Pemancingan dengan AHP"

### From Various SINTA 5 Journals (2024-2025):
- "Smart Mapping Berbasis QGIS: Pemetaan Digital Daerah Rawan Bencana"
- "Penerapan Metode Gross Untuk Perhitungan PPH 21 pada Sistem Informasi Penggajian"
- "Segmentasi Wilayah Indonesia Berdasarkan IHK Menggunakan AHC dan Spectral Clustering"
- "Prediksi Harga Penutupan Saham BBRI dengan Model Hybrid LSTM-XGBoost"
- "Prototype Perancangan Alat Pendeteksi Ketinggian Tandon Air Berbasis Arduino"
- "Desain dan Implementasi Jaringan Wireless Point to Multipoint"

---

## 6. RECOMMENDED STRATEGY FOR PUBLISHING

### Step 1: Choose Your Category
Pick from Categories A-J above based on your skills and available resources.

### Step 2: Select a Specific Topic
Choose a topic that applies a well-known method to a NEW domain or LOCAL case. The key formula for SINTA 4-6 papers is:

**[Known Method] + [New Application Domain/Local Context] = Publishable Paper**

Examples:
- SAW + Pemilihan Catering Terbaik di Kota X = Paper
- Naive Bayes + Sentimen terhadap Brand Lokal Y = Paper
- SUS + Evaluasi Website Dinas Z di Kabupaten W = Paper

### Step 3: Target Journals (Examples)

**SINTA 4 Journals (CS/IT):**
- KLIK: Kajian Ilmiah Informatika dan Komputer (djournals.com/klik)
- CO-SCIENCE (jurnal.bsi.ac.id/index.php/co-science)
- JUTISI (ojs.stmik-banjarbaru.ac.id/index.php/jutisi)
- Infotek (e-journal.hamzanwadi.ac.id/index.php/infotek)
- JUPITER (jurnal.polsri.ac.id/index.php/jupiter)
- Jurnal Tekinkom (jurnal.murnisadar.ac.id/index.php/Tekinkom)
- TIN: Terapan Informatika Nusantara
- JTOS: Jurnal Teknologi dan Open Source

**SINTA 5 Journals (CS/IT):**
- IJCS: Indonesian Journal of Computer Science (ijcs.net)
- Resolusi (djournals.com/resolusi)
- JISTI (journal.jisti.unipol.ac.id)
- JISAMAR (journal.stmikjayakarta.ac.id/index.php/jisamar)
- JuMIn: Jurnal Media Informatika (ejournal.sisfokomtek.org)
- JUPTI: Jurnal Publikasi Teknik Informatika (journalcenter.org/index.php/jupti)
- JDMIS: Journal of Data Mining and Information Systems
- JISICOM (journal.stmikjayakarta.ac.id/index.php/jisicom)

**SINTA 6 Journals:**
- Various university-specific journals; check SINTA portal directly

### Step 4: Methodology Template

For a typical SINTA 4-6 paper, follow this structure:

1. **Pendahuluan** -- Problem statement, why this matters locally
2. **Tinjauan Pustaka** -- 10-20 references, cite 3-5 similar studies
3. **Metodologi** -- Method description (SAW/AHP/Naive Bayes/etc.), data collection
4. **Hasil dan Pembahasan** -- Results, accuracy metrics, confusion matrix / ranking table
5. **Kesimpulan** -- Summary and future work

### Step 5: Timeline
- Data collection: 1-2 weeks
- Implementation: 1-2 weeks
- Writing: 1-2 weeks
- Review process: 2-8 weeks (S4-S6 typically faster)
- **Total: 1-3 months from start to publication**

---

## 7. TOP 20 EASIEST SPECIFIC PAPER IDEAS (Ranked by Feasibility)

| # | Topic | Method | Resources Needed | Time |
|---|-------|--------|-----------------|------|
| 1 | SPK Pemilihan Laptop Terbaik untuk WFH | SAW + WP | Spreadsheet, 5 criteria, 10 alternatives | 2 weeks |
| 2 | Evaluasi Usability Website SIAKAD Kampus | SUS | Google Forms, 30 respondents | 2 weeks |
| 3 | SLR: Tren Penelitian SPK di Indonesia 2019-2024 | PRISMA + VOSviewer | Google Scholar, VOSviewer | 3 weeks |
| 4 | Sentimen Ulasan Aplikasi di Play Store | Naive Bayes | Python, scrape 1000 reviews | 2 weeks |
| 5 | Sistem Pakar Diagnosa Penyakit Tanaman X | Forward Chaining + CF | PHP, 15-25 rules from expert | 3 weeks |
| 6 | SPK Pemilihan Beasiswa Mahasiswa | AHP-TOPSIS | Spreadsheet + simple web app | 2 weeks |
| 7 | Perbandingan 3 Algoritma Klasifikasi (NB/KNN/RF) | Accuracy comparison | sklearn, Kaggle dataset | 2 weeks |
| 8 | Rancang Bangun Sistem Informasi X Berbasis Web | Waterfall + Laravel | PHP/Laravel, MySQL | 3 weeks |
| 9 | Analisis QoS Jaringan WiFi Kampus | QoS metrics (ITU-T) | Wireshark, speedtest | 2 weeks |
| 10 | Clustering Mahasiswa Berdasarkan IPK | K-Means | sklearn, institutional data | 2 weeks |
| 11 | Evaluasi Heuristik Aplikasi E-Government | Nielsen's 10 Heuristics | 3-5 evaluators, scoring sheets | 2 weeks |
| 12 | Market Basket Analysis Data Transaksi Toko | Apriori | Python, transaction data | 2 weeks |
| 13 | Perbandingan UI/UX Aplikasi Ojek Online | SUS + Heuristic Eval | Surveys, 50 respondents | 2 weeks |
| 14 | Pemetaan Fasilitas Kesehatan Berbasis WebGIS | QGIS + Leaflet.js | Free GIS data from BPS | 3 weeks |
| 15 | Social Network Analysis Hashtag Twitter | SNA + Gephi | Python + Gephi (free) | 2 weeks |
| 16 | Prediksi Kelulusan Mahasiswa | Decision Tree C4.5 | Weka/sklearn, academic data | 2 weeks |
| 17 | Sistem Pakar Rekomendasi Diet | Fuzzy Logic | PHP/Python, diet rules | 3 weeks |
| 18 | Aplikasi Presensi Foto Selfie + Geolocation | Prototype method | Android Studio/Flutter | 3 weeks |
| 19 | Analisis Sentimen Kebijakan Publik di Twitter | SVM + TF-IDF | Python, 2000 tweets | 2 weeks |
| 20 | Monitoring IoT Suhu Ruangan | Arduino + ESP32 | Hardware ~Rp100k, Blynk | 3 weeks |

---

## 8. KEY SINTA CS JOURNAL REFERENCES

- SINTA Portal: https://sinta.kemdikbud.go.id/
- KLIK Journal: https://djournals.com/klik
- CO-SCIENCE: https://jurnal.bsi.ac.id/index.php/co-science
- BITS Journal: https://ejurnal.seminar-id.com/index.php/bits
- JUTISI: https://journal.maranatha.edu/index.php/jutisi
- Sinkron Journal: https://jurnal.polgan.ac.id/index.php/sinkron
- Infotek: https://e-journal.hamzanwadi.ac.id/index.php/infotek
- IJCS: http://ijcs.net
- Resolusi: https://djournals.com/resolusi

---

*Document generated: February 14, 2026*
*Based on deep research of SINTA-indexed Indonesian CS journals, published papers from 2024-2026, and web search analysis.*
