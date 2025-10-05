## 🇬🇧 `README_EN.md`

```markdown
# 🦺 Warehouse Safety Analysis with YOLO and Kalman Filtering

This project was developed as part of a **Master’s Thesis** focused on **automated risk detection between pedestrians and forklifts** in warehouse environments, using **YOLOv8**, **Kalman Filtering**, and **Leading Safety Indicators**.

---

## 🎯 Purpose
To develop a system that:
- Detects pedestrians and forklifts using YOLO in real-time.
- Calculates safety indicators such as:
  - **TTC (Time-To-Collision)**  
  - **TTCE (Time-To-Critical Event)**  
  - **DCE (Dynamic Criticality Estimate)**  
  - **FI (Fusion Index)**
- Visualizes risk evolution through plots and statistics.

---

## ⚙️ Main Files and Modules

| File / Module | Description |
|----------------|-------------|
| **Leading_Indicators_Cal.py** | Main script: takes a video input, performs detection, and computes all leading indicators and plots. |
| **FUCTIONS.py** | Contains all mathematical formulas, safety indicator equations, and Kalman filtering logic. |
| **Try_models.py** | Utility for quick YOLO model testing (e.g., YOLOv8n, YOLOv8m, or custom trained models). |
| **homography_points.py** | Selects 4 reference ground points and computes homography for pixel-to-meter conversion. |
| **dataset_Develop.py** | Generates a custom pose and detection dataset from one or multiple videos. |
| **CONVERT_TXT_to_JSON.py** | Converts pose datasets from `.txt` format to `.json` for platforms like Roboflow. |

---

## 📊 Output
- Calculates and stores **TTC, TTCE, DCE, FI** per frame and per interaction.
- Produces:
  - CSV files with per-pair time series.  
  - Mean values per frame.  
  - Histograms and risk category heatmaps.  

All results are stored under:
leading_indicators_results/
├── fi_timeseries_per_pair.csv
├── ttci_timeseries_per_pair.csv
├── means_per_time.csv
├── hist_w1_w2_matrix.png
└── outputs_crit_value/
├── fi_max_per_pair.csv
└── mins_per_pair.csv

yaml
Αντιγραφή κώδικα

---

## 🔧 Installation

```bash
git clone https://github.com/Thanasis19678/warehouse_safety_yolo.git
cd warehouse_safety_yolo
pip install -r requirements.txt
🚀 Run the project
bash
Αντιγραφή κώδικα
python Leading_Indicators_Cal.py
Select a video file to analyze.
Results are automatically stored under leading_indicators_results/.

🧠 Technologies
Python 3.10+

OpenCV

NumPy, Pandas, Matplotlib

Ultralytics YOLOv8

Kalman Filter

👨‍💻 Author
Thanasis [Surname]
Master’s Thesis — [University / Department Name]
📘 GitHub: Thanasis19678

✨ Warehouse Safety Analytics through Computer Vision and Predictive Indicators.

yaml
Αντιγραφή κώδικα

---

## 📦 Οδηγία για χρήση

1. Στο repo σου, δημιούργησε δύο νέα αρχεία:
   - `README_EL.md`
   - `README_EN.md`
2. Κάνε copy–paste το αντίστοιχο κείμενο.
3. (Προαιρετικά) στο κύριο `README.md`, μπορείς να προσθέσεις:
   ```markdown
   🇬🇷 [Διάβασε στα Ελληνικά](README_EL.md)  
   🇬🇧 [Read in English](README_EN.md)
