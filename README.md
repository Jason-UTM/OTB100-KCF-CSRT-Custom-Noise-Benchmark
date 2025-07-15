# 🎯 OTB-100 Tracking Pipeline

![Python](https://img.shields.io/badge/python-v3.11+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11.0-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📖 Overview

This project implements a comprehensive tracking pipeline using the **OTB-100 dataset** to evaluate the performance of visual tracking algorithms (**KCF** and **CSRT**) under various noise and occlusion conditions. 

The pipeline applies **Gaussian** and **salt-and-pepper** noise with multiple occlusion levels (0.2, 0.4, 0.6) to video sequences, tracks objects using OpenCV trackers, and evaluates comprehensive metrics including Expected Average Overlap (EAO), Robustness, Precision, Tracking Time, FPS, Number of Frames, and Number of Failures.

## ✨ Features

- 📊 **Dataset**: Processes sequences from the OTB-100 dataset (e.g., Basketball)
- 🔍 **Trackers**: Evaluates KCF and CSRT trackers from OpenCV
- 🌪️ **Noise & Occlusion**: Applies gaussian and salt_pepper noise with occlusion levels 0.2, 0.4, and 0.6
- 📈 **Metrics**: Computes EAO, Robustness, Precision, TrackingTime, FPS, NumFrames, and NumFailures
- 💾 **Outputs**: Saves results as CSVs and videos with structured naming
- 👁️ **Visualization**: Real-time tracking display with ground truth and predicted bounding boxes
- ⚡ **Performance**: Processes Basketball sequence (~725 frames) in ~6.5 minutes on Core i5 8th Gen CPU

## 📁 Project Structure
```
📦 Traditional Tracker
├── 📄 config.json                      # Configuration file for paths, trackers, and settings
├── 🐍 main.py                          # Main script to load config and run pipeline
├── 🔧 main_noise_application.py        # Generates noisy frames
├── 🧹 main_purge.py                    # Purge noisy generated frames and folders
├── 📁 dataset/                         # Contains all data related
│   └── 📁 OTB100/                      # OTB-100 dataset directory
│       ├── 🏀 Basketball/  
│       │   ├── 🖼️ img/                 # Original frames
│       │   ├── 🌫️ img_gaussian_0.2/    # Noisy frames
│       │   └── 📋 groundtruth_rect.txt # Ground truth bounding boxes
├── 📊 results/                         # Output directory
│   └── 🏀 Basketball/  
│       ├── 📈 kcf_gaussian_0.2.csv     # Metrics for KCF, Gaussian noise, 0.2 occlusion
│       ├── 🎥 kcf_gaussian_0.2.avi     # Video output
│       └── ... 
├── 📖 README.md                        # This file
└── 📋 requirements.txt                 # All dependencies to run this project
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11.7
- pip (Python package installer)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Jason-UTM/OTB100-KCF-CSRT-Custom-Noise-Benchmark.git
   cd OTB100-KCF-CSRT-Custom-Noise-Benchmark
   ```

2. **Download OTB-100 Dataset**
   - Visit [Kaggle OTB2015 Dataset](https://www.kaggle.com/datasets/zly1402875051/otb2015)
   - Download the dataset (requires Kaggle account)
   - Extract the downloaded file to create the following structure:
     ```
     dataset/
     └── OTB100/
         ├── Basketball/
         ├── Biker/
         ├── Bird1/
         └── ... (other sequences)
     ```

3. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the complete pipeline**
   ```bash
   python main.py
   ```
   This will automatically:
   - Generate noisy sequences with configured noise types and occlusion levels
   - Run tracking evaluation on all combinations
   - Generate analysis plots and consolidated metrics tables
   - Save results as CSV files and videos

### Alternative: Run Individual Components

If you want to run specific parts of the pipeline separately:

- **Generate noisy sequences only:**
  ```bash
  python main_noise_application.py
  ```

- **Run tracking evaluation only:** (requires noisy sequences)
  ```bash
  python main_tracking.py
  ```

- **Generate analysis plots only:** (requires tracking results)
  ```bash
  python main_analysis.py
  ```

- **Purge generated noisy frames:**
  ```bash
  python main_purge.py
  ```

## ⚙️ Configuration

Edit `config.json` to customize your tracking pipeline:

```json
{
  "DATASET_PATH": "./dataset/OTB100",
  "RESULTS_DIR": "./results",
  "SEQUENCES": ["Basketball"],
  "TRACKERS": ["KCF", "CSRT"],
  "NOISE_TYPES": ["gaussian", "salt_pepper"],
  "OCCLUSION_LEVELS": [0.2, 0.4, 0.6],
  "VISUALIZE": true,
  "SAVE_VIDEOS": true,
  "FAILURE_THRESHOLD": 0.1,
  "FRAME_DELAY": 25
}
```

## 📊 Output Examples

The pipeline generates:
- **Consolidated CSV**: `results/Basketball/Basketball_metrics_table.csv` (all tracker results in one file)
- **Analysis Plots**: 
  - `results/Basketball/Basketball_metrics_bar.png` (bar charts)
  - `results/Basketball/Basketball_eao_trends.png` (EAO trends)
  - `results/Basketball/Basketball_precision_vs_robustness.png` (scatter plot)
- **Video Output**: `results/Basketball/kcf_gaussian_0.2.avi` (tracking videos)
- **Markdown Table**: `results/Basketball/Basketball_metrics_table.md` (formatted results)

## 📄 License

This project is licensed under the **MIT License**.

## 🙏 Acknowledgments

- **OTB-100 dataset**: http://cvlab.hanyang.ac.kr/tracker_benchmark
- **OpenCV** for tracking algorithms