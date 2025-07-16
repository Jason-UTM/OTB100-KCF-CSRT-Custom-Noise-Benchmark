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
├── 🐍 main.py                          # Main script to run complete pipeline
├── 🎯 main_demo.py                     # Quick demonstration script (Deer sequence)
├── 🔧 main_noise_application.py        # Generates noisy frames only
├── 🎯 main_tracking.py                 # Runs tracking evaluation only
├── 📊 main_analysis.py                 # Generates analysis plots only
├── 🧹 main_purge.py                    # Purge noisy generated frames and folders
├── 🔧 services/                        # Core pipeline modules
│   ├── config.py                       # Configuration loader
│   ├── noise.py                        # Noise and occlusion application
│   ├── benchmark.py                    # Tracking evaluation
│   ├── analysis.py                     # Results analysis and plotting
│   ├── results.py                      # Results saving utilities
│   ├── data.py                         # Dataset loading utilities
│   └── helper.py                       # Helper functions
├── 📁 ./dataset/                        # Contains all data related
│   └── 📁 OTB100/                      # OTB-100 dataset directory
│       ├── 🏀 Basketball/  
│       │   ├── 🖼️ img/                 # Original frames
│       │   ├── 🌫️ img_gaussian_0.2/    # Generated noisy frames
│       │   ├── 🌫️ img_gaussian_0.4/    # Generated noisy frames
│       │   ├── 🌫️ img_salt_pepper_0.2/ # Generated noisy frames
│       │   ├── 🌫️ ... (other combinations)
│       │   └── 📋 groundtruth_rect.txt # Ground truth bounding boxes
├── 📊 ./results/                        # Output directory
│   └── 🏀 Basketball/  
│       ├── 📈 Basketball_metrics_table.csv     # Consolidated metrics for all trackers/combinations
│       ├── 📋 Basketball_metrics_table.md      # Markdown formatted table
│       ├── 📊 Basketball_metrics_bar.png       # Bar chart analysis plots
│       ├── 📈 Basketball_eao_trends.png        # EAO trends line plots
│       ├── 🎯 Basketball_precision_vs_robustness.png # Scatter plot analysis
│       ├── 🎥 kcf_gaussian_0.2.avi            # Tracking video outputs
│       ├── 🎥 csrt_gaussian_0.2.avi           # Tracking video outputs
│       └── 🎥 ... (other tracker/combination videos)
├── 📖 README.md                        # This file
└── 📋 requirements.txt                 # All dependencies to run this project
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11.7
- pip (Python package installer)
- **Storage Space**: ~42GB free disk space for generated noisy frames (when processing all OTB-100 sequences)

⚠️ **Storage Warning**: The pipeline generates noisy versions of all frames for each noise type and occlusion level combination. With 6 combinations (2 noise types × 3 occlusion levels) across all OTB-100 sequences, this can require up to 42GB of additional storage space.

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
     ./dataset/
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

## 🎯 Quick Demo

For a quick demonstration of the complete pipeline, use the demo script which processes the shortest sequence (Deer - 71 frames):

```bash
python main_demo.py
```

**Demo Features:**
- ⚡ **Fast Execution**: Completes in ~2 minutes (vs. hours for full dataset)
- 🧹 **Auto-cleanup**: Automatically clears existing results for fresh demonstration
- 📊 **Complete Pipeline**: Demonstrates all features including noise application, tracking, and analysis
- 🎬 **Full Output**: Generates videos, CSV files, and visualization plots
- 📈 **Performance Summary**: Shows best EAO scores for each tracker

**Demo Output Example:**
```
🎯 Sequence: Deer
📦 Trackers evaluated: 2
🔄 Total combinations processed: 12
   🏆 KCF best EAO: 0.5136 (combo: gaussian_0.2)
   🏆 CSRT best EAO: 0.7814 (combo: gaussian_0.2)
⏱️  Total demo time: 121.95 seconds (2.03 minutes)
📊 Processing rate: 0.6 frames/second
```

The demo is perfect for:
- Testing the installation and setup
- Understanding the pipeline workflow
- Quick performance comparison between trackers
- Generating sample results for verification

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
  "SEQUENCES": [
    "Dudek", "Freeman1",
    "Car4", "Car1",
    "Tiger1", "Bird1",
    "Basketball", "Soccer",
    "Box", "Rubik",
    "BlurCar1", "FaceOcc1"
  ],
  "SEQUENCE_CATEGORIES": {
    "Human/Person": ["Dudek", "Freeman1"],
    "Vehicle": ["Car4", "Car1"],
    "Animal": ["Tiger1", "Bird1"],
    "Sports/Athletics": ["Basketball", "Soccer"],
    "Object/Rigid": ["Box", "Rubik"],
    "Quality_Challenges": ["BlurCar1", "FaceOcc1"]
  },
  "TRACKERS": ["KCF", "CSRT"],
  "NOISE_TYPES": ["gaussian", "salt_pepper"],
  "OCCLUSION_LEVELS": [0.2, 0.4, 0.6],
  "VISUALIZE": true,
  "SAVE_VIDEOS": true,
  "FAILURE_THRESHOLD": 0.1,
  "FRAME_DELAY": 25
}
```

### Sequence Configuration

- **`"SEQUENCES": []`** - Process all sequences with valid ground-truth files
- **`"SEQUENCES": ["Basketball"]`** - Process specific sequences
- **`"SEQUENCES": ["Dudek", "Freeman1", ...]`** - Current configuration uses **Enhanced 12-Sequence Set** representing 6 categories for comprehensive evaluation

The current configuration includes a curated **Enhanced 12-Sequence Set** that provides representative coverage across different tracking scenarios:

| Category | Sequences | Description |
|----------|-----------|-------------|
| **Human/Person** | Dudek, Freeman1 | Person tracking scenarios |
| **Vehicle** | Car4, Car1 | Vehicle tracking in different conditions |
| **Animal** | Tiger1, Bird1 | Animal movement and behavior tracking |
| **Sports/Athletics** | Basketball, Soccer | Sports scenarios with fast motion |
| **Object/Rigid** | Box, Rubik | Rigid object tracking |
| **Quality_Challenges** | BlurCar1, FaceOcc1 | Challenging conditions (blur, occlusion) |

⚠️ **Note**: Some OTB-100 sequences lack ground-truth files (e.g., `Human4`, `Skating2`) and will be automatically excluded from processing to prevent errors.

### Other Configuration Options

- **`VISUALIZE`**: Show real-time tracking visualization during evaluation
- **`SAVE_VIDEOS`**: Generate tracking result videos (.avi files)
- **`FAILURE_THRESHOLD`**: IoU threshold below which tracking is considered failed
- **`FRAME_DELAY`**: Delay in milliseconds between frames during visualization

## 📊 Output Examples

The pipeline generates:
- **Consolidated CSV**: `./results/Basketball/Basketball_metrics_table.csv` (all tracker results in one file)
- **Analysis Plots**: 
  - `./results/Basketball/Basketball_metrics_bar.png` (bar charts)
  - `./results/Basketball/Basketball_eao_trends.png` (EAO trends)
  - `./results/Basketball/Basketball_precision_vs_robustness.png` (scatter plot)
- **Video Output**: `./results/Basketball/kcf_gaussian_0.2.avi` (tracking videos)
- **Markdown Table**: `./results/Basketball/Basketball_metrics_table.md` (formatted results)

## 📄 License

This project is licensed under the **MIT License**.

## 🙏 Acknowledgments

- **OTB-100 dataset**: http://cvlab.hanyang.ac.kr/tracker_benchmark
- **OpenCV** for tracking algorithms