# Dynamic Analysis of Android Malware Detection: 

### An overview and setup using DroidXP Benchmark Framework with Machine Learning Integration


This repository provides an end-to-end pipeline for extracting features from Android malware/benign app benchmarks and building machine learning models to classify apps.  
It is inspired by the [droidxp/benchmark](https://github.com/droidxp/benchmark) framework for automated Android security evaluation.

Below is a walkthrough of:  
- How to set up DroidXP and collect benchmark data  
- How to extract ML features from results  
- How to train, evaluate ML models and automate the testing using this repository

---

## 1. **Install and Configure DroidXP**

To generate analysis results, you must first set up DroidXP and collect benign/malicious app execution traces.

**Steps:**

1. **Clone the DroidXP repository:**
    ```shell
    git clone https://github.com/droidxp/benchmark
    ```

2. **Clone the configuration repository:**
    ```shell
    git clone https://github.com/droidxp/benchmark-vm
    ```

3. **Create a folder called `script` in your home directory:**
    ```shell
    mkdir ~/script
    ```

4. **Copy the 3 files from `benchmark-vm/ubuntu_20.04_python3_x86/` to `~/script`:**
    ```shell
    cp benchmark-vm/ubuntu_20.04_python3_x86/* ~/script/
    ```

5. **Run `./config.sh` in `~/script` to install and configure DroidXP with all required tools:**  
   This includes dependencies like DroidBot, Monkey, Java 7, Android SDK 23+, Python 2/3, Pixel 2 emulator, and ensures commands (`adb`, `java`, `jarsigner`, `emulator`, `aapt`) are in your `PATH`.
    ```shell
    cd ~/script
    ./config.sh
    ```

---

## 2. **Download Pair Apps (Malicious/Benign) for Analysis (Main Dataset)**

1. **Use apps from `LargeI.csv` in the DroidXP repo's `data/input/` folder:**  
   (Exclude app-25, app-32, app-36, app-41, app-88, app-93 for 96 pairs; or use `SmallE.csv` for a smaller experiment.)

2. **Get an AndroZoo access key** and insert it into `getApps.py` (in `data/input/`):  
   Replace the placeholder in `key="Insert here your key"`.

3. **Edit `getApps.py` to select the CSV:**  
   For example, set: `with open('LargeI.csv') as csvfile:`

4. **Run the script to download and format apps as malicious/benign:**  
   Store them in `benchmark/data/input/`.
    ```shell
    python getApps.py
    ```

---

## 3. **Run the Benchmark**

1. **Main experiment** (disable static analysis, 120s timeout, 3 repetitions, with tools Monkey, DroidBot, DroidMate, Humanoid):
    ```shell
   e.g., python3 main.py -tools monkey droidbot droidmate humanoid -t time -r repetitions --disable-static
    ```

- **Results appear in**:  
  `benchmark/results/<timestamp>/report/`  
  (e.g., `benchmark/results/20210220212608/report/`)

---

# **To Replicate the Project**

This proposal builds on the benchmark outputs (e.g., logs from DroidXP) to extract features, train models, and predict malware. 

**Sequential steps:**  
Generate logs via benchmarking (do this before feature extraction), extract features next (requires benchmark outputs), train models after that (uses extracted features), and finally predict or automate.  
**Customize paths** (e.g., BASE_DIR, model paths, API keys) in all scripts before running.

### **A. Set Up Environment (Do This First):**

- Use Python 3.6+ (Specifically 3.6.6 used in the notebook).
- Create a virtual environment:
    ```shell
    python -m venv autodroid_env
    source autodroid_env/bin/activate  # or autodroid_env\Scripts\activate on Windows
    ```
- Install libraries:
    ```shell
    pip install pandas scikit-learn imbalanced-learn xgboost lightgbm joblib google-generativeai textual tkinter numpy ast
    ```
- For AI parsing in `logs_parser.py`, replace `"GEMINI_API_KEY"` using Gemini API key.
- Jupyter/Kaggle for the notebook:
- Gather benign/malicious APK's security_logs or src.txt files (assuming they are either replicated or referenced from here). [Recommended - Kaggle for training]

### **B. Generate Logs via Benchmarking (Before Feature Extraction):**

-To produce structured logs (`src.txt` and `security_report.log`) in folders like:
  ```
  RESULTS_DIR/execution/timeout/repetition/tool/app/security_report/
  ```

### **C. Extract Features (After Generating Logs):**

- **To extract API features from `src.txt`:**  
    Use suitable inp/output dirs (`BASE_DIR`, `RESULTS_DIR`, `EXECUTIONS`) in `src_parse.py` and run:
    ```bash
    python3 src_parse.py
    ```
    Outputs `benchmark_ml_features.csv`.

- **For detailed log features from `security_report.log`:**  
    Edit paths, `OUTPUT_PATH`, and API key in `logs_parser.py` and run:
    ```bash
    python3 logs_parser.py
    ```
    Outputs a CSV (convert to Excel if needed, like `securitylog_features.xlsx`).

### **D. Train the Model (After Feature Extraction):**

- **For small API-based model (XGBoost):**  
    Edit dataset path in `training_small.py` (e.g., to `benchmark_ml_features.csv` or `src_features.csv`) and run:
    ```bash
    python3 training_small.py
    ```
    Outputs (e.g., `detection_MODEL.pkl` (model) and `scaler_MODEL.pkl`.)

- **For larger features:**  
    Edit path in `train_big.py` (e.g., to `securitylog_features.xlsx`) and run:
    ```bash
    python3 train_big.py
    ```
- For reference visit `android-malware-analysis-4.ipynb` (includes version checks and use cases).

### **E. Predict or Automate:**

- **Basic prediction:**  
    Edit paths in `model.py` (to trained model/scaler and a `src.txt` file) and run:
    ```shell
    python model.py
    ```
    Outputs benign/malicious label and probability.

- **Full automation:**  
    Edit paths in `automate.py` (`APK_INPUT_DIR`, `RESULTS_DIR`, `DROIDXP_PATH`, model/scaler) and run:
    ```shell
    python automate.py
    ```
    Use the TUI to select APK, run benchmark (via subprocess to DroidXP), parse, and predict.

### **Troubleshooting and Testing:**

- Test with samples like `src_features.csv` or `securitylog_features.xlsx`.
- Check metrics (e.g., F1-score ~0.74-0.82) in console outputs.
- Common issues: Path mismatches or API errors—verify logs.

---



Here's a breakdown of each file's purpose, key functions, and how they work (based on the project files):

- **src_parse.py:**  
  Parses `src.txt` files (API category counts) from benchmark dirs.  
  Function `parse_src_txt` reads tab-separated data, sums counts (e.g., "VIDEO": sum of values).  
  It traverses dirs to collect benign/malicious pairs, outputs CSV with "App", "Label" (0=benign, 1=malicious), and features.  
  Dependencies: Pandas, os. Run after benchmarking for small datasets.

- **logs_parser.py:**  
  Parses `security_report.log` using Gemini AI.  
  Function `parse_security_log_with_gemini` sends log to AI with a prompt, parses response as dict (e.g., `{"Callback_onCreate": 4}`) via `ast.literal_eval`.  
  Traverses dirs like above, outputs CSV with dynamic features.  
  Dependencies: Pandas, os, google-generativeai, ast. Requires API key; handles errors in AI responses.

- **training_small.py:**  
  Trains XGBoost on small API datasets.  
  Loads CSV, splits train/test (stratified 75/25), scales with StandardScaler, applies SMOTE for imbalance, grid searches params (e.g., max_depth, learning_rate).  
  Evaluates with classification report, confusion matrix, tunes F1 threshold. Saves model/scaler via joblib.  
  Dependencies: Pandas, Scikit-learn, imblearn, XGBoost, joblib.

- **train_big.py:**  
  Similar to above but for large log datasets with LightGBM (more efficient).  
  Loads Excel/CSV, same preprocessing/grid search/evaluation.  
  No default save—uncomment joblib lines.  
  Dependencies: Same as above plus LightGBM.

- **src_features.csv:**  
  Sample output from src_parse.py. Columns: "App", "Label", API features (e.g., "NETWORK" sums). Use for quick training tests.

- **securitylog_features.xlsx:**  
  Sample from logs_parser.py. Columns: "App", "Label", detailed features (e.g., callback counts). For large-scale training.

- **model.py:**  
  Predicts on single src.txt. Function `parse_inconsistent_src_txt` sums features flexibly.  
  Loads model/scaler, scales input, predicts label/probability (e.g., >0.5 = malicious).  
  Dependencies: Pandas, joblib, numpy. Interactive via input prompt.

- **automate.py:**  
  TUI for end-to-end workflow. Uses Textual for interface (buttons for APK select, benchmark, predict).  
  Runs subprocess for benchmarking (customize MAIN_COMMAND to DroidXP), parses like src_parse.py, loads model for prediction.  
  Logs in console via threading.  
  Dependencies: os, glob, shutil, subprocess, threading, tkinter, datetime, pandas, numpy, joblib, xgboost, textual.

- **android-malware-analysis-4.ipynb:**  
  Notebook version of training. Cells: Load data, train XGBoost/LightGBM with grid search, evaluate/save.  
  For testing on Kaggle (e.g., with GPU). Dependencies: Same as scripts plus Jupyter.

---

## 6. **References**

- [droidxp/benchmark](https://github.com/droidxp/benchmark)
- [droidxp/benchmark-vm](https://github.com/droidxp/benchmark-vm)
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)

---

**Questions or contributions?**  
Open an issue or pull request!
