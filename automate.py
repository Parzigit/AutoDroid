import os
import glob
import shutil
import warnings
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

from textual.app import App, ComposeResult
from textual.widgets import Static, Header, Footer, Button
from textual.containers import Vertical, ScrollableContainer


from typing import List, Optional, Union  
# --- keep console noise low
warnings.filterwarnings("ignore")

# === PATH CONFIG ===
APK_INPUT_DIR = "/source path/benchmark/data/input"
RESULTS_DIR   = "/source path/benchmark/results"
DROIDXP_PATH  = "/root path/benchmark"

# === DROIDXP CLI ===
# Update according yo requirements
MAIN_COMMAND = [
    "python3", "main.py",
    "-tools", "droidbot",
    "-t", "60",
    "-r", "1",
    "--disable-static",
]

# === MODEL CONFIG ===
MODEL_PATH = 'path to your saved model' 
SCALER_PATH = 'path to your saved model'

FEATURE_NAMES = [
    "VIDEO", "BLUETOOTH_INFORMATION", "CALENDAR_INFORMATION", "SMS_MMS", "ACCOUNT_INFORMATION",
    "EMAIL_INFORMATION", "FILE_INFORMATION", "SYNCHRONIZATION_DATA", "PHONE_CONNECTION", "NETWORK",
    "AUDIO", "IMAGE", "ACCOUNT_SETTINGS", "VOIP", "FILE", "DATABASE_INFORMATION",
    "NETWORK_INFORMATION", "HARDWARE_INFO", "NFC", "SYSTEM_SETTINGS", "CONTACT_INFORMATION",
    "VOIP_INFORMATION", "PHONE_INFORMATION", "EMAIL_SETTINGS", "PHONE_STATE", "BROWSER_INFORMATION",
    "NO_CATEGORY", "LOG", "BLUETOOTH", "UNIQUE_IDENTIFIER", "INTER_APP_COMMUNICATION",
    "EMAIL", "ALL", "LOCATION_INFORMATION"
]

def parse_inconsistent_src_txt(src_txt_path: str, feature_names: List[str]) -> Optional[pd.DataFrame]:
    """Parse src.txt -> single-row dataframe of sums per feature."""
    feature_dict = {feat: 0.0 for feat in feature_names}
    try:
        with open(src_txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                parts = raw.strip().split()
                if len(parts) < 2:
                    continue
                feat = parts[0]
                if feat not in feature_dict:
                    continue
                try:
                    vals = [float(x) for x in parts[1:] if x.strip() != ""]
                    feature_dict[feat] = float(np.sum(vals)) if vals else 0.0
                except Exception:
                    feature_dict[feat] = 0.0
        return pd.DataFrame([feature_dict]).fillna(0)
    except Exception:
        return None


class BenchmarkApp(App):
    """Textual TUI: separate Benchmark and Predict flows."""
    CSS = """
    .logbox {
        height: 20;
        border: solid $primary;
        scrollbar-gutter: stable;
    }
    #log_content { padding: 1; }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Button("Select APK File", id="select_apk", variant="primary"),
            Static("No APK selected", id="apk_label"),
            Button("Run Benchmark", id="run_button", disabled=True, variant="success"),
            Button("Predict from Result", id="predict_button", disabled=False, variant="warning"),
            ScrollableContainer(
                Static("Ready.\n", id="log_content"),
                classes="logbox",
                id="log_container",
            ),
        )
        yield Footer()

    def on_mount(self) -> None:
        self.apk_path: Optional[str] = None
        self.apk_name: Optional[str] = None
        self.execution_id: Optional[str] = None
        self.last_result_src_path: Optional[str] = None
        self.model: Optional[Union[xgb.XGBClassifier, xgb.Booster]] = None  # Flexible for XGBClassifier or Booster
        self.scaler: Optional[Union[object, None]] = None  # Adjust based on scaler type
        self._try_load_model_artifacts()

    def _now(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")

    def log_message(self, msg: str):
        try:
            log_content = self.query_one("#log_content", Static)
            current_text = str(log_content.renderable) if log_content.renderable else ""
            lines = current_text.splitlines()
            lines.append(f"[{self._now()}] {msg}")
            if len(lines) > 150:
                lines = ["📜 ... (earlier logs truncated)"] + lines[-150:]
            log_content.update("\n".join(lines))
            self.refresh()
            self.call_later(self._scroll_log_to_bottom)
        except Exception:
            print(f"[LOG] {msg}")

    def _scroll_log_to_bottom(self):
        try:
            container = self.query_one("#log_container")
            container.scroll_end(animate=False)
        except Exception:
            pass

    def _safe_log(self, msg: str):
        try:
            self.call_from_thread(self.log_message, msg)
        except Exception:
            print(msg)

    def _try_load_model_artifacts(self):
        try:
            # Load pickled XGBoost model (e.g., XGBClassifier) from .pkl
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            self.log_message("XGBoost model and scaler loaded from .pkl files.")
        except Exception as e:
            self.model = None
            self.scaler = None
            self.log_message(f"Could not load model/scaler: {e}")

    def on_button_pressed(self, event: Button.Pressed):
        bid = event.button.id
        if bid == "select_apk":
            self._select_apk()
        elif bid == "run_button":
            self._start_benchmark_thread()
        elif bid == "predict_button":
            self._start_predict_thread()

    def _select_apk(self):
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            fpath = filedialog.askopenfilename(
                title="Select APK",
                filetypes=[("APK files", "*.apk"), ("All files", "*.*")]
            )
            root.destroy()
        except Exception:
            fpath = None

        if not fpath:
            self.log_message("📝 No file selected.")
            return

        try:
            self.apk_path = fpath
            self.apk_name = os.path.basename(fpath)
            os.makedirs(APK_INPUT_DIR, exist_ok=True)
            for ap in glob.glob(os.path.join(APK_INPUT_DIR, "*.apk")):
                try:
                    os.remove(ap)
                except Exception:
                    pass
            shutil.copy2(self.apk_path, os.path.join(APK_INPUT_DIR, self.apk_name))
            self.query_one("#apk_label", Static).update(f"📱 Selected: {self.apk_name}")
            self.query_one("#run_button", Button).disabled = False
            self.log_message(f"APK ready for benchmark: {self.apk_name}")
        except Exception as e:
            self.log_message(f"Failed preparing APK: {e}")

    def _start_benchmark_thread(self):
        threading.Thread(target=self.run_benchmark, daemon=True).start()

    def run_benchmark(self):
        if not self.apk_path:
            self._safe_log("ℹ️ Please select an APK first.")
            return

        self.execution_id = datetime.now().strftime("%Y%m%d%H%M%S")
        self._safe_log(f"🚀 Benchmark started. Execution ID: {self.execution_id}")

        try:
            trace_dir = os.path.join(DROIDXP_PATH, "trace")
            if os.path.exists(trace_dir):
                shutil.rmtree(trace_dir)
                self._safe_log("Cleaned previous trace.")
        except Exception as e:
            self._safe_log(f"Trace cleanup issue: {e}")

        try:
            env = os.environ.copy()
            env["EXECUTION_ID"] = self.execution_id
            process = subprocess.Popen(
                MAIN_COMMAND,
                cwd=DROIDXP_PATH,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
            )
            for i, line in enumerate(process.stdout, start=1):
                ln = line.strip()
                if ln and i % 10 == 0:
                    self._safe_log(ln)
            process.wait()
            self._safe_log("Benchmark finished.")
        except Exception as e:
            self._safe_log(f"Benchmark run error: {e}")

        expected_src = os.path.join(
            RESULTS_DIR, self.execution_id, "60", "1", "droidbot",
            (self.apk_name or "app.apk"),
            "security_report", "src.txt",
        )
        if os.path.exists(expected_src):
            self.last_result_src_path = expected_src
            self._safe_log(f"Found src.txt: {expected_src}")
        else:
            self.last_result_src_path = None
            self._safe_log("Could not find src.txt. Use 'Predict from Result' to browse.")

    def _start_predict_thread(self):
        threading.Thread(target=self.run_prediction, daemon=True).start()

    def _guess_src_path(self) -> Optional[str]:
        if self.last_result_src_path and os.path.exists(self.last_result_src_path):
            return self.last_result_src_path
        if self.execution_id and self.apk_name:
            guess = os.path.join(
                RESULTS_DIR, self.execution_id, "60", "1", "droidbot",
                self.apk_name, "security_report", "src.txt"
            )
            if os.path.exists(guess):
                return guess
        try:
            pattern = os.path.join(RESULTS_DIR, "**", "droidbot", self.apk_name or "*", "**", "src.txt")
            candidates = glob.glob(pattern, recursive=True)
            if candidates:
                candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                return candidates[0]
        except Exception:
            pass
        return None

    def run_prediction(self):
        if (self.model is None) or (self.scaler is None):
            self._safe_log("Model/scaler not loaded. Reloading...")
            self._try_load_model_artifacts()
            if (self.model is None) or (self.scaler is None):
                self._safe_log("Could not load model/scaler. Prediction skipped.")
                return

        src_path = self._guess_src_path()
        if not src_path or not os.path.exists(src_path):
            self._safe_log("Please select the generated src.txt")
            try:
                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                picked = filedialog.askopenfilename(
                    title="Select generated src.txt",
                    filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
                )
                root.destroy()
            except Exception:
                picked = None
            if picked and os.path.exists(picked):
                src_path = picked
                self.last_result_src_path = picked
                self._safe_log(f" Using src.txt: {picked}")
            else:
                self._safe_log(" No src.txt selected. Prediction aborted.")
                return

        feats = parse_inconsistent_src_txt(src_path, FEATURE_NAMES)
        if feats is None or feats.empty:
            self._safe_log("Could not parse features from src.txt")
            return

        self._safe_log("Parsed Features:\n" + feats.to_string(index=False))

        if feats.shape[1] != len(FEATURE_NAMES):
            self._safe_log(f"Error: Expected {len(FEATURE_NAMES)} features, got {feats.shape[1]}.")
            return

        try:
            fv_scaled = self.scaler.transform(feats)
            
            # If model is XGBoost, optionally use DMatrix for prediction (improves compatibility)
            if isinstance(self.model, xgb.XGBClassifier):
                dmatrix = xgb.DMatrix(fv_scaled, feature_names=FEATURE_NAMES)
                proba = self.model.predict_proba(dmatrix)[0][1]  # Probability of class 1 (malware)
                pred = self.model.predict(dmatrix)[0]
            else:
                proba = self.model.predict_proba(fv_scaled)[0][1]
                pred = self.model.predict(fv_scaled)[0]
            
            # Optional: Use a custom threshold (e.g., from your training tuning)
            threshold = 0.5  # Adjust to your best_thr if known (can be checked from running training.py)
            pred = 1 if proba >= threshold else 0
            
            label = "Malicious" if pred == 1 else "Benign"
            self._safe_log(f"Prediction: {label} (Malicious probability: {proba:.4f})")
        except Exception as e:
            self._safe_log(f"Scaling or prediction error: {e}")


if __name__ == "__main__":
    BenchmarkApp().run()
