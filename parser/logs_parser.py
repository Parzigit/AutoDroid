import os
import pandas as pd
import google.generativeai as genai
import ast # ADDED ast for safe parsing

# CONFIGURATION
# Replace with your actual paths and API key
BASE_DIR = r"path to base dir"
OUTPUT_PATH = r"path to output dir"
API_KEY = "GEMINI_API_KEY"  # Replace with your actual Gemini API key

print("[INFO] Loaded configuration")

# Gemini API Setup
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
print("[INFO]Gemini model loaded")

# Prompt
GEMINI_PROMPT = """
You are given a static analysis log from a tool like DUA-Forensics or DroidFax, which analyzes Android apps for security-relevant features.

Your task is to:
1. Identify and extract all **security-relevant features or behaviors**, such as:
   - Callback methods (e.g., onCreate, onReceive, onDestroy)
   - ICC links
   - Activities, views, dialogs (from app or library)
2. For each identified feature, count how many times it appears.
3. Return ONLY a Python dictionary in this format:

{
  "Callback_onCreate": 4,
  "Callback_onDestroy": 3,
  "App_Activities": 6,
  "Lib_Views": 10
}

Ignore warnings or unrelated logs. Format response strictly as valid Python dictionary.
"""

# Parse a log with Gemini
def parse_security_log_with_gemini(log_path):
    print(f"[INFO] 🔍 Reading log: {log_path}")
    if not os.path.exists(log_path):
        print(f"[WARN] Log file not found: {log_path}")
        return {}

    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
    except Exception as e:
        print(f"[ERROR] Failed to read file: {log_path} | {e}")
        return {}

    try:
        print("[INFO] 🚀 Sending content to Gemini...")
        response = model.generate_content([GEMINI_PROMPT, content])
        response_text = response.text.strip()
        
        # IMPROVED: Safely parse the response, handling the ```python block
        if response_text.startswith("```python"):
            response_text = response_text.replace("```python", "").replace("```", "").strip()

        if response_text.startswith("{") and response_text.endswith("}"):
            print("[INFO] ✅ Gemini returned a valid dictionary string")
            return ast.literal_eval(response_text) # Using ast.literal_eval for safety
        else:
            print(f"[WARN] Gemini returned unexpected format:\n{response_text}")
            return {}
    except Exception as e:
        print(f"[ERROR] Gemini API failed or failed to parse response: {e}")
        return {}

# Main Processing
api_categories = set()
app_rows = []
print(f"[INFO] 🔁 Traversing base directory: {BASE_DIR}")

# The provided console log has a specific directory structure. We'll adjust the traversal to match.
# The structure appears to be: BASE_DIR/timeout/repetition/tool/app_result_dir/...
for timeout in os.listdir(BASE_DIR):
    timeout_dir = os.path.join(BASE_DIR, timeout)
    if not os.path.isdir(timeout_dir):
        continue
    print(f"[INFO] Entering timeout: {timeout}")

    for repetition in os.listdir(timeout_dir):
        rep_dir = os.path.join(timeout_dir, repetition)
        if not os.path.isdir(rep_dir):
            continue
        print(f"[INFO] Entering repetition: {repetition}")

        for tool in os.listdir(rep_dir):
            tool_dir = os.path.join(rep_dir, tool)
            if not os.path.isdir(tool_dir):
                continue
            print(f"[INFO] 🛠️ Processing tool: {tool}")

            apps = {}
            for app_result_dir in os.listdir(tool_dir):
                app_path = os.path.join(tool_dir, app_result_dir)
                if not os.path.isdir(app_path):
                    continue

                if app_result_dir.startswith('benign'):
                    key = app_result_dir[6:]
                    apps.setdefault(key, [None, None])[0] = app_result_dir
                elif app_result_dir.startswith('malicious'):
                    key = app_result_dir[9:]
                    apps.setdefault(key, [None, None])[1] = app_result_dir

            for app_name, (benign_dir, malicious_dir) in apps.items():
                print(f"[INFO] Processing app pair: {app_name}")

                # Benign
                if benign_dir:
                    print(f"[INFO] 🔹 Benign app: {benign_dir}")
                    log_path = os.path.join(tool_dir, benign_dir, 'security_report', 'security_report.log')
                    api_calls = parse_security_log_with_gemini(log_path)
                    api_categories.update(api_calls.keys())
                    row = {'App': benign_dir, 'Label': 0}
                    row.update(api_calls)
                    app_rows.append(row)

                # Malicious
                if malicious_dir:
                    print(f"[INFO] 🔸 Malicious app: {malicious_dir}")
                    log_path = os.path.join(tool_dir, malicious_dir, 'security_report', 'security_report.log')
                    api_calls = parse_security_log_with_gemini(log_path)
                    api_categories.update(api_calls.keys())
                    row = {'App': malicious_dir, 'Label': 1}
                    row.update(api_calls)
                    app_rows.append(row)

print(f"[INFO] Finished parsing all logs. Total apps processed: {len(app_rows)}")

# DataFrame Creation
api_categories = sorted(api_categories)
df = pd.DataFrame(app_rows)

print("[INFO] 🧱 Ensuring all columns exist...")
for cat in api_categories:
    if cat not in df.columns:
        df[cat] = 0
df.fillna(0, inplace=True)

# Save to CSV
print(f"[INFO] 💾 Saving CSV to: {OUTPUT_PATH}")
df.to_csv(OUTPUT_PATH, index=False)
print(f"[SUCCESS] 🎉 Done! CSV saved at: {OUTPUT_PATH}")
