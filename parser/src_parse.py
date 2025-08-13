# This file is to replicate the dataset used for training & features extracted for the model 
import os
import pandas as pd
 # MUST BE RUN AFTER RUNNING THE BENCHMARK OF ALL THE FILES
# CHANGE THESE PATHS TO MATCH YOUR SETUP
BASE_DIR = 'select your base dir where the execution folder is present'
RESULTS_DIR = os.path.join(BASE_DIR, 'Downloads')
EXECUTIONS = ['20250727092834']  # select the 

def parse_src_txt(src_path):
    """Parse src.txt and return a dict {api_category: total_count}"""
    if not os.path.exists(src_path):
        return {}
    # Reads tab-separated src.txt
    df = pd.read_csv(src_path, sep="\t", header=None)
    # First column: API category, rest: counts
    # Sum all columns except first as the total count
    return {row[0]: sum([int(x) for x in row[1:] if str(x).isdigit()]) for _, row in df.iterrows()}

api_categories = set()
app_rows = []

for execution in EXECUTIONS:
    execution_dir = os.path.join(RESULTS_DIR, execution)
    for timeout in os.listdir(execution_dir):
        timeout_dir = os.path.join(execution_dir, timeout)
        if not os.path.isdir(timeout_dir):
            continue
        for repetition in os.listdir(timeout_dir):
            rep_dir = os.path.join(timeout_dir, repetition)
            if not os.path.isdir(rep_dir):
                continue
            for tool in os.listdir(rep_dir):
                tool_dir = os.path.join(rep_dir, tool)
                if not os.path.isdir(tool_dir):
                    continue
                # Find all benign/malicious pairs
                apps = {}
                for app_result_dir in os.listdir(tool_dir):
                    app_path = os.path.join(tool_dir, app_result_dir)
                    if not os.path.isdir(app_path):
                        continue
                    if app_result_dir.startswith('benign'):
                        key = app_result_dir[6:]  # Remove 'benign' prefix
                        apps.setdefault(key, [None, None])[0] = app_result_dir
                    elif app_result_dir.startswith('malicious'):
                        key = app_result_dir[9:]  # Remove 'malicious' prefix
                        apps.setdefault(key, [None, None])[1] = app_result_dir
                for app_name, (benign_dir, malicious_dir) in apps.items():
                    # Benign
                    if benign_dir:
                        src_path = os.path.join(tool_dir, benign_dir, 'security_report', 'src.txt')
                        api_calls = parse_src_txt(src_path)
                        api_categories.update(api_calls.keys())
                        row = {'App': benign_dir, 'Label': 0}
                        for cat, count in api_calls.items():
                            row[cat] = count
                        app_rows.append(row)
                    # Malicious
                    if malicious_dir:
                        src_path = os.path.join(tool_dir, malicious_dir, 'security_report', 'src.txt')
                        api_calls = parse_src_txt(src_path)
                        api_categories.update(api_calls.keys())
                        row = {'App': malicious_dir, 'Label': 1}
                        for cat, count in api_calls.items():
                            row[cat] = count
                        app_rows.append(row)

# Create DataFrame with all possible API categories as columns
api_categories = sorted(api_categories)
df = pd.DataFrame(app_rows)
for cat in api_categories:
    if cat not in df.columns:
        df[cat] = 0
df.fillna(0, inplace=True)
df.to_csv('benchmark_ml_features.csv', index=False)
print("Extracted data saved to src_features.csv")
