"""Print test metrics from best_model_*/test_metrics.json. Run from source_code_v2/."""

import json
import glob

BASE = "."

# Scan checkpoints (run from source_code_v2/)
for model_dir in sorted(glob.glob(f"{BASE}/best_model_*")):
    for metrics_file in glob.glob(f"{model_dir}/*metrics.json"):
        if metrics_file.endswith("/metrics.json"):
            continue

        with open(metrics_file) as f:
            m = json.load(f)

        report = m["classification_report"]

        print("=" * 80)
        print("experiment:", m["experiment"])
        print("test_dataset:", m["test_dataset"])
        print("cross_domain:", m["cross_domain"])
        print("sarcasm_f1:", report["1"]["f1-score"])
        print("macro_f1:", report["macro avg"]["f1-score"])
        print("accuracy:", report["accuracy"])
