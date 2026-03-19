"""
Saves precomputed predictions for the Streamlit Cloud demo.
Runs locally, outputs a small JSON file that the deployed app uses.
"""
import os, json, joblib, numpy as np, pandas as pd

BASE     = r"C:\Users\aryab\Coding\mri_quality_pattern_parkinson"
OUT_PATH = os.path.join(BASE, "demo_data.json")

# Load pipeline results
results_path = os.path.join(BASE, "outputs", "pipeline_results",
                            "pipeline_results.json")
with open(results_path) as f:
    results = json.load(f)

# Build clean demo dataset — only keep what the app needs
demo = []
for r in results:
    qa   = r.get("quality", {})
    pred = r.get("prediction")
    demo.append({
        "subject":     r["subject"],
        "true_label":  r.get("true_label", -1),
        "dataset":     "neurocon" if "032014" <= r["subject"][-6:] <= "032056"
                       else "taowu",
        "qa": {
            "grade":         qa.get("grade", "?"),
            "quality_score": qa.get("quality_score", 0),
            "snr":           qa.get("snr", 0),
            "cnr":           qa.get("cnr", 0),
            "sharpness":     qa.get("sharpness", qa.get("Sharpness", 0)),
            "fov_coverage":  qa.get("fov_coverage",
                                    qa.get("FOV Coverage", 0)),
            "uniformity":    qa.get("uniformity",
                                    qa.get("Uniformity", 0)),
            "ghosting":      qa.get("ghosting", qa.get("Ghosting", 0)),
        },
        "prediction": {
            "label":          pred["label"] if pred else None,
            "prediction":     pred["prediction"] if pred else None,
            "pd_probability": pred["pd_probability"] if pred else None,
            "hc_probability": pred["hc_probability"] if pred else None,
            "confidence":     pred["confidence"] if pred else None,
        } if pred else None
    })

with open(OUT_PATH, "w") as f:
    json.dump(demo, f, indent=2)

print(f"Saved {len(demo)} subjects to demo_data.json")
print(f"File size: {os.path.getsize(OUT_PATH)/1024:.1f} KB")