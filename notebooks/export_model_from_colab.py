# Run this near the end of the Colab notebook after final_model, FINAL_THRESHOLD,
# FEATURE_COLS, ORIGIN_DELAY_MAP, DEST_DELAY_MAP, ORIGIN_FALLBACK, and DEST_FALLBACK exist.

import joblib
from google.colab import files

bundle = {
    "model": final_model,
    "threshold": FINAL_THRESHOLD,
    "feature_cols": FEATURE_COLS,
    "origin_delay_map": ORIGIN_DELAY_MAP,
    "dest_delay_map": DEST_DELAY_MAP,
    "origin_fallback": ORIGIN_FALLBACK,
    "dest_fallback": DEST_FALLBACK,
}

joblib.dump(bundle, "flight_delay_bundle.joblib")
files.download("flight_delay_bundle.joblib")
print("Downloaded flight_delay_bundle.joblib")
