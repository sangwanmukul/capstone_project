import os
import pandas as pd

class ExperimentLogger:
    def __init__(self, path="results"):
        self.path = path
        os.makedirs(path, exist_ok=True)

    def save(self, df, name="experiment.csv"):
        file_path = os.path.join(self.path, name)
        df.to_csv(file_path, index=False)
        print(f"[INFO] Saved → {file_path}")