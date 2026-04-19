import numpy as np

def summarize(df):
    return {
        "mean_ARI": df["ARI"].mean(),
        "bot_success": df["bot_success"].mean(),
        "human_score": df["human_score"].mean(),
        "stability": np.std(df["ARI"])
    }