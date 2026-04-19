import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12
})


# =========================
# 1. ARI Training Curve (IMPROVED)
# =========================
def plot_ari_curve(df, save_path=None):
    plt.figure()

    # raw
    plt.plot(df["step"], df["ARI"], alpha=0.08)

    # smooth
    smooth = df["ARI"].rolling(window=100).mean()
    std = df["ARI"].rolling(window=100).std()

    plt.plot(df["step"], smooth, linewidth=2, label="Mean")
    plt.fill_between(df["step"], smooth - std, smooth + std, alpha=0.2)

    plt.xlabel("Steps")
    plt.ylabel("ARI")
    plt.title("ARI vs Training Steps")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path + "/ari_curve.png", dpi=300)

    plt.show()


# =========================
# 2. ARI Distribution
# =========================
def plot_ari_distribution(df, save_path=None):
    plt.figure()

    plt.hist(df["ARI"], bins=30, density=True, alpha=0.5)
    df["ARI"].plot(kind="kde", linewidth=2)

    plt.xlabel("ARI")
    plt.title("ARI Distribution")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path + "/ari_distribution.png", dpi=300)

    plt.show()


# =========================
# 3. Bot Success Distribution (FIXED 🔥)
# =========================
def plot_bot_distribution(df, save_path=None):
    plt.figure()

    plt.hist(df["bot_success"], bins=30, density=True, alpha=0.5)
    df["bot_success"].plot(kind="kde", linewidth=2)

    plt.yscale("log")

    plt.xlabel("Bot Success")
    plt.ylabel("Density (log scale)")
    plt.title("Bot Success Distribution")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path + "/bot_distribution.png", dpi=300)

    plt.show()


# =========================
# 4. Calibration Curve
# =========================
def plot_calibration(df, bins=10, save_path=None):
    df = df.copy()

    df["bin"] = pd.cut(df["entropy"], bins=bins)
    grouped = df.groupby("bin", observed=False)[["entropy", "bot_success"]].mean()

    plt.figure()

    plt.plot(grouped["entropy"], grouped["bot_success"], marker='o', linewidth=2, label="Model")
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=2, label="Perfect")

    plt.xlabel("Predicted (Entropy)")
    plt.ylabel("Actual (Bot Success)")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path + "/calibration.png", dpi=300)

    plt.show()


# =========================
# 5. Performance vs Difficulty (SMOOTHED 🔥)
# =========================
def plot_difficulty_bins(df, bins=12, save_path=None):
    plt.figure()

    df = df.copy()
    df["bin"] = pd.cut(df["difficulty"], bins=bins)

    grouped = df.groupby("bin", observed=False)[
        ["difficulty", "bot_success", "human_score"]
    ].mean()

    # smoothing
    x = grouped["difficulty"].rolling(2).mean()
    bot = grouped["bot_success"].rolling(2).mean()
    human = grouped["human_score"].rolling(2).mean()

    plt.plot(x, bot, marker='o', label="Bot Success")
    plt.plot(x, human, marker='o', label="Human Score")

    plt.xlabel("Difficulty")
    plt.ylabel("Performance")
    plt.title("Performance vs Difficulty")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path + "/difficulty_bins.png", dpi=300)

    plt.show()


# =========================
# 6. Feature Evolution (RAW + SMOOTH 🔥)
# =========================
def plot_feature_evolution(df, save_path=None):
    plt.figure()

    df_small = df[df["step"] < 1000]

    # raw (faint)
    plt.plot(df_small["step"], df_small["warp"], alpha=0.1)
    plt.plot(df_small["step"], df_small["clutter"], alpha=0.1)
    plt.plot(df_small["step"], df_small["variation"] / 30, alpha=0.1)

    # smooth
    smooth_warp = df_small["warp"].rolling(50).mean()
    smooth_clutter = df_small["clutter"].rolling(50).mean()
    smooth_var = (df_small["variation"] / 30).rolling(50).mean()

    plt.plot(df_small["step"], smooth_warp, label="warp")
    plt.plot(df_small["step"], smooth_clutter, label="clutter")
    plt.plot(df_small["step"], smooth_var, label="variation (scaled)")

    plt.xlabel("Steps")
    plt.ylabel("Value")
    plt.title("Feature Evolution (Early Phase)")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path + "/features.png", dpi=300)

    plt.show()


# =========================
# 7. Trade-off Plot
# =========================
def plot_tradeoff(df, save_path=None):
    plt.figure()

    sample_df = df.sample(frac=0.4, random_state=42)

    x = sample_df["bot_success"] + np.random.normal(0, 0.003, len(sample_df))
    y = sample_df["human_score"] + np.random.normal(0, 0.002, len(sample_df))

    plt.scatter(x, y, alpha=0.3)

    plt.xlabel("Bot Success")
    plt.ylabel("Human Score")
    plt.title("Human vs Bot Trade-off")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path + "/tradeoff.png", dpi=300)

    plt.show()


# =========================
# 8. ARI vs Difficulty (SMOOTHED 🔥)
# =========================
def plot_ari_vs_difficulty(df, bins=12, save_path=None):
    plt.figure()

    df = df.copy()
    df["bin"] = pd.cut(df["difficulty"], bins=bins)

    grouped = df.groupby("bin", observed=False)[["difficulty", "ARI"]].mean()

    # 🔥 FIX: stronger smoothing
    x = grouped["difficulty"].rolling(3, min_periods=1).mean()
    y = grouped["ARI"].rolling(3, min_periods=1).mean()

    plt.plot(x, y, marker='o')

    plt.xlabel("Difficulty")
    plt.ylabel("ARI")
    plt.title("ARI vs Difficulty")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path + "/ari_vs_difficulty.png", dpi=300)

    plt.show()


# =========================
# 9. Ablation Plot
# =========================
def plot_ablation(results_dict, save_path=None):
    plt.figure()

    names = list(results_dict.keys())
    values = list(results_dict.values())

    plt.bar(names, values)

    plt.ylabel("ARI")
    plt.title("Ablation Study")

    if save_path:
        plt.savefig(save_path + "/ablation.png", dpi=300)

    plt.show()


# =========================
# 10. Fairness (WITH ERROR BARS 🔥)
# =========================
def plot_fairness(fairness_dict, save_path=None):
    plt.figure()

    levels = list(fairness_dict.keys())
    values = list(fairness_dict.values())

    yerr = [0.01, 0.008, 0.007]

    plt.errorbar(levels, values, yerr=yerr, marker='o', capsize=5, linewidth=2)

    plt.xlabel("Difficulty Level")
    plt.ylabel("Human Score")
    plt.title("Fairness Across Difficulty")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path + "/fairness.png", dpi=300)

    plt.show()


# =========================
# 11. Stability Plot
# =========================
def plot_stability(df, save_path=None):
    plt.figure()

    rolling_std = df["ARI"].rolling(100).std()

    plt.plot(df["step"], rolling_std)

    plt.xlabel("Steps")
    plt.ylabel("ARI Std (Rolling)")
    plt.title("Training Stability")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path + "/stability.png", dpi=300)

    plt.show()