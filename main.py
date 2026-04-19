# =========================
# 🔹 IMPORTS
# =========================
# Load configuration settings
from config import SIMULATION_CONFIG, CURRICULUM_CONFIG

# Core pipeline components
from core.generator import ChallengeSynthesizer      # Generates CAPTCHA parameters
from core.attacker import MultiAgentAttacker         # Simulates bot attacks
from core.human_model import HumanBehaviorProxy      # Simulates human behavior
from core.curriculum import CurriculumScheduler      # Updates parameters adaptively

# Engine and utilities
from engine.simulator import SimulationEngine        # Runs full simulation loop
from engine.logger import ExperimentLogger           # Saves results

# Machine Learning
from models.trainer import ModelTrainer              # Trains LightGBM model

# Analysis modules
from analysis.evaluator import summarize             # Basic summary
from analysis.metrics import *                       # Advanced analysis metrics
from analysis.validation_metrics import *            # Validation metrics

# CAPTCHA + OCR
from captcha_image_generator import generate_and_save_captchas  # Generate real CAPTCHA images
from real_world_eval import evaluate_ocr_folder                 # OCR evaluation

# Visualization
from visualization import *                          # Plotting functions

# Standard libraries
import os
import pandas as pd

# Start message
print("🚀 Running Advanced CAPTCHA Framework...")


# =========================
# 🔹 INIT (Initialize system components)
# =========================
generator = ChallengeSynthesizer()                   # Generates CAPTCHA configs (theta)
attacker = MultiAgentAttacker()                      # Bot simulation module
human = HumanBehaviorProxy()                         # Human simulation module
curriculum = CurriculumScheduler(CURRICULUM_CONFIG)  # Adaptive learning module

# Create simulation engine (core loop controller)
engine = SimulationEngine(generator, attacker, human, curriculum)


# =========================
# 🔹 RUN SIMULATION
# =========================
# Run simulation for given number of steps (learning loop)
df = engine.run(SIMULATION_CONFIG["steps"])

# Save simulation results to CSV
logger = ExperimentLogger()
logger.save(df)


# =========================
# 🔹 TRAIN ML MODEL
# =========================
# Train LightGBM model on simulation data
trainer = ModelTrainer()
model = trainer.train(df)


# =========================
# 🔹 SUMMARY
# =========================
# Compute summary statistics (mean ARI, etc.)
summary = summarize(df)

print("\n📊 FINAL RESULTS:")
for k, v in summary.items():
    print(f"{k}: {v:.4f}")


# =========================
# 🔹 ADVANCED ANALYSIS
# =========================
print("\n🔬 Running Advanced Analysis...")

# Analyze impact of parameter changes
cf_df = counterfactual_analysis(df)

# Check uncertainty calibration
calib = uncertainty_calibration(df)

# Measure diversity in adversarial behavior
div = adversarial_diversity(df)

print(f"• Counterfactual Impact: {cf_df['impact'].mean():.4f}")
print(f"• Calibration Error: {calib['expected_calibration_error']:.4f}")
print(f"• Diversity: {div['mean_diversity']:.4f}")


# =========================
# 🔹 HIGH LEVEL EVALUATION
# =========================
# Stress testing (worst-case performance)
stress = stress_test(df)

# Ablation study (remove features and compare)
ablation = ablation_study(df)

# Fairness analysis across conditions
fairness = fairness_analysis(df)

print("\n🧪 High-Level Evaluation:")
print(f"• Worst-case Bot: {stress['worst_case_bot_success']:.4f}")
print(f"• Worst-case ARI: {stress['worst_case_ARI']:.4f}")

print(f"• Base ARI: {ablation['base_ARI']:.4f}")
print(f"• No Warp: {ablation['without_warp']:.4f}")
print(f"• No Entropy: {ablation['without_entropy']:.4f}")

print("• Fairness:")
for k, v in fairness.items():
    print(f"   {k}: {v:.4f}")


# =========================
# 🔹 VALIDATION
# =========================
print("\n🧪 Running Validation & Real-World Tests...")

# Load real CAPTCHA images if folder exists
sample_images = []
if os.path.exists("real_captcha_samples"):
    sample_images = [
        os.path.join("real_captcha_samples", f)
        for f in os.listdir("real_captcha_samples")
        if f.endswith(".png")
    ]

# Run OCR-based evaluation
ocr = real_ocr_test(sample_images=sample_images)

# Print OCR results if available
if ocr.get("ocr_available", False):
    print(f"• OCR Avg Extracted Length: {ocr['avg_extracted_length']:.4f}")
else:
    print("• OCR Test skipped (pytesseract not available)")

# Identify and save failure cases
fail_df = failure_cases(df)
fail_df.to_csv("results/failure_cases.csv", index=False)
print("• Failure cases saved → results/failure_cases.csv")

# Compare with baseline system
base = baseline_comparison(df)
print(f"• Baseline Bot Success: {base['baseline_bot']:.2f}")
print(f"• Proposed Bot Success: {base['proposed_bot']:.4f}")

# Compute relative improvement
improve = relative_improvement(df)
print(f"• Relative Improvement: {improve*100:.2f}%")

# Confidence interval for ARI
low, high = confidence_interval(df["ARI"])
print(f"• 95% CI (ARI): [{low:.4f}, {high:.4f}]")

# Effect size (Cohen's d)
d = effect_size(df)
print(f"• Effect Size (Cohen's d): {d:.4f}")


# =========================
# 🔹 REAL-WORLD OCR
# =========================
print("\n🖼️ Generating real CAPTCHA samples...")
generate_and_save_captchas(20)   # Generate 20 CAPTCHA images

print("\n🌍 Running Real-World OCR Evaluation...")
ocr_results = evaluate_ocr_folder()

# Print OCR accuracy results
if ocr_results.get("samples_tested", 0) > 0:
    print(f"• OCR Avg Accuracy: {ocr_results['avg_accuracy']:.4f}")
    print(f"• OCR Max Accuracy: {ocr_results['max_accuracy']:.4f}")
    print(f"• Samples Tested: {ocr_results['samples_tested']}")
else:
    print("• OCR evaluation skipped or no data available")


# =========================
# 🔹 HUMAN EVALUATION
# =========================
print("\n👤 Human Evaluation Summary:")

try:
    # Load human evaluation results
    human_df = pd.read_csv("results/human_results.csv")

    # Compute accuracy and average solving time
    human_acc = human_df["correct"].mean()
    human_time = human_df["time"].mean()

    print(f"• Accuracy: {human_acc:.4f}")
    print(f"• Avg Time: {human_time:.2f} sec")

except Exception as e:
    # If file not found, notify user
    print("⚠️ Human evaluation file not found. Run human_ui.py first.")


# =========================
# 🔹 VISUALIZATION PLOTS
# =========================
print("\n📊 Generating Plots...")

# Generate all analysis plots
plot_ari_curve(df, save_path="results")
plot_ari_distribution(df, save_path="results")
plot_ari_vs_difficulty(df, save_path="results")
plot_difficulty_bins(df, save_path="results")

# Ablation comparison plot
plot_ablation({
    "Base": ablation["base_ARI"],
    "No Warp": ablation["without_warp"],
    "No Entropy": ablation["without_entropy"]
}, save_path="results")

# Fairness and feature analysis plots
plot_fairness(fairness, save_path="results")
plot_feature_evolution(df, save_path="results")
plot_tradeoff(df, save_path="results")

# Calibration and stability plots
plot_calibration(df, save_path="results")
plot_stability(df, save_path="results")

print("\n✅ All plots generated and saved in /results/")