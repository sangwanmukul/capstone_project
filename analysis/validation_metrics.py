import numpy as np
import pandas as pd

# =========================
# 1. Real OCR Test
# =========================
def real_ocr_test(sample_images):
    """
    Run OCR on sample CAPTCHA images
    sample_images: list of image paths
    """
    try:
        import pytesseract
        from PIL import Image
    except:
        return {"ocr_available": False}

    results = []

    for img_path in sample_images:
        try:
            text = pytesseract.image_to_string(Image.open(img_path))
            results.append(len(text.strip()))
        except:
            results.append(0)

    return {
        "ocr_available": True,
        "avg_extracted_length": float(np.mean(results)) if results else 0
    }


# =========================
# 2. Failure Case Analysis
# =========================
def failure_cases(df, threshold=0.1):
    """
    Extract cases where bot success is high
    """
    failures = df[df["bot_success"] > threshold]
    return failures.head(10)


# =========================
# 3. Baseline Comparison
# =========================
def baseline_comparison(df):
    """
    Compare with a simulated baseline CAPTCHA
    """
    baseline_bot = 0.30
    baseline_human = 0.90

    proposed_bot = df["bot_success"].mean()
    proposed_human = df["human_score"].mean()

    return {
        "baseline_bot": baseline_bot,
        "baseline_human": baseline_human,
        "proposed_bot": float(proposed_bot),
        "proposed_human": float(proposed_human)
    }


# =========================
# 4. Relative Improvement
# =========================
def relative_improvement(df, baseline_bot=0.30):
    """
    % reduction in bot success compared to baseline
    """
    proposed_bot = df["bot_success"].mean()
    improvement = (baseline_bot - proposed_bot) / baseline_bot
    return float(improvement)


# =========================
# 5. Confidence Interval (95%)
# =========================
def confidence_interval(data):
    """
    Compute 95% confidence interval
    """
    mean = np.mean(data)
    std = np.std(data)
    n = len(data)

    margin = 1.96 * (std / np.sqrt(n))
    return float(mean - margin), float(mean + margin)


# =========================
# 6. Effect Size (Cohen's d)
# =========================
def effect_size(df, baseline=0.3):
    """
    Compute Cohen's d effect size
    """
    data = df["ARI"]
    mean = np.mean(data)
    std = np.std(data) + 0.05

    d = (mean - baseline) / std
    return float(d)