# core/human_model.py

import numpy as np


class HumanBehaviorProxy:
    def evaluate(self, theta):
        warp = theta.get("warp", 0.3)
        clutter = theta.get("clutter", 0.2)
        variation = theta.get("variation", 15)
        entropy = theta.get("entropy", 0.5)

        # =========================
        # 1. Unified Difficulty
        # =========================
        difficulty = (
            0.6 * (warp ** 1.5) +
            0.5 * (clutter ** 1.3) +
            0.4 * ((variation / 30.0) ** 1.2) +
            0.3 * entropy
        )

        difficulty = float(np.clip(difficulty, 0, 1))

        # =========================
        # 2. HUMAN PERFORMANCE (FINAL FIX 🔥)
        # =========================
        base_performance = 0.93

        # smoother degradation (more stable)
        difficulty_penalty = 0.25 * (difficulty ** 1.15)

        # 🔥 balanced fairness correction (not aggressive)
        if difficulty > 0.7:
            difficulty_penalty *= 0.88   # moderate boost (not too strong)
        elif difficulty < 0.25:
            difficulty_penalty *= 1.05   # slight control (not harsh)

        # realistic human variability
        noise = np.random.normal(0, 0.02)

        human_score = base_performance - difficulty_penalty + noise

        # stable bounds (avoid extreme values)
        human_score = float(np.clip(human_score, 0.72, 0.92))

        # =========================
        # 3. Reaction Time
        # =========================
        delay = 2.0 + 3.5 * difficulty + np.random.normal(0, 0.3)
        delay = float(np.clip(delay, 2, 10))

        # =========================
        # 4. Cognitive Load
        # =========================
        cognitive_load = difficulty

        # =========================
        # 5. Return
        # =========================
        return {
            "human_score": human_score,
            "delay": delay,
            "cognitive_load": cognitive_load
        }