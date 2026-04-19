# engine/simulator.py

import pandas as pd
import numpy as np
from core.metrics import compute_ari


# =========================
# 🔥 Unified Difficulty Function
# =========================
def compute_difficulty(theta, entropy):
    warp = theta.get("warp", 0.3)
    clutter = theta.get("clutter", 0.2)
    variation = theta.get("variation", 15) / 30.0

    difficulty = (
        0.6 * (warp ** 1.5) +
        0.5 * (clutter ** 1.3) +
        0.4 * (variation ** 1.2) +
        0.3 * entropy
    )

    return float(np.clip(difficulty, 0, 1))


# =========================
# 🔥 Simulation Engine
# =========================
class SimulationEngine:
    def __init__(self, generator, attacker, human, curriculum):
        self.generator = generator
        self.attacker = attacker
        self.human = human
        self.curriculum = curriculum

    def run(self, steps):
        theta = self.generator.sample()
        logs = []

        ari_history = []

        for step in range(steps):

            # =========================
            # 1. Attacker Evaluation
            # =========================
            attack = self.attacker.evaluate(theta)

            # =========================
            # 2. Compute Unified Difficulty
            # =========================
            difficulty = compute_difficulty(theta, attack["entropy"])

            # =========================
            # 3. Human Evaluation
            # =========================
            human_eval = self.human.evaluate(theta)

            # =========================
            # 4. Compute ARI (CORRECTED)
            # =========================
            ari = compute_ari(
                human_eval["human_score"],
                attack["bot_success"],
                difficulty   # ✅ FIXED (not cognitive_load)
            )

            # =========================
            # 5. Smooth ARI (stability)
            # =========================
            ari_history.append(ari)

            if len(ari_history) > 10:
                ari = 0.8 * ari_history[-1] + 0.2 * ari

            # =========================
            # 6. Log everything
            # =========================
            logs.append({
                "step": step,
                **theta,
                **attack,
                **human_eval,
                "difficulty": difficulty,   # ✅ IMPORTANT (for plots)
                "ARI": ari
            })

            # =========================
            # 7. Curriculum Update
            # =========================
            theta = self.curriculum.update(theta, ari, attack["entropy"])

        return pd.DataFrame(logs)