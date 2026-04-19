import random
import numpy as np


class MultiAgentAttacker:
    def __init__(self):
        # List of different attacker types representing varying bot capabilities
        self.modes = ["weak", "average", "strong"]

    def evaluate(self, theta):
        # Extract CAPTCHA parameters from input dictionary
        # Use default values if keys are missing
        warp = theta.get("warp", 0.3)          # controls distortion of text
        clutter = theta.get("clutter", 0.2)    # controls background noise
        variation = theta.get("variation", 15) # controls randomness in styles/fonts

        # =========================
        # 1. Lightweight Difficulty Proxy
        # =========================
        # Combine parameters into a single difficulty score using weighted sum
        # Each parameter contributes differently to overall difficulty
        base = (
            0.6 * warp +                     # distortion has highest impact
            0.5 * clutter +                  # noise adds moderate difficulty
            0.4 * (variation / 30.0)         # normalize variation to [0,1] range
        )

        # Restrict difficulty to range [0, 1] for stability
        difficulty = float(np.clip(base, 0, 1))

        # =========================
        # 2. Attacker Mode Selection
        # =========================
        # Randomly choose an attacker type to simulate different bot strengths
        mode = random.choice(self.modes)

        # Assign numerical strength based on selected mode
        # Higher strength means better ability to solve CAPTCHA
        if mode == "weak":
            strength = 0.5
        elif mode == "strong":
            strength = 1.2
        else:
            strength = 0.8  # average attacker

        # =========================
        # 3. Bot Success
        # =========================
        # Compute probability of bot solving CAPTCHA using exponential decay
        # As difficulty increases → success decreases rapidly
        # Stronger bots reduce the effect of difficulty
        bot_success = np.exp(-12 * difficulty * strength)

        # Add small random noise to simulate real-world variability
        bot_success += random.uniform(-0.01, 0.01)

        # Clamp bot success to avoid extreme values (too low or too high)
        bot_success = float(np.clip(bot_success, 0.01, 0.35))

        # =========================
        # 4. Entropy (Calibration-aware)
        # =========================
        # Estimate uncertainty (entropy) based on bot success
        # Add small Gaussian noise for realism
        entropy = bot_success + np.random.normal(0, 0.005)

        # Ensure entropy stays within valid probability bounds [0, 1]
        entropy = float(np.clip(entropy, 0, 1))

        # =========================
        # 5. Return Results
        # =========================
        # Output dictionary containing attack simulation results
        return {
            "bot_success": bot_success,  # probability that bot solves CAPTCHA
            "entropy": entropy,          # uncertainty in prediction
            "mode": mode                # type of attacker used
        }