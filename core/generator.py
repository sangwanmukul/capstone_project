import random

class ChallengeSynthesizer:
    def __init__(self):
        # Balanced bounds → good for humans + strong against bots
        self.bounds = {
            "warp": (0.1, 0.4),
            "clutter": (0.1, 0.3),
            "variation": (10, 25),
            "entropy": (0.3, 0.7)
        }

    def sample(self):
        return {
            "warp": random.uniform(*self.bounds["warp"]),
            "clutter": random.uniform(*self.bounds["clutter"]),
            "variation": random.randint(*self.bounds["variation"]),
            "entropy": random.uniform(*self.bounds["entropy"])
        }

    def sample_with_difficulty(self, difficulty="medium"):
        if difficulty == "easy":
            return {
                "warp": random.uniform(0.1, 0.25),
                "clutter": random.uniform(0.05, 0.15),
                "variation": random.randint(10, 18),
                "entropy": random.uniform(0.3, 0.5)
            }

        elif difficulty == "hard":
            return {
                "warp": random.uniform(0.3, 0.5),
                "clutter": random.uniform(0.2, 0.4),
                "variation": random.randint(18, 30),
                "entropy": random.uniform(0.5, 0.8)
            }

        return self.sample()