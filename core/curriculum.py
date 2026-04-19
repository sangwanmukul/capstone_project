import numpy as np

class CurriculumScheduler:
    def __init__(self, config):
        self.lr = config["lr"]
        self.target_ari = config["target_ari"]
        self.entropy_weight = config.get("entropy_weight", 0.2)

    def update(self, theta, ari, entropy):
        new_theta = theta.copy()

        delta = (self.target_ari - ari) + self.entropy_weight * entropy

        new_theta["warp"] += self.lr * delta
        new_theta["clutter"] += self.lr * delta
        new_theta["variation"] += int(self.lr * delta * 8)

        # clamp
        new_theta["warp"] = float(np.clip(new_theta["warp"], 0.1, 0.5))
        new_theta["clutter"] = float(np.clip(new_theta["clutter"], 0.05, 0.4))
        new_theta["variation"] = int(np.clip(new_theta["variation"], 5, 30))
        return new_theta