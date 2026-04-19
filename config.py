import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

SIMULATION_CONFIG = {
    "steps": 5000,
    "log_interval": 50,
    "modes": ["M1_STATIC", "M2_HEURISTIC", "M3_CURRICULUM", "M4_TCCF"]
}

CURRICULUM_CONFIG = {
    "lr": 0.08,
    "entropy_weight": 0.4,
    "target_ari": 0.7
}