def compute_ari(human_score, bot_success, difficulty):
    import numpy as np

    alpha = 0.45
    beta = 0.45
    gamma = 0.1

    human_score = np.clip(human_score, 0, 1)
    bot_success = np.clip(bot_success, 0, 1)
    difficulty = np.clip(difficulty, 0, 1)

    ari = (
        alpha * human_score +
        beta * (1 - bot_success) -
        gamma * difficulty
    )

    return float(np.clip(ari, 0, 1))

def robustness_variance(history):
    import numpy as np
    return np.var(history)