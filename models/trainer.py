import lightgbm as lgb

class ModelTrainer:
    def train(self, df):
        X = df[["warp", "clutter", "variation", "entropy"]]
        y = df["ARI"]

        model = lgb.LGBMRegressor(
            n_estimators=300,
            min_gain_to_split=0.0,
            min_data_in_leaf=5,
            verbose=-1   # 🔥 THIS LINE REMOVES WARNINGS
        )
        model.fit(X, y)

        return model