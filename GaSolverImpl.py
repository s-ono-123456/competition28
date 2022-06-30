from sklearn.model_selection import train_test_split
import lightgbm as lgb
from Individual import GaSolver
import optuna
from sklearn.metrics import mean_absolute_error

class GaSolverImpl(GaSolver):

    # override
    def evaluate_individual(self, individual, X, y):
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': 42,
            'max_depth': 7,
            "feature_fraction": 0.8,
            'subsample_freq': 1,
            "bagging_fraction": 0.95,
            'min_data_in_leaf': 2,
            'learning_rate': 0.1,
            "boosting": "gbdt",
            "lambda_l1": 0.1,
            "lambda_l2": 10,
            "verbosity": -1,
            "random_state": 42,
            "early_stopping_rounds": 100
        }
        
        use_cols = [bool(gene) for gene in individual.chromosome]
        X_temp = X.iloc[:, use_cols]

        scores = []
        # for _ in range(30):
        X_train, X_test, y_train, y_test = train_test_split(X_temp, y, test_size=0.3, random_state=42)

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
        lgb_results = {}
        callbacks = [
            lgb.log_evaluation(-1),       # ログを100置きに表示
            lgb.record_evaluation(lgb_results),
        ]
        model = lgb.train(
                params=params,                    # ハイパーパラメータをセット
                train_set=lgb_train,              # 訓練データを訓練用にセット
                valid_sets=[lgb_train, lgb_test], # 訓練データとテストデータをセット
                valid_names=['Train', 'Test'],    # データセットの名前をそれぞれ設定
                callbacks=callbacks,
                num_boost_round = 10
                )
        test_pred = model.predict(X_test, num_iteration=model.best_iteration)
        mae = mean_absolute_error(y_test, test_pred)
#         scores.append(mae)

#         eval = float(np.array(scores).mean())
        return mae