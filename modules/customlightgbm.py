import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from modules.dinamiclr import LrSchedulingCallback
from sklearn.metrics import mean_absolute_error

# BaseEstimatorとRegressorMixinを継承する
class CustomLightGBM(BaseEstimator, RegressorMixin):
    # fit()を実装
    def __init__(self):
        self.params = {'num_leaves': 50, 'max_depth': 12, 'feature_fraction': 0.9936244753324097, 'lambda_l1': 1.99436831601929, 'lambda_l2': 41.53561366155952, 'subsample_freq': 4, 'bagging_fraction': 0.9055551419531833, 'min_data_in_leaf': 2, 'tree_learner': 'voting', 'learning_rate': 0.3, 'objective': 'regression', 'metric': 'mae', 'boosting': 'gbdt', 'verbosity': -1, 'random_state': 42, 'early_stopping_rounds': 50}
    
        
    def sample_scheduler_func(self, current_lr, eval_history, best_round, is_higher_better):
        """次のラウンドで用いる学習率を決定するための関数 (この中身を好きに改造する)

        :param current_lr: 現在の学習率 (指定されていない場合の初期値は None)
        :param eval_history: 検証用データに対する評価指標の履歴
        :param best_round: 現状で最も評価指標の良かったラウンド数
        :param is_higher_better: 高い方が性能指標として優れているか否か
        :return: 次のラウンドで用いる学習率

        NOTE: 学習を打ち切りたいときには callback.EarlyStopException を上げる
        """
        # 学習率が設定されていない場合のデフォルト
        current_lr = current_lr or 0.2

        # 試しに 20 ラウンド毎に学習率を半分にしてみる
        if len(eval_history) > 900:
            if len(eval_history) % 100 == 0:
                current_lr /= 1.1

        # 小さすぎるとほとんど学習が進まないので下限も用意する
        min_threshold = 0.01
        current_lr = max(min_threshold, current_lr)

        # if len(eval_history) % 300 == 0:
            # print('現在の学習率：{}'.format(current_lr))

        return current_lr
    
    def fit(self, X, y):
        # X, yを利用して、train/valid-poolを作成する。
        print('学習開始：LightGBM')
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_valid, y_valid)
        lr_scheduler_cb = LrSchedulingCallback(strategy_func=self.sample_scheduler_func)
        
        callbacks = [
            lgb.log_evaluation(500),       # ログを500置きに表示
            lr_scheduler_cb,
        ]
        self.model = lgb.train(
                        params=self.params,                    # ハイパーパラメータをセット
                        train_set=lgb_train,              # 訓練データを訓練用にセット
                        valid_sets=[lgb_train, lgb_eval], # 訓練データとテストデータをセット
                        valid_names=['Train', 'Test'],    # データセットの名前をそれぞれ設定
                        callbacks=callbacks,
                        num_boost_round = 50000                   
                    )
        
        self.is_fitted_ = True
        print(f'Score: {mean_absolute_error(y_valid, self.model.predict(X_valid))}')
        print('学習終了：LightGBM')
        
        # fit は self を返す
        return self

    # predict()を実装
    def predict(self, X):
        return self.model.predict(X)
    
    def set_params(self, params):
        self.params = params
        return self