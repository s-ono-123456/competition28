import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from catboost import Pool
from catboost import CatBoostRegressor

# BaseEstimatorとRegressorMixinを継承する
class CustomCatBoostRegressor(BaseEstimator, RegressorMixin):
    # fit()を実装
    def __init__(self):
        self.params = {'l2_leaf_reg': 33.51917340457483, 'random_strength': 0.16779085114202497, 'subsample': 0.9828709713363581, 'objective': 'MAE', 'colsample_bylevel': 0.09, 'depth': 12, 'boosting_type': 'Plain', 'bootstrap_type': 'Bernoulli', 'eval_metric': 'MAE', 'learning_rate': 0.1, 'early_stopping_rounds': 50, 'iterations': 20000, 'verbose': 500, 'loss_function': 'MAE', 'random_seed': 42}

    def fit(self, X, y):
        # X, yを利用して、train/valid-poolを作成する。
        print('学習開始：CatBoost')
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
        train_pool = Pool(X_train, y_train)
        validate_pool = Pool(X_valid, y_valid)
        self.model = CatBoostRegressor(**self.params)
        self.model.fit(train_pool, eval_set=validate_pool)
        self.is_fitted_ = True
        
        print('学習終了：CatBoost')
        # fit は self を返す
        return self

    # predict()を実装
    def predict(self, X):
        return self.model.predict(X)
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self