import numpy as np
import lightgbm as lgb
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold

class LrSchedulingCallback(object):
    """ラウンドごとの学習率を動的に制御するためのコールバック"""

    def __init__(self, strategy_func):
        # 学習率を決定するための関数
        self.scheduler_func = strategy_func
        # 検証用データに対する評価指標の履歴
        self.eval_metric_history = []

    def __call__(self, env):
        # 現在の学習率を取得する
        current_lr = env.params.get('learning_rate')

        # 検証用データに対する評価結果を取り出す (先頭の評価指標)
        first_eval_result = env.evaluation_result_list[0]
        # スコア
        metric_score = first_eval_result[2]
        # 評価指標は大きい方が優れているか否か
        is_higher_better = first_eval_result[3]

        # 評価指標の履歴を更新する
        self.eval_metric_history.append(metric_score)
        # 現状で最も優れたラウンド数を計算する
        best_round_find_func = np.argmax if is_higher_better else np.argmin
        best_round = best_round_find_func(self.eval_metric_history)

        # 新しい学習率を計算する
        new_lr = self.scheduler_func(current_lr=current_lr,
                                     eval_history=self.eval_metric_history,
                                     best_round=best_round,
                                     is_higher_better=is_higher_better)

        # 次のラウンドで使う学習率を更新する
        update_params = {
            'learning_rate': new_lr,
        }
        env.model.reset_parameter(update_params)
        env.params.update(update_params)

    @property
    def before_iteration(self):
        # コールバックは各イテレーションの後に実行する
        return False

