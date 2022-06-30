"""
著作 Azunyan https://github.com/Azunyan1111/OneMax
"""
"""
改変 Copyright 2020 ground0state All Rights Reserved.
"""
import random
import time
from abc import ABCMeta, abstractmethod


class Individual():
    """個体.

    Parameters
    ----------
    chromosome : list of {0 or 1}
        染色体.

    evaluation : float
        評価.
    """
    chromosome = None
    evaluation = None

    def __init__(self, chromosome, evaluation):
        self.chromosome = chromosome
        self.evaluation = evaluation


class GaSolver(metaclass=ABCMeta):
    """遺伝的アルゴリズムの抽象クラス.
    染色体に対して、評価値を出力するメソッド「evaluate_individual」は要実装.

    Parameters
    ----------
    chromosome_length : int
        染色体の長さ.

    population_size : int
        集団の大きさ.

    pick_out_size : int
        エリート染色体選抜数.

    individual_mutation_probability : float
        個体突然変異確率.

    gene_mutation_probability : float
        遺伝子突然変異確率.

    iteration : int
        繰り返す世代数.
    """

    def __init__(self, chromosome_length, population_size, pick_out_size,
        individual_mutation_probability=0.3, gene_mutation_probability=0.1, iteration=1, verbose=True):        
        self.chromosome_length = chromosome_length
        self.population_size = population_size
        self.pick_out_size = pick_out_size
        self.individual_mutation_probability = individual_mutation_probability
        self.gene_mutation_probability = gene_mutation_probability
        self.iteration = iteration
        self.verbose = verbose
        self.history = None

    def _create_individual(self, length):
        """引数で指定された桁のランダムな染色体を生成、格納した個体を返します.

        Parameters
        ----------
        length : int
            染色体の長さ.

        Returns
        -------
        individual : Individual
            個体.
        """
        individual = Individual([random.randint(0, 1) for i in range(length)], 0)
        return individual

    @abstractmethod
    def evaluate_individual(self, individual, X, y):
        """評価関数.

        Parameters
        ----------
        individual : Individual
            個体.
        X : pandas.DataFrame
            説明変数.
        y : pandas.DataFrame
            目的変数.

        Returns
        -------
        eval : float
            評価値.
        """
        raise NotImplementedError()

    def _extract_elites(self, population, num):
        """選択関数.

        Parameters
        ----------
        population : list of Individual
            集団.
        num : int
            個体選択数.

        Returns
        -------
        elites : list of Individual
            選択処理をした集団.
        """        
        # 現行世代個体集団の評価を小さい順にソートする
        sort_result = sorted(population, reverse=False, key=lambda individual: individual.evaluation)

        # 一定の上位を抽出する
        elites = sort_result[:num]
        return elites

    def _crossover(self, individual1, individual2, chromosome_length):
        """交叉関数.
        二点交叉を行います.

        Parameters
        ----------
        individual1 : Individual
            交叉する個体1.
        individual2 : Individual
            交叉する個体2.
        chromosome_length : int
            染色体の長さ.

        Returns
        -------
        offsprings : list of Individual
            二つの孫.
        """

        # 入れ替える二点の点を設定します
        cross_one = random.randint(0, chromosome_length)
        cross_second = random.randint(cross_one, chromosome_length)

        # 遺伝子を取り出します
        one = individual1.chromosome
        second = individual2.chromosome

        # 交叉させます
        progeny_one = one[:cross_one] + second[cross_one:cross_second] + one[cross_second:]
        progeny_second = second[:cross_one] + one[cross_one:cross_second] + second[cross_second:]

        # 子孫
        offsprings = [Individual(progeny_one, 0), Individual(progeny_second, 0)]
        return offsprings

    def _create_next_generation(self, population, elites, offsprings):
        """世代交代処理を行います.

        Parameters
        ----------
        population : list of Individual
            現行世代個体集団.
        elites : list of Individual
            現行世代エリート集団.
        offsprings : list of Individual
            現行世代子孫集団.

        Returns
        -------
        next_generation_population : list of Individual
            次世代個体集団.
        """
        # 現行世代個体集団の評価を低い順番にソートする
        next_generation_population = sorted(population, reverse=False, key=lambda individual: individual.evaluation)

        # 追加するエリート集団と子孫集団の合計ぶんを取り除く
        next_generation_population = next_generation_population[len(elites)+len(offsprings):]

        # エリート集団と子孫集団を次世代集団を次世代へ追加します
        next_generation_population.extend(elites)
        next_generation_population.extend(offsprings)
        return next_generation_population

    def _mutation(self, population, induvidual__mutation_probability, gene__mutation_probability):
        """突然変異関数.

        Parameters
        ----------
        population : list of Individual
            集団.
        induvidual__mutation_probability : float in [0, 1]
            個体突然変異確率.
        gene__mutation_probability : float in [0, 1]
            遺伝子突然変異確率.

        Returns
        -------
        new_population : list of Individual
            突然変異処理した集団.
        """
        new_population = []
        for individual in population:
            # 個体に対して一定の確率で突然変異が起きる
            if induvidual__mutation_probability > random.random():
                new_chromosome = []
                for gene in individual.chromosome:
                    # 個体の遺伝子情報一つ一つに対して突然変異がおこる
                    if gene__mutation_probability > random.random():
                        new_chromosome.append(random.randint(0, 1))
                    else:
                        new_chromosome.append(gene)

                individual.chromosome = new_chromosome
                new_population.append(individual)
            else:
                new_population.append(individual)

        return new_population

    def solve(self, X, y):
        """遺伝的アルゴリズムのメインクラス.

        Returns
        -------
        list of {0 or 1}
            最も優れた個体の染色体.
        """
        self.history = {"Min":[], "Max":[], "Avg":[], "BestChromosome":[]}

        # 現行世代の個体集団を初期化します
        current_generation_population = [self._create_individual(self.chromosome_length) for i in range(self.population_size)]

        # 現行世代個体集団の個体を評価
        for individual in current_generation_population:
            individual.evaluation = self.evaluate_individual(individual, X, y)

        for count in range(self.iteration):
            # 各ループの開始時刻
            start = time.time()

            # エリート個体を選択します
            elites = self._extract_elites(current_generation_population, self.pick_out_size)

            # エリート遺伝子を交叉させ、リストに格納します
            offsprings = []
            for i in range(0, self.pick_out_size-1):
                offsprings.extend(self._crossover(elites[i], elites[i+1], self.chromosome_length))

            # 次世代個体集団を現行世代、エリート集団、子孫集団から作成します
            next_generation_population = self._create_next_generation(current_generation_population, elites, offsprings)

            # 次世代個体集団全ての個体に突然変異を施します。
            next_generation_population = self._mutation(next_generation_population,
                                                  self.individual_mutation_probability,
                                                  self.gene_mutation_probability)

            # 現行世代個体集団の個体を評価
            for individual in current_generation_population:
                individual.evaluation = self.evaluate_individual(individual, X, y)

            # 1世代の進化的計算終了。評価に移ります

            # 各個体の評価値を配列化します。
            fits = [individual.evaluation for individual in current_generation_population]

            # 最も評価値のよい個体を取り出します（評価値の小さいものを取得）
            best_individual = self._extract_elites(current_generation_population, 1)
            best_chromosome = best_individual[0].chromosome
            print('最小値の遺伝子：')
            print(best_chromosome)

            # 進化結果を評価します
            min_val = min(fits)
            max_val = max(fits)
            avg_val = sum(fits) / len(fits)

            # 現行世代の進化結果を出力します
            if self.verbose:
                print("-----第{}世代の結果-----".format(count+1))
                print("  Min:{}".format(min_val))
                print("  Max:{}".format(max_val))
                print("  Avg:{}".format(avg_val))

            # history作成
            self.history["Min"].append(min_val)
            self.history["Max"].append(max_val)
            self.history["Avg"].append(avg_val)
            self.history["BestChromosome"].append(best_chromosome)

            # 現行世代と次世代を入れ替えます
            current_generation_population = next_generation_population

            # 時間計測
            elapsed_time = time.time() - start
            print ("  {}/{} elapsed_time:{:.2f}".format(count+1, self.iteration, elapsed_time) + "[sec]")

        # 最終結果出力
        if self.verbose:
            print("")  # 改行
            print("最も優れた個体は{}".format(elites[0].chromosome))

        return self.history