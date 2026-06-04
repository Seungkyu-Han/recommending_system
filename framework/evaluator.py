from surprise import AlgoBase
from surprise.dataset import DatasetAutoFolds

from framework.evaluation_data import EvaluationData
from framework.evaluated_algorithm import EvaluatedAlgorithm


class Evaluator:

    algorithms: list[EvaluatedAlgorithm] = []

    def __init__(
            self,
            dataset: DatasetAutoFolds,
            rankings: dict[int, int]
    ):
        self.dataset = EvaluationData(dataset, rankings)

    def add_algorithm(
            self,
            algorithm: AlgoBase,
            name: str,
    ):
        algo = EvaluatedAlgorithm(algorithm, name)
        self.algorithms.append(algo)

    def evaluate(
            self,
            do_top_n: bool,
    ):
        results = {}

        for algorithm in self.algorithms:
            print(f"Evaluating {algorithm.get_name()}")
            results[algorithm.get_name()] = algorithm.evaluate(self.dataset, do_top_n)

        print("\n")


        if do_top_n:
            print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                "Algorithm",
                "RMSE",
                "MAE",
                "HR",
                "cHR",
                "ARHR",
                "Coverage",
                "Diversity",
                "Novelty",
            ))
            for name, metrics in results.items():
                print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                    name,
                    metrics["rmse"],
                    metrics["mae"],
                    metrics["HR"],
                    metrics["cHR"],
                    metrics["ARHR"],
                    metrics["coverage"],
                    metrics["diversity"],
                ))
        else:
            print("{:<10} {:<10} {:<10}".format(
                "Algorithm",
                "RMSE",
                "MAE",
            ))
            for name, metrics in results.items():
                print("{:<10} {:<10.4f} {:<10.4f}".format(
                    name,
                    metrics["rmse"],
                    metrics["mae"],
                ))