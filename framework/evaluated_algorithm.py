from surprise import AlgoBase

from evaluating.recommender_metrics import rmse, mae, get_top_n, hit_rate, cumulative_hit_rate, \
    average_reciprocal_hit_rank, user_coverage, diversity
from framework.evaluation_data import EvaluationData


class EvaluatedAlgorithm:

    def __init__(
            self,
            algorithm: AlgoBase,
            name: str,
    ):
        self.algorithm: AlgoBase = algorithm
        self.name: str = name

    def evaluate(
            self,
            evaluation_data: EvaluationData,
            do_top_n: bool,
            n=10,
            verbose=True,
    ):
        metrics = {}

        if verbose:
            print(f"Evaluating {self.name}...")

        self.algorithm.fit(evaluation_data.get_train_set())

        predictions = self.algorithm.test(evaluation_data.get_test_set())

        metrics['rmse'] = rmse(predictions=predictions)
        metrics['mae'] = mae(predictions=predictions)

        if do_top_n:

            if verbose:
                print("Evaluating top-N with leave-one-out...")

            self.algorithm.fit(evaluation_data.get_loo_cv_train_set())

            left_out_predictions = self.algorithm.test(evaluation_data.get_loo_cv_test_set())

            all_predictions = self.algorithm.test(evaluation_data.get_loo_cv_anti_test_set())

            top_n_predicted = get_top_n(all_predictions, n)

            if verbose:
                print("Computing recommendations with full data set...")

            metrics["HR"] = hit_rate(top_n_predicted, left_out_predictions)
            metrics["cHR"] = cumulative_hit_rate(top_n_predicted, left_out_predictions)
            metrics["ARHR"] = average_reciprocal_hit_rank(top_n_predicted, left_out_predictions)

            if verbose:
                print(f"Done evaluating {self.name}!")

            self.algorithm.fit(evaluation_data.get_full_train_set())
            all_predictions = self.algorithm.test(evaluation_data.get_full_anti_test_set())
            top_n_predicted = get_top_n(all_predictions, n)

            if verbose:
                print("Analyzing coverage, diversity, and novelty...")

            metrics["coverage"] = user_coverage(top_n_predicted, evaluation_data.get_full_train_set().n_users)
            metrics["diversity"] = diversity(top_n_predicted, evaluation_data.get_sims_algo())

        return metrics

    def get_algorithm(self):
        return self.algorithm

    def get_name(self):
        return self.name
