from surprise import KNNBaseline
from surprise.dataset import DatasetAutoFolds
from surprise.model_selection import train_test_split, LeaveOneOut


class EvaluationData:

    def __init__(
            self,
            data: DatasetAutoFolds,
            popularity_rankings: dict[int, int],
    ):
        self.popularity_rankings: dict[int, int] = popularity_rankings

        self.full_train_set = data.build_full_trainset()
        self.full_anti_test_set = self.full_train_set.build_anti_testset()

        self.train_set, self.test_set = train_test_split(data, test_size=0.25, random_state=1)

        loo_cv = LeaveOneOut()

        for train, test in loo_cv.split(data):
            self.loo_train_set = train
            self.loo_test_set = test

        self.loo_cv_anti_test_set = self.loo_train_set.build_anti_testset()

        sim_options = {'name': 'cosine', 'user_based': False}
        self.sims_algo = KNNBaseline(sim_options=sim_options)
        self.sims_algo.fit(self.full_train_set)

    def get_full_train_set(self):
        return self.full_train_set

    def get_full_anti_test_set(self):
        return self.full_anti_test_set

    def get_anti_test_set_for_user(self, test_subject):

        train_set = self.full_train_set

        fill = train_set.global_mean

        u = train_set.to_inner_uid(str(test_subject))

        user_items = set([j for (j, _) in train_set.ur[u]])

        anti_test_set = [(train_set.to_raw_uid(u), train_set.to_raw_iid(i), fill) for i in train_set.all_items() if i not in user_items]

        return anti_test_set

    def get_train_set(self):
        return self.train_set

    def get_test_set(self):
        return self.test_set

    def get_loo_cv_train_set(self):
        return self.loo_train_set

    def get_loo_cv_test_set(self):
        return self.loo_test_set

    def get_loo_cv_anti_test_set(self):
        return self.loo_cv_anti_test_set

    def get_sims_algo(self):
        return self.sims_algo

    def get_popularity_rankings(self):
        return self.popularity_rankings