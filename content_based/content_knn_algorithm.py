from surprise import AlgoBase, Trainset

from movie_lens import MovieLens


class ContentKNNAlgorithm(AlgoBase):

    def __init__(
            self,
            k: int = 40,
    ):
        super().__init__()
        self.k = k

    def fit(
            self,
            trainset: Trainset,
    ):
        super().fit(trainset)

        movie_lens = MovieLens()

        