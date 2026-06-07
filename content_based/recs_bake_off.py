from surprise import SVD, NormalPredictor

from framework.evaluator import Evaluator
from movie_lens import MovieLens

import random
import numpy as np

movie_lens = MovieLens()

print('loading movie ratings...')

data = movie_lens.load_movie_lens_latest_small()

print('\nComputing movie popularity ranks so we can measure novelty later')

rankings = movie_lens.get_popularity_rankings()

np.random.seed(0)
random.seed(0)

evaluator = Evaluator(data, rankings)

svd_algorithm = SVD(random_state=10)
evaluator.add_algorithm(svd_algorithm, 'SVD')

random = NormalPredictor()
evaluator.add_algorithm(random, 'Random')

evaluator.evaluate(do_top_n=True)