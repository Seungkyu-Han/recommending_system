import random
import numpy as np
from surprise import KNNBasic, NormalPredictor

from content_based.evaluator import Evaluator
from movie_lens import MovieLens

movie_lens = MovieLens()

data = movie_lens.load_movie_lens_latest_small()

rankings = movie_lens.get_popularity_rankings()

np.random.seed(0)
random.seed(0)

evaluator = Evaluator(data, rankings)

user_knn = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
evaluator.add_algorithm(user_knn, 'user KNN')

item_knn = KNNBasic(sim_options={'name': 'cosine', 'user_based': False})
evaluator.add_algorithm(item_knn, 'item KNN')

random = NormalPredictor()
evaluator.add_algorithm(random, 'Random')

evaluator.evaluate(do_top_n=True)
