import heapq
from collections import defaultdict

from surprise import KNNBasic

from collaborative_filtering.simple_user_cf import k_neighbors
from movie_lens import MovieLens

test_subject = '85'
k = 10

movie_lens = MovieLens()

data = movie_lens.load_movie_lens_latest_small()

train_set = data.build_full_trainset()

sim_options = {
    'name': 'cosine',
    'user_based': False,
}

model = KNNBasic(sim_options=sim_options, verbose=False)
model.fit(train_set)
sims_matrix = model.compute_similarities()

test_user_inner_id = train_set.to_inner_uid(test_subject)
test_user_ratings = train_set.ur[test_user_inner_id]

k_neighbors = heapq.nlargest(k, test_user_ratings, key=lambda x: x[1])

candidates = defaultdict(float)

for item_id, rating in k_neighbors:
    similar_items = sims_matrix[item_id]
    for inner_id, score in enumerate(similar_items):
        candidates[inner_id] += (score * (rating / 5.0))

watched = {}

for item_id, rating in train_set.ur[test_user_inner_id]:
    watched[item_id] = 1

pos = 0

for item_id, rating_sum in sorted(candidates.items(), key=lambda x: x[1], reverse=True):
    if not item_id in watched:
        movie_id = train_set.to_raw_iid(item_id)
        print(pos, movie_id, rating_sum)
        pos += 1
        if pos > k:
            break