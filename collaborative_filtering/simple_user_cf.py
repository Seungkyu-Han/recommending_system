import heapq
from collections import defaultdict
from email.policy import default

from surprise import KNNBasic

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

model = KNNBasic(sim_options=sim_options)
model.fit(train_set)

sims_matrix = model.compute_similarities()

test_user_inner_id = train_set.to_inner_uid(test_subject)
similarity_row = sims_matrix[test_user_inner_id]

similar_users = []

for inner_id, score in enumerate(similarity_row):
    if inner_id != test_user_inner_id:
        similar_users.append((inner_id, score))

k_neighbors = heapq.nlargest(k, similar_users, key=lambda x: x[1])

candidates = defaultdict(float)

for similar_user in k_neighbors:
    inner_id = similar_user[0]
    user_similarity_score = similar_user[1]
    ratings = train_set.ur[inner_id]
    for rating in ratings:
        candidates[rating[0]] += (rating[1] / 5.0) * user_similarity_score

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