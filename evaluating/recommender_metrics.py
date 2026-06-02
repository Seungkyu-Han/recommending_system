from collections import defaultdict

import itertools
from surprise import accuracy


def mae(predictions):
    return accuracy.mae(predictions, verbose=False)

def rmse(predictions):
    return accuracy.rmse(predictions, verbose=False)

def get_top_n(predictions, n = 10, minimum_rating=3.0):

    top_n = defaultdict(list)

    for user_id, movie_id, actual_rating, est, _ in predictions:
        if est >= minimum_rating:
            top_n[int(user_id)].append((int(movie_id), est))


    for user_id, ratings in top_n.items():
        ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[user_id] = ratings[:n]

    return top_n

def hit_rate(top_n_predicted, left_out_predictions):
    hits = 0
    total = 0

    for left_out in left_out_predictions:
        user_id = left_out[0]
        left_out_movie_id = left_out[1]

        hit = False

        for movie_id, rating in top_n_predicted[int(user_id)]:
            if int(left_out_movie_id) == int(movie_id):
                hit = True
                break

        if hit:
            hits += 1

    if total == 0: return 0

    return hits / total

def rating_hit_rate(top_n_predicted, left_out_predictions):
    hits = defaultdict(float)
    total = defaultdict(float)

    for user_id, left_out_movie_id, actual_rating, est, _ in left_out_predictions:
        hit = False

        for movie_id, predicted_rating in top_n_predicted[int(user_id)]:
            if int(left_out_movie_id) == int(movie_id):
                hit = True
                break

        if hit:
            hits[actual_rating] += 1


        total[actual_rating] += 1

    for rating in sorted(hits.keys()):
        print(f"{rating} : {hits[rating] / total[rating]}")

def cumulative_hit_rate(top_n_predicted, left_out_predictions, rating_cutoff=0):
    hits = 0
    total = 0

    for user_id, left_out_movie_id, actual_rating, estimated_rating, _ in left_out_predictions:
        if actual_rating >= rating_cutoff:
            hit = False

            for movie_id, predicted_rating in top_n_predicted[int(user_id)]:
                if int(left_out_movie_id) == int(movie_id):
                    hit = True
                    break

            if hit:
                hits += 1

            total += 1

    return hits / total


def average_reciprocal_hit_rank(top_n_predicted, left_out_predictions):
    summation = 0
    total = 0

    for user_id, left_out_movie_id, actual_rating, estimated_rating, _ in left_out_predictions:
        hit_rank = 0
        rank = 0

        for movie_id, predicted_rating in top_n_predicted[int(user_id)]:
            rank += 1
            if int(left_out_movie_id) == int(movie_id):
                hit_rank = rank
                break

        if hit_rank > 0:
            summation += 1 / hit_rank

        total += 1

    return summation / total


def user_coverage(top_n_predicted, num_users):
    hits = 0

    for user_id in top_n_predicted.keys():
        hit = False
        for movie_id, rating in top_n_predicted[user_id]:
            if rating >= 0.0:
                hit = True
                break
        if hit:
            hits += 1

    return hits / num_users

def diversity(top_n_predicted, sims_algo):
    n = 0
    total = 0

    sims_matrix = sims_algo.compute_similarities()

    for user_id in top_n_predicted.keys():
        pairs = itertools.combinations(top_n_predicted[user_id], 2)

        for pair in pairs:

            movie_id_1 = pair[0][0]
            movie_id_2 = pair[1][0]

            inner_id1 = sims_algo.trainset.to_inner_iid(str(movie_id_1))
            inner_id2 = sims_algo.trainset.to_inner_iid(str(movie_id_2))

            similarity = sims_matrix[inner_id1][inner_id2]

            total += similarity

            n += 1

    return 1 - (total / n)

