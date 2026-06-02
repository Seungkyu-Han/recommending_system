from surprise import KNNBaseline, SVD
from surprise.model_selection import train_test_split, LeaveOneOut

from evaluating.recommender_metrics import user_coverage, diversity
from movie_lens import MovieLens
from recommender_metrics import rmse, mae, get_top_n, hit_rate, rating_hit_rate, cumulative_hit_rate, average_reciprocal_hit_rank

movie_lens = MovieLens()

print("loading movie ratings...")
data = movie_lens.load_movie_lens_latest_small()

print("성공적으로 로드되었습니다:", data)

print(f'Computing movie popularity ranks so we can measure novelty later...')
rankings = movie_lens.get_popularity_rankings()


print(f'Computing item similarities so we can measure diversity later...')
full_train_set = data.build_full_trainset()
sim_options = {'name': 'pearson_baseline', 'user_based': False}
sims_algo = KNNBaseline(sim_options=sim_options)
sims_algo.fit(full_train_set)

print(f'Building recommendation model...')
train_set, test_set = train_test_split(data, test_size=0.25, random_state=1)

algo = SVD(random_state=10)
algo.fit(train_set)

print(f'Computing recommendations...')
predictions = algo.test(test_set)

print("\nEvaluating accuracy of model...")
print("RMSE: ", rmse(predictions))
print("MAE: ", mae(predictions))

print(f'Evaluating accuracy of model...')

loocv = LeaveOneOut(n_splits=1, random_state=1)

for train_set, test_set in loocv.split(data):
    print("computing recommendations with leave-one-out cross validation...")

    algo.fit(train_set)

    print("predict ratings for left-out set...")
    left_out_predictions = algo.test(test_set)

    print("predict all missing ratings...")
    big_test_set = train_set.build_anti_testset()
    all_predictions = algo.test(big_test_set)

    print("compute top 10 recs per user...")
    top_n_predicted = get_top_n(all_predictions, 10)

    print("Hit rate: ", hit_rate(top_n_predicted, left_out_predictions))

    print('rHR (Hit rate by rating value)')
    rating_hit_rate(top_n_predicted, left_out_predictions)

    print(f'cHR (cumulative Hit rate, rating >= 4.0)', cumulative_hit_rate(top_n_predicted, left_out_predictions))

    print(f'ARHR (Average Reciprocal Hit Rank)', average_reciprocal_hit_rank(top_n_predicted, left_out_predictions))

print("Computing complete recommendations...")
algo.fit(full_train_set)
big_test_set = full_train_set.build_anti_testset()
all_predictions = algo.test(big_test_set)
top_n_predicted = get_top_n(all_predictions, 10)

print(f'user coverage: {user_coverage(top_n_predicted, full_train_set.n_users)}')

print(f'diversity: {diversity(top_n_predicted, sims_algo)}')