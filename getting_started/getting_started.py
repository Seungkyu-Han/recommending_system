from movie_lens import MovieLens
from surprise import SVD


def build_anti_test_set_for_user(test_subject, trainset):
    fill = trainset.global_mean

    anti_test_set = []

    u = trainset.to_inner_uid(str(test_subject))

    user_items = set([j for (j, _) in trainset.ur[u]])

    anti_test_set += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                      i in trainset.all_items() if
                      i not in user_items]
    return anti_test_set

test_subject = 85

movie_lens = MovieLens()

print("loading movie ratings...")

data = movie_lens.load_movie_lens_latest_small()

user_ratings = movie_lens.get_user_ratings(test_subject)
loved = []
hated = []

for ratings in user_ratings:
    if float(ratings[1]) > 4.0:
        loved.append(ratings[0])
    elif float(ratings[1]) < 3.0:
        hated.append(ratings[0])

print(f"user {test_subject} loved these movies")

for movie in loved:
    print(movie_lens.get_movie_name(movie))

print(f"user {test_subject} hated these movies")
for movie in hated:
    print(movie_lens.get_movie_name(movie))

print("\nBuilding recommendation model...")
trainSet = data.build_full_trainset()

algo = SVD()
algo.fit(trainSet)

print("Computing recommendations...")
testSet = build_anti_test_set_for_user(test_subject, trainSet)
predictions = algo.test(testSet)

recommendations = []

print ("\nWe recommend:")
for userID, movieID, actualRating, estimatedRating, _ in predictions:
    intMovieID = int(movieID)
    recommendations.append((intMovieID, estimatedRating))

recommendations.sort(key=lambda x: x[1], reverse=True)

for ratings in recommendations[:10]:
    print(movie_lens.get_movie_name(ratings[0]))


