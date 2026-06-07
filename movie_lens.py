import csv
import re
import os
from collections import defaultdict

from surprise import Reader, Dataset


class MovieLens:

    movie_id_to_name = {}
    name_to_movie_id = {}

    def __init__(
            self,
    ):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.ratings_path = os.path.join(base_dir, 'ratings.csv')
        self.movies_path = os.path.join(base_dir, 'movies.csv')

    def load_movie_lens_latest_small(self):

        self.movie_id_to_name = {}
        self.name_to_movie_id = {}

        reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)

        ratings_dataset: Dataset = Dataset.load_from_file(self.ratings_path, reader=reader)

        with open(self.movies_path, newline='', encoding='ISO-8859-1') as csv_file:
            movie_reader = csv.reader(csv_file)
            next(movie_reader)

            for row in movie_reader:
                movie_id = int(row[0])
                movie_name = row[1]
                self.movie_id_to_name[movie_id] = movie_name
                self.name_to_movie_id[movie_name] = movie_id

        return ratings_dataset

    def get_popularity_rankings(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)

        with open(self.ratings_path, newline='') as csv_file:
            ratings_reader = csv.reader(csv_file)
            next(ratings_reader)

            for row in ratings_reader:
                movie_id = int(row[1])
                ratings[movie_id] += 1

        rank = 1
        for movie_id, rating in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[movie_id] = rank
            rank += 1

        return rankings


    def get_user_ratings(self, user: int):

        user_ratings = []
        hit_user = False

        with open(self.ratings_path, newline='', encoding='ISO-8859-1') as csv_file:
            ratings_reader = csv.reader(csv_file)
            next(ratings_reader)

            for row in ratings_reader:
                user_id = int(row[0])

                if user == user_id:
                    movie_id = int(row[1])
                    rating = float(row[2])
                    user_ratings.append((movie_id, rating))
                    hit_user = True

                if hit_user and user_id != user:
                    break

        return user_ratings

    def get_genres(self):
        genres = defaultdict(list)

        genre_ids = {}

        max_genre_id = 0

        with open(self.movies_path, newline='', encoding='ISO-8859-1') as csv_file:
            movie_reader = csv.reader(csv_file)
            next(movie_reader)
            for row in movie_reader:
                movie_id = int(row[0])
                genre_list = row[2].split('|')
                genre_id_list = []

                for genre in genre_list:
                    if genre in genre_ids:
                        genre_id = genre_ids[genre]
                    else:
                        genre_id = max_genre_id
                        genre_ids[genre] = genre_id
                        max_genre_id += 1
                    genre_id_list.append(genre_id)
                genres[movie_id] = genre_id_list
        for movie_id, genre_id_list in genres.items():
            bit_field = [0] * max_genre_id
            for genre_id in genre_id_list:
                bit_field[genre_id] = 1
            genres[movie_id] = bit_field

        return genres

    def get_years(self):
        p = re.compile(r"(?:\((\d{4})\))?\s*$")

        years = defaultdict(int)

        with open(self.movies_path, newline='', encoding='ISO-8859-1') as csv_file:
            movie_reader = csv.reader(csv_file)
            next(movie_reader)
            for row in movie_reader:
                movie_id = int(row[0])
                title = row[1]
                match = p.search(title)

                if not match:
                    continue

                year = match.group(1)

                if year:
                    years[movie_id] = int(year)

        return years

    def get_mise_en_scene(self):
        mes = defaultdict(list)

        with open(os.path.join(self.base_dir, 'LLVisualFeatures13K_Log.csv'), newline='', encoding='ISO-8859-1') as csv_file:
            mes_reader = csv.reader(csv_file)
            next(mes_reader)
            for row in mes_reader:
                movie_id = int(row[0])
                avg_shot_length = float(row[1])
                mean_color_variance = float(row[2])
                stddev_color_variance = float(row[3])
                mean_motion = float(row[4])
                stddev_motion = float(row[5])
                mean_lighting_key = float(row[6])
                num_shots = float(row[7])
                mes[movie_id] = [
                    avg_shot_length,
                    mean_color_variance,
                    stddev_color_variance,
                    mean_motion,
                    stddev_motion,
                    mean_lighting_key,
                    num_shots,
                ]
        return mes


    def get_movie_name(self, movie_id: int) -> str:
        return self.movie_id_to_name[movie_id] or ''

    def get_movie_id(self, movie_name: str) -> int:
        return self.name_to_movie_id[movie_name] or 0