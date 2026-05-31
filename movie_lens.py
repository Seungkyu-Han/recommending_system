import csv
import re
from collections import defaultdict

from surprise import Reader, Dataset


class MovieLens:

    movie_id_to_name = {}
    name_to_movie_id = {}

    ratings_path = 'ratings.csv'
    movies_path = 'movies.csv'

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

    def get_movie_name(self, movie_id: int) -> str:
        return self.movie_id_to_name[movie_id] or ''

    def get_movie_id(self, movie_name: str) -> int:
        return self.name_to_movie_id[movie_name] or 0