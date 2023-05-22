from pyspark.mllib.recommendation import ALS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommendationEngine:
    
    def __init__(self, spark):
        logger.info("Starting up the Recommendation Engine: ")
        self.spark = spark
        self.movies_data, self.ratings_data, self.users_data = self.__load_data_from_database()
        self.number_of_movie_ratings, self.movie_ID_with_avg_ratings = self.__count_and_average_ratings()
        self.model = self.__train_model() 
    
    def __load_data_from_database(self) :
        logger.info("Loading Movies data...")
        movies_data = self.spark.read.format("jdbc").option("driver","com.mysql.cj.jdbc.Driver") \
                                .option("url", "jdbc:mysql://cap2-database/MovieLens") \
                                .option("dbtable", "movies").option("user", "root") \
                                .option("password", "123").load().rdd.cache()
        
        logger.info("Loading Ratings data...")
        ratings_data = self.spark.read.format("jdbc").option("driver","com.mysql.cj.jdbc.Driver") \
                                 .option("url", "jdbc:mysql://cap2-database/MovieLens") \
                                 .option("dbtable", "ratings").option("user", "root").option("password", "123") \
                                 .option('fetchSize', '10000').option('partitionColumn', 'ratingTime')\
                                 .option('lowerBound', '1995-01-09 11:46:44').option('upperBound', '2018-09-26 06:59:09')\
                                 .option('numPartitions', '23').load().rdd
        ratings_data = ratings_data.map(lambda x: (x[1], x[2], x[3])).cache()
        
        logger.info("Loading Users data...")
        users_data = ratings_data.map(lambda x: (x[0])).distinct().cache()
        return movies_data, ratings_data, users_data
    
    def __count_and_average_ratings(self):
        logger.info("Counting how many ratings per movie...")
        def get_counts_and_averages(ID_and_ratings_tuple):
            nratings = len(ID_and_ratings_tuple[1])
            return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1]))/nratings)

        movie_ID_with_ratings = self.ratings_data.map(lambda x: (x[1], x[2])).groupByKey()
        movie_ID_with_avg_ratings = movie_ID_with_ratings.map(get_counts_and_averages)
        number_of_movie_ratings = movie_ID_with_avg_ratings.map(lambda x: (x[0], x[1][0]))
        return number_of_movie_ratings, movie_ID_with_avg_ratings

    def __train_model(self):
        rank = 12
        seed = 1
        iterations = 10
        regularization_parameter = 0.1
        logger.info("Training the ALS model...")
        model = ALS.train(self.ratings_data, rank = rank, seed = seed, iterations = iterations, lambda_ = regularization_parameter)
        logger.info("ALS model built!")
        return model

    def __predict_ratings(self, user_and_movie):
        predicted = self.model.predictAll(user_and_movie)
        predicted_rating = predicted.map(lambda x: (x.product, x.rating))
        predicted_rating_title_and_count = predicted_rating.join(self.number_of_movie_ratings)
        predicted_rating_title_and_count = predicted_rating_title_and_count.map(lambda r: (r[0], r[1][0], r[1][1]))
        return predicted_rating_title_and_count
    
    def get_top_ratings(self, user_id, movies_count = 5):
        user_ratings_ids = self.ratings_data.filter(lambda rating: rating[0] == user_id).map(lambda x: x[1]).collect()
        if user_ratings_ids == [] : 
            new_user_title_and_count = self.movie_ID_with_avg_ratings
            new_user_recommendation = new_user_title_and_count.map(lambda r: (r[0], r[1][1], r[1][0]))
            top_ratings_recommendation = new_user_recommendation.filter(lambda r: r[2]>=50000)
            return top_ratings_recommendation.map(lambda x: (x[0], round(x[1], 1), x[2])).takeOrdered(movies_count, key=lambda x: (-x[1], -x[2]))
        user_unrated_movies = self.movies_data.filter(lambda r: r[0] not in user_ratings_ids).map(lambda x: (user_id, x[0]))
        ratings = self.__predict_ratings(user_unrated_movies).filter(lambda r: r[2]>=20000)
        recommendation = ratings.map(lambda x: (x[0], round(x[1], 1), x[2])).takeOrdered(movies_count, key=lambda x: (-x[1], -x[2]))
        return recommendation