import logging
import numpy as np
from utilities import preprocessing
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import CountVectorizer
from pyspark.sql.functions import col, udf, rand
from pyspark.ml.clustering import LDA
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.classification import LogisticRegression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextClassificationEngine:
    
    def __init__(self, spark):
        logger.info("Starting up text classification engine: ")
        self.spark = spark
        self.text_classification_data = self.__load_data_from_database()
        self.preprocessed_data = self.__data_preprocessing()
        self.hashing_tf, self.idf_vectorizer, self.rescaled_data = self.__vectorize_data()
        self.model = self.__train_model() 
    
    def __load_data_from_database(self) :
        logger.info("Loading labled data...")
        text_classification_data = self.spark.read \
                                            .format("jdbc") \
                                            .option("driver","com.mysql.cj.jdbc.Driver") \
                                            .option("url", "jdbc:mysql://web-database/Web") \
                                            .option("dbtable", "textClassification") \
                                            .option("user", "root") \
                                            .option("password", "123") \
                                            .load()
        text_classification_data = text_classification_data.select('category', 'descriptions')
        text_classification_data = text_classification_data.dropna(subset = ('category'))
        logger.info("Loading completed")
        return text_classification_data

    def __data_preprocessing(self):
        logger.info("Preprocessing data...")
        preprocessed_data = preprocessing(self.text_classification_data, 'descriptions')
        logger.info("Preprocessing completed")
        return preprocessed_data
    
    def __vectorize_data(self):
        logger.info("Vectorize data...")
        hashing_tf = HashingTF(inputCol = "filtered", outputCol = "raw_features")
        featurized_data = hashing_tf.transform(self.preprocessed_data)

        idf = IDF(inputCol = "raw_features", outputCol = "features", minDocFreq = 100)
        idf_vectorizer = idf.fit(featurized_data)
        rescaled_data = idf_vectorizer.transform(featurized_data)
        logger.info("Vectorization completed")
        return hashing_tf, idf_vectorizer, rescaled_data

    def __train_model(self):
        labelEncoder = StringIndexer(inputCol = 'category',outputCol = 'label').fit(self.rescaled_data)
        df = labelEncoder.transform(self.rescaled_data)
        
        regParam = 0.1
        elasticNetParam = 0
        
        logger.info("Training text classification model...")
        lr = LogisticRegression(featuresCol = 'features',
                                labelCol = 'label',
                                regParam = regParam,
                                elasticNetParam = regParam)
        model = lr.fit(df)
        logger.info("Text classification model built!")
        return model
    
    def predict_label(self, input_data):
        schema = StructType([StructField("post_id", StringType(), True)\
                            ,StructField("descriptions", StringType(), True)])
        input_df = self.spark.createDataFrame(data = input_data, schema = schema)
        input_df = preprocessing(input_df, 'descriptions')
        
        featurized_input_df = self.hashing_tf.transform(input_df)
        rescaled_input_df = self.idf_vectorizer.transform(featurized_input_df) 
        predictions = self.model.transform(rescaled_input_df)
        def get_label(label): 
            label_dict = {0.0: 'Business',
                         1.0: 'Sci/Tech',
                         2.0: 'Sports',
                         3.0: 'World'}
            return label_dict[label]
        get_label_udf = udf(get_label, StringType())
        predictions = predictions.withColumn('label_name', get_label_udf(col('prediction')))
        return predictions.select('post_id', 'descriptions', 'label_name')

    
class TopicModellingModel:
    
    def __init__(self, spark, label_name, k):
        logger.info("Starting up model LDA Business: ")
        self.spark = spark
        self.label_name = label_name
        self.k = k
        self.data = self.__load_data_from_database()
        self.preprocessed_data = self.__data_preprocessing()
        self.vectorizer, self.wordVectors = self.__vectorize_data()
        self.model, self.final_df = self.__train_model() 
    
    def __load_data_from_database(self) :
        logger.info("Loading data...")
        data = self.spark.read \
                    .format("jdbc") \
                    .option("driver","com.mysql.cj.jdbc.Driver") \
                    .option("url", "jdbc:mysql://web-database/Web") \
                    .option("dbtable", "redditData") \
                    .option("user", "root") \
                    .option("password", "123") \
                    .load() \
                    .filter(col('category') == self.label_name)
        logger.info("Loading completed")
        return data

    def __data_preprocessing(self):
        logger.info("Preprocessing data...")
        preprocessed_data = self.data.select('id', 'category', 'descriptions')
        preprocessed_data = preprocessed_data.dropna(subset = ('category'))
        preprocessed_data = preprocessing(preprocessed_data, 'descriptions')
        logger.info("Preprocessing completed")
        return preprocessed_data
    
    def __vectorize_data(self):
        vectorizer = CountVectorizer().setInputCol("filtered").setOutputCol("features").fit(self.preprocessed_data)
        wordVectors_business = vectorizer.transform(self.preprocessed_data)
        return vectorizer, wordVectors_business

    def __train_model(self):
        maxIter = 100
        seed = 2
        lda = LDA(k = self.k, maxIter = maxIter, featuresCol = 'features', seed = seed)
        ldaModel = lda.fit(self.wordVectors)
        final_df = ldaModel.transform(self.wordVectors)

        to_array = udf(lambda v: v.toArray().tolist(), ArrayType(FloatType()))
        max_index = udf(lambda x: x.index(max(x)) if x is not None else None, IntegerType())
        final_df = final_df.withColumn('topicDistribution', to_array(final_df['topicDistribution']))
        final_df = final_df.withColumn('topic', max_index(final_df['topicDistribution']))
        logger.info("LDA Business model built!")
        return ldaModel, final_df
    
    def predict_topic(self, input_data):
        input_df = preprocessing(input_data, 'descriptions')
        input_wordVectors = self.vectorizer.transform(input_df)
        predictions = self.model.transform(input_wordVectors)
        
        to_array = udf(lambda v: v.toArray().tolist(), ArrayType(FloatType()))
        max_index = udf(lambda x: x.index(max(x)) if x is not None else None, IntegerType())
        predictions = predictions.withColumn('topicDistribution', to_array(predictions['topicDistribution']))
        predictions = predictions.withColumn('topic', max_index(predictions['topicDistribution']))
        return predictions.select('post_id', 'descriptions', 'label_name', 'topic')
    
    def get_recommendation(self, topic):
        relevant_posts = self.final_df.filter(col('topic') == topic)
        relevant_posts = relevant_posts.orderBy(rand()).limit(5).select('id', 'topic')
        recommendation = relevant_posts.join(self.data, relevant_posts.id == self.data.id, "inner").drop(relevant_posts.id)
        return recommendation