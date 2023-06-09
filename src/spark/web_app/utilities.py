import nltk
import requests
import configparser
import urllib3
import datetime
import mysql.connector
import os
import re
from bs4 import BeautifulSoup, SoupStrainer
from urllib.parse import urljoin, urlsplit
from emot.emo_unicode import UNICODE_EMOJI
from emot.emo_unicode import EMOTICONS_EMO
from nltk.stem import WordNetLemmatizer
from pyspark.sql.types import StringType, ArrayType
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.functions import col, udf
from pyspark.sql import functions

nltk.download('wordnet')
configFilePath = 'project.conf'
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def delete_numbers(df, column_name):
    remove_numbers_udf = udf(lambda text: ''.join([c for c in text if not c.isdigit()]), StringType())
    df = df.withColumn(column_name, remove_numbers_udf(col(column_name)))
    return df

def decode_abbreviations(df, column_name):
    standardizing_dict = {}
    with open('en-abbreviations.txt', 'r', encoding = 'utf-8') as f:
        for row in f : 
            row = row.replace('\n', '')
            row = row.replace('\t\t\t', '\t')
            row = row.replace('\t\t', '\t')
            acronym, word = row.split('\t')[0], row.split('\t')[1]
            acronym = acronym.replace(' ', '')
            word = word.replace(' ', '')
            if ',' in word: 
                word = word.split(',')[0]
            standardizing_dict[acronym] = word
    replace_abbreviations_udf = udf(lambda text: ' '.join([standardizing_dict.get(word, word) for word in text.split()]), StringType())
    df = df.withColumn(column_name, replace_abbreviations_udf(col(column_name)))
    return df

def handle_emojis(df, column_name):
    def converting_emojis(text):
        for x in EMOTICONS_EMO:
            text = text.replace(x, "_".join(EMOTICONS_EMO[x].replace(",","").replace(":","").split()))
        for x in UNICODE_EMOJI:
            text = text.replace(x, "_".join(UNICODE_EMOJI[x].replace(",","").replace(":","").split()))
        return text
    converting_emojis_udf = udf(converting_emojis, StringType())
    df = df.withColumn(column_name, converting_emojis_udf(col(column_name)))
    return df

def tokenize_text(df, column_name):
    tokenizer = RegexTokenizer(inputCol = column_name, outputCol = "descriptions_words", pattern = "\\W")
    df = tokenizer.transform(df)
    return df

def lemmatization(df):
    lemmatizer = WordNetLemmatizer()
    def lemmatize_text(row):
        row = [lemmatizer.lemmatize(word,'v') for word in row]
        return row
    lemmatize_udf = udf(lemmatize_text, ArrayType(StringType()))
    df = df.withColumn('descriptions_words', lemmatize_udf(col('descriptions_words')))
    return df

def remove_stopword(df): 
    stopwords_remover = StopWordsRemover(inputCol = "descriptions_words", outputCol = "filtered")
    df = stopwords_remover.transform(df)
    return df

def preprocessing(df, column_name):
    df = delete_numbers(df, column_name)
    df = decode_abbreviations(df, column_name)
    df = handle_emojis(df, column_name)
    df = tokenize_text(df, column_name)
    df = lemmatization(df)
    df = remove_stopword(df)
    return df

# create the authentication header to query 
def get_reddit_bearer_token():
    # get spotify api info
    parser = configparser.ConfigParser()
    parser.read(configFilePath)
    auth_url = parser.get("reddit_api_config", "reddit_auth_url")
    username = parser.get("reddit_api_config", "reddit_api_username")
    password = parser.get("reddit_api_config", "reddit_api_password")
    app_id = parser.get("reddit_api_config", "reddit_api_app_id")
    app_secret = parser.get("reddit_api_config", "reddit_api_app_secret")
    user_agent = parser.get("reddit_api_config", "reddit_api_user_agent")

    # connect to API
    payload = {'username': username, 'password': password, 'grant_type': 'password'}
    auth = requests.auth.HTTPBasicAuth(app_id, app_secret)
    response = requests.post(auth_url, data = payload, headers={'user-agent': user_agent}, auth = auth)
    if response.status_code != 200:
        print("Error Response Code: " + str(response.status_code))
        raise Exception(response.status_code, response.text)
    access_token = response.json()["access_token"]
    return "Bearer " + access_token

# send request to get new post
def send_request_reddit_get_new_post(url, access_token = get_reddit_bearer_token()):
    parser = configparser.ConfigParser()
    parser.read(configFilePath)
    user_agent = parser.get("reddit_api_config", "reddit_api_user_agent")
    parameters = {'limit' : 100}
    response = requests.get(url, headers = {'Authorization' : access_token, 'user-agent': user_agent}, params = parameters)
    if response.status_code != 200:
        print("Error Response Code: " + str(response.status_code))
        raise Exception(response.status_code, response.text)
    return response.json(), response.status_code

def get_subtopic_top_word(topic_engine): 
    vocabulary = topic_engine.vectorizer.vocabulary

    topics = topic_engine.model.describeTopics().collect()
    topics_words = []

    for topic in topics:
        topic_id = topic['topic']
        topic_words_indices = topic['termIndices']
        word_probabilities = topic['termWeights']

        topic_words = [vocabulary[idx] for idx in topic_words_indices]
        topics_words.append(topic_words)
    return topics_words
    
def get_most_popular_topic(grouped, label_topic):
    max_count = grouped.agg(functions.max("count")).first()[0]
    highest_count_groups = grouped.filter(col('count') == max_count)
    label_topic_name = highest_count_groups.select(label_topic).first()[0]
    return label_topic_name

def get_background_image(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html5lib")
    image_link = None
    for tag in soup.find_all():
        for attr in tag.attrs.values():
            if isinstance(attr, str) and re.match(r'^https?://\S+\.(?:jpg|jpeg|png|gif)(?:\?.*)?$', attr):
                image_link = attr
                break
        if image_link:
            break
    return image_link