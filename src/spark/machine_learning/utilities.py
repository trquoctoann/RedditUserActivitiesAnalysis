import nltk
from emot.emo_unicode import UNICODE_EMOJI
from emot.emo_unicode import EMOTICONS_EMO
from nltk.stem import WordNetLemmatizer
from pyspark.sql.types import StringType, ArrayType
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.functions import col, udf

nltk.download('wordnet')
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