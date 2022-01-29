import nltk
import river
import redis
import pickle
import warnings
import numpy as np
import pandas as pd
import texthero as hero
from river import stream
from nltk import word_tokenize
from nltk.corpus import stopwords
from river.compose import Pipeline
from river import feature_extraction
from stop_words import get_stop_words
from river.naive_bayes import MultinomialNB as MNB_RIVER

class model_predictions:

    def __init__(self):
        pass
        #self.data = pd.read_json(json, lines=True)

    @staticmethod
    def transform_data(data_to_transform: pd.DataFrame, column: str) -> pd.DataFrame:

        default_stopwords = hero.stopwords.DEFAULT
        custom_stopwords = default_stopwords.union(get_stop_words('french'))
        custom_stopwords = custom_stopwords.union(stopwords.words('french'))

        w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
        lemmatizer = nltk.stem.WordNetLemmatizer()

        data_to_transform = data_to_transform.astype(np.str)
        data_to_transform[column + '_clean'] = [word_tokenize(row) for row in data_to_transform[column]]
        data_to_transform[column + '_clean'] = data_to_transform[column + '_clean'].apply(lambda x: ' '.join(x))
        data_to_transform[column + '_clean'] = hero.lowercase(data_to_transform[column + '_clean'])
        data_to_transform[column + '_clean'] = hero.remove_punctuation(data_to_transform[column + '_clean'])
        data_to_transform[column + '_clean'] = hero.remove_whitespace(data_to_transform[column + '_clean'])
        data_to_transform[column + '_clean'] = hero.remove_stopwords(data_to_transform[column + '_clean'],custom_stopwords)

        return data_to_transform

    def clean_data(self, input_data: pd.DataFrame):
        df_work = input_data[["texte"]]
        df_work = self.transform_data(data_to_transform = df_work, column = "texte")

        df_work["Texte"] = df_work["texte_clean"]
        df_work = df_work.astype(np.str)

        return df_work

    def predict(self, data_cleaned: pd.DataFrame):
        r = redis.Redis(host='localhost', port=6379, db=0)
        model = pickle.loads(r.get('model'))

        return model.predict_proba_one({'Texte': data_cleaned})