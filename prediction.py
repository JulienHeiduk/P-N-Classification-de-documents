import nltk
import river
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
    
    def __init__(self, json: str):
        self.data = pd.read_json(json, lines=True)
        
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
    
    def get_data(self):
        df = self.data
        df_all = pd.DataFrame()

        for i in df.columns:
            df_class = df[i][0]
            df_class = pd.DataFrame(df_class)
            df_class['label'] = i
            df_all = pd.concat([df_all, df_class])
            df_class.drop(['label'], inplace = True, axis = 1)
            
        return df_all    
    
    def clean_data(self, input_data: pd.DataFrame):    
        df_work = input_data[["title", "texte"]]  
        df_work = self.transform_data(data_to_transform = df_work, column = "texte")
        df_work = self.transform_data(data_to_transform = df_work, column = "title")
        
        df_work["Title+Texte"] = df_work["title_clean"] + " " + df_work["texte_clean"]
        df_work = df_work.astype(np.str)        
        
        return df_work    
    
    def predict(self, data_cleaned: pd.DataFrame):
        model = pickle.load(open("./model.sav", 'rb'))
            
        return model.predict_proba_one({'Title+Texte': data_cleaned})
