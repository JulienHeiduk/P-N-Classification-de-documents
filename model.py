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
from river.metrics import ClassificationReport
from sklearn.model_selection import train_test_split
from river.naive_bayes import MultinomialNB as MNB_RIVER


class init_model:
    
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
        
        df_labels = pd.DataFrame(data_to_transform['label'].value_counts() > 1)
        df_labels.reset_index(drop = False, inplace = True)
        list_to_keep = list(df_labels[df_labels['label'] == True]['index'])

        data_to_transform = data_to_transform[data_to_transform['label'].isin(list_to_keep)]
        
        return data_to_transform
    
    def get_data(self):
        df = self.data
        df_all = pd.DataFrame()

        for i in df.columns:
            df_class = df[i][0]
            df_class = pd.DataFrame(df_class)
            df_class['label'] = i
            df_all = pd.concat([df_all, df_class])
            
        return df_all
        
    def clean_data(self, input_data: pd.DataFrame):    
        df_work = input_data[["title", "texte", "label"]]  
        df_work = self.transform_data(data_to_transform = df_work, column = "texte")
        df_work = self.transform_data(data_to_transform = df_work, column = "title")
        
        df_work["Title+Texte"] = df_work["title_clean"] + " " + df_work["texte_clean"]
        df_work = df_work.astype(np.str)        
        
        return df_work
    
    def model_train(self, data_cleaned: pd.DataFrame, performance: str):
        
        if performance == "False":
            r = redis.Redis(host='localhost', port=6379, db=0)  
            
            try:
                model = pickle.loads(r.get('model'))
            except:
                print("Pas de modèles dans redis - Initialisation d'un modèle")
                pass

            # Nécéssite au moins 2 classes différentes si pas de modèles
            X = data_cleaned
            y = data_cleaned.pop('label')

            data_stream_texte_train = stream.iter_pandas(X, y)

            pipe_nb = Pipeline(('vectorizer', river.feature_extraction.BagOfWords(on="Title+Texte",strip_accents=False, lowercase=False)),
                               ('nb', MNB_RIVER()))        

            for text,label in data_stream_texte_train:
                pipe_nb = pipe_nb.learn_one(text,label)        

            # save the model to disk     
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.set('model', pickle.dumps(pipe_nb))

        if performance == "True":
            
            X = data_cleaned
            y = data_cleaned.pop('label')

            # split the dataset into test set and train set
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = 0.2, random_state = 23)  

            data_stream_texte_train = stream.iter_pandas(X_train, y_train)
            data_stream_texte_test = stream.iter_pandas(X_test, y_test)

            pipe_nb = Pipeline(('vectorizer', river.feature_extraction.BagOfWords(on="Title+Texte",strip_accents=False, lowercase=False)),
                               ('nb', MNB_RIVER()))        

            for text,label in data_stream_texte_train:
                pipe_nb = pipe_nb.learn_one(text,label)        

            y_pred = []
            for x,y in data_stream_texte_test:
                res = pipe_nb.predict_one(x)
                y_pred.append(res)

            #Classification report
            report = ClassificationReport()

            for yt,yp in zip(y_test,y_pred):
                report = report.update(yt,yp)

            with open('./performance.log', 'w') as f:
                print(report, file=f)

            # save the model to disk     
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.set('model', pickle.dumps(pipe_nb))