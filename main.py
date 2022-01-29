import sys
import pandas as pd
import prediction as pred
from pymongo import MongoClient
from bson.objectid import ObjectId

print("Mode: ", sys.argv[1])

if sys.argv[1] == "model" and sys.argv[2] == "True":
    import model as ml
    from pymongo import MongoClient
    from bson.objectid import ObjectId
    import pandas as pd
    import numpy as np

    data = ml.init_model(performance = sys.argv[2])
    #df_all = data.get_data()
    # Paramètres pour la connexion à la base de donnée Mongo
    client = MongoClient('mongodb://152.228.161.49:27017/')
    database = client['document']
    text_parts_collection = database.get_collection("docs")
    tps = text_parts_collection.find({}, {'_id': 1, 'typology_ids':1})

    df = pd.DataFrame(columns = ['id','label'])
    cursor = list(tps)

    df = pd.DataFrame(cursor)
    df.columns = ['id', 'label']

    df.dropna(inplace = True)

    database = client['document']
    text_parts_collection = database.get_collection("text_parts")
    tps = text_parts_collection.find({} ,{'_id': 1,'doc_id': 1, 'text': 1})
    cursor = list(tps)

    df_text = pd.DataFrame(columns = ['id', 'doc_id' ,'text'])
    df_text = pd.DataFrame(cursor)
    df_text = df_text[['doc_id', 'text']]
    df_text.columns = ['id', 'texte']

    df_text = df_text.groupby('id').texte.apply(' '.join)
    df_text.dropna(inplace = True)

    df_all = df.merge(df_text, left_on='id', right_on = 'id', how = 'inner')

    # Random sur les labels car vide lors du test
    df_all['label'] = np.random.randint(1, 3, df_all.shape[0])

    data_cleaned = data.clean_data(df_all)

    data.model_train(data_cleaned, "True")

if sys.argv[1] == "model" and sys.argv[2] == "False":
    import model as ml

    data = ml.init_model(sys.argv[2])
    # Paramètres pour la connexion à la base de donnée Mongo
    client = MongoClient('mongodb://152.228.161.49:27017/')
    database = client['document']
    text_parts_collection = database.get_collection("text_parts")
    tps = text_parts_collection.find({'doc_id': ObjectId(sys.argv[3])}, {'text': 1})

    text = ''

    for i in tps:
            text = text.join(i['text'])

    df_all = pd.DataFrame([text])
    df_all.columns = ['texte']

    label_parts_collection = database.get_collection("docs")
    lbs = label_parts_collection.find({'_id': ObjectId(sys.argv[3])})

    label = ''

    for i in lbs:
            label = i['labels']

    df_all['label'] = label[0]
    df_all['texte'] = df_all['texte'].astype(str)
    data_cleaned = data.clean_data(df_all)

    data.model_train(data_cleaned, "False")

if sys.argv[1] == "prediction":
    data_topredict = pred.model_predictions()

    # Paramètres pour la connexion à la base de donnée Mongo
    client = MongoClient('mongodb://152.228.161.49:27017/')
    database = client['document']
    text_parts_collection = database.get_collection("text_parts")
    tps = text_parts_collection.find({'doc_id': ObjectId(sys.argv[3])}, {'text': 1})

    text = ''

    for i in tps:
            text = text.join(i['text'])
    df_all = pd.DataFrame([text])
    df_all.columns = ['texte']

    data_cleaned = data_topredict.clean_data(df_all)

    prediction = data_topredict.predict(data_cleaned['Texte'].iloc[0])
    print(prediction)
