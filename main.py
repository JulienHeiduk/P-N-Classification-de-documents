import sys
import pandas as pd

print("Mode: ", sys.argv[1])

if sys.argv[1] == "model" & sys.argv[2] == True:
    import model as model
    
    data = model.init_model('./POC IA Classification 2021-10-12 093136 0000.json')
    df_all = data.get_data()
    data_cleaned = data.clean_data(df_all)
    
    data.model(data_cleaned)
    
if sys.argv[1] == "model" & sys.argv[2] == False:    
    import model as model
    
    data = model.init_model('./POC IA Classification 2021-10-12 093136 0000.json')
    df_all = data.get_data()
    data_cleaned = data.clean_data(df_all)
    
    data.model(data_cleaned)    

if sys.argv[1] == "prediction":
    import prediction as pred
    
    data_topredict = pred.model_predictions('./POC IA Classification 2021-10-12 093136 0000.json')
    df_all = data_topredict.get_data()
    data_cleaned = data_topredict.clean_data(df_all)
    
    prediction = data_topredict.predict(data_cleaned['Title+Texte'].iloc[1])
    print(prediction)