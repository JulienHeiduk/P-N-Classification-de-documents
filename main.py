import sys
import pandas as pd

print("Mode: ", sys.argv[1])

if sys.argv[1] == "initialisation":
    import initialisation as init
    
    data = init.init_model('./POC IA Classification 2021-10-12 093136 0000.json')
    df_all = data.get_data()
    data_cleaned = data.clean_data(df_all)
    
    data.first_model(data_cleaned)

if sys.argv[1] == "prediction":
    import prediction as pred
    
    data_topredict = pred.model_predictions('./POC IA Classification 2021-10-12 093136 0000.json')
    df_all = data_topredict.get_data()
    data_cleaned = data_topredict.clean_data(df_all)
    
    data_topredict.predict(data_cleaned['Title+Texte'].iloc[1])
    
if sys.argv[1] == "update":    
    import update as updt
    
    data = updt.update_model('./POC IA Classification 2021-10-12 093136 0000.json')
    df_all = data.get_data()
    data_cleaned = data.clean_data(df_all)
    
    data.new_model(pd.DataFrame(data_cleaned[['Title+Texte','label']].iloc[1]).T)
    
if sys.argv[1] not in ["initialisation", "prediction", "update"]:
    print("Commande non reconnue !")