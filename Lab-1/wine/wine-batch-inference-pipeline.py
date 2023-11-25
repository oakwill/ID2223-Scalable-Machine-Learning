import os
import modal

LOCAL=False

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","scikit-learn==1.1.1","dataframe-image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("erland-hopsworks-ai"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests
    import matplotlib.pyplot as plt
    import numpy as np
    import time

    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("wine_model", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")
    
    feature_view = fs.get_feature_view(name="wine_updated3", version=1)
    batch_data = feature_view.get_batch_data()
    y_pred = model.predict(batch_data)
    loop = True
    offset = 157
    while loop:
        wine = int(y_pred[y_pred.size-offset])
        wine_url = "https://raw.githubusercontent.com/oakwill/ID2223-Scalable-Machine-Learning/master/Lab-1/wine/wine_pictures/" + str(wine) + ".png"

        print("Wine quality predicted: " + str(wine))
        img = Image.open(requests.get(wine_url, stream=True).raw)    
        img.save("./latest_wine.png")
        dataset_api = project.get_dataset_api()    
        dataset_api.upload("./latest_wine.png", "Resources/images", overwrite=True)
    
        wine_fg = fs.get_feature_group(name="wine_updated3", version=1)
        df = wine_fg.read() 
        label = int(df.iloc[-offset]["quality"])
        label_url = "https://raw.githubusercontent.com/oakwill/ID2223-Scalable-Machine-Learning/master/Lab-1/wine/wine_pictures/" + str(label) + ".png"
        print("Wine actual: " + str(label))
        img = Image.open(requests.get(label_url, stream=True).raw)           
        img.save("./actual_wine.png")
        dataset_api.upload("./actual_wine.png", "Resources/images", overwrite=True)
        
        monitor_fg = fs.get_or_create_feature_group(name="wine_predictions",
                                                    version=1,
                                                    primary_key=["datetime"],
                                                    description="Wine Quality Prediction/Outcome Monitoring"
                                                    )
    
        now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        data = {
            'prediction': [wine],
            'label': [label],
            'datetime': [now],
        }
        monitor_df = pd.DataFrame(data)
        monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
        
        history_df = monitor_fg.read()
        # Add our prediction to the history, as the history_df won't have it - 
        # the insertion was done asynchronously, so it will take ~1 min to land on App
        history_df = pd.concat([history_df, monitor_df])

        df_recent = history_df.tail(4)
        dfi.export(df_recent, './df_recent.png', table_conversion = 'matplotlib')
        dataset_api.upload("./df_recent.png", "Resources/images", overwrite=True)
        
        predictions = history_df[['prediction']]
        labels = history_df[['label']]

        # Only create the histogram when our wine_predictions feature group has examples of all 7 wine qualities
        print("Number of different wine predictions to date: " + str(predictions.value_counts().count()))
        if predictions.value_counts().count() == 5:
            prediction_series = predictions['prediction'].to_numpy().flatten()
            #prediction_series = pd.Series(predictions.flatten(), index=labels.index)
            df_errors = pd.DataFrame({'True': labels.squeeze(), 'Predicted': prediction_series})
            df_errors['Absolute_Error'] = abs(df_errors['True'] - df_errors['Predicted'])
            grouped_errors = df_errors.groupby('True')['Absolute_Error'].mean().reset_index()

            plt.figure(figsize=(10, 6))
            plt.bar(grouped_errors['True'], grouped_errors['Absolute_Error'], color='skyblue')
            plt.xlabel('Quality')
            plt.ylabel('Mean Absolute Error')
            plt.title('Mean Absolute Error for Each Quality Group')
            plt.savefig("./mae_per_quality.png")  
            
            dataset_api.upload("./mae_per_quality.png", "Resources/images", overwrite=True)
            loop = False
        offset += 1
        time.sleep(20)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()

'''img_new_height = 300
img_width, img_height = img.size
img_aspect_ratio = img_width/img_height
img_new_width = int(img_new_height * img_aspect_ratio)
img = img.resize((img_new_width, img_new_height))'''  
