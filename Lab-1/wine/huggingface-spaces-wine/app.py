import gradio as gr
from PIL import Image
import requests
import hopsworks
import joblib
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("wine_model", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/wine_model.pkl")
print("Model downloaded")

def wine(volatile_acidity, citric_acid, free_sulfur_dioxide, total_sulfur_dioxide, alcohol):
    print("Calling function")
#     df = pd.DataFrame([[sepal_length],[sepal_width],[petal_length],[petal_width]], 
    df = pd.DataFrame([[volatile_acidity, citric_acid, free_sulfur_dioxide, total_sulfur_dioxide, alcohol]], 
                      columns=['volatile_acidity', 'citric_acid', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'alcohol'])
    
    print("Predicting")
    print(df)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(df) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
#     print("Res: {0}").format(res)
    print(res)
    wine_url = "https://raw.githubusercontent.com/featurestoreorg/serverless-ml-course/main/src/01-module/assets/" + res[0] + ".png"
    img = Image.open(requests.get(wine_url, stream=True).raw)            
    return img
        
demo = gr.Interface(
    fn=wine,
    title="Wine Quality Predictive Analytics",
    description="Experiment with different features to predict which quality the wine is.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=0.34, label="Volatile acidity"),
        gr.inputs.Number(default=0.32, label="Citric acid"),
        gr.inputs.Number(default=30, label="Free sulfur dioxide"),
        gr.inputs.Number(default=115, label="Total sulfur dioxide"),
        gr.inputs.Number(default=10.5, label="Alcohol"),
        ],
    outputs=gr.Image(type="pil"))

demo.launch(debug=True)

