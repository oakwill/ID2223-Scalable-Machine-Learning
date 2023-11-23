import os
import modal

LOCAL=False

if LOCAL == False:
   stub = modal.Stub("wine_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("erland-hopsworks-ai"))
   def f():
       g()


def generate_wine(quality_value, volatile_acidity_max, volatile_acidity_min, citric_acid_max, citric_acid_min, 
                    free_sulfur_dioxide_max, free_sulfur_dioxide_min, total_sulfur_dioxide_max, total_sulfur_dioxide_min,
                    alcohol_max, alcohol_min):
    """
    Returns a single iris flower as a single row in a DataFrame
    """
    import pandas as pd
    import random

    df = pd.DataFrame({ "volatile_acidity": [random.uniform(volatile_acidity_max, volatile_acidity_min)],
                       "citric_acid": [random.uniform(citric_acid_max, citric_acid_min)],
                       "free_sulfur_dioxide": [random.uniform(free_sulfur_dioxide_max, free_sulfur_dioxide_min)],
                       "total_sulfur_dioxide": [random.uniform(total_sulfur_dioxide_max, total_sulfur_dioxide_min)],
                       "alcohol": [random.uniform(alcohol_max, alcohol_min)]
                      })
    df['quality'] = quality_value
    return df


def get_random_wine_quality():
    """
    Returns a DataFrame containing one random iris flower
    """
    import pandas as pd
    import random

    three_df = generate_wine(3, 0.98, 0.23, 0.46, 0.02, 120, 5, 235, 14, 11.5, 8.95)
    four_df =  generate_wine(4, 0.78, 0.23, 0.49, 0.03, 41, 5, 189, 22, 11.5, 9.00)
    five_df =  generate_wine(5, 0.65, 0.21, 0.50, 0.10, 57, 8, 197, 30, 11.0, 9.00)
    six_df =   generate_wine(6, 0.53, 0.17, 0.49, 0.17, 54, 10, 186, 31, 12.2, 9.20)
    seven_df = generate_wine(7, 0.44, 0.16, 0.49, 0.24, 49, 11, 166, 28, 12.8, 9.60)
    eight_df = generate_wine(8, 0.45, 0.16, 0.46, 0.25, 53, 15, 169, 73, 13.0, 9.80)
    nine_df =  generate_wine(9, 0.36, 0.25, 0.47, 0.31, 47, 25, 133, 96, 12.8, 11.20)

    # randomly pick one of these 7 and write it to the featurestore
    pick_random = random.uniform(0,7)
    if pick_random >= 6:
        wine_df = three_df
        print("Quality 3 added")
    elif pick_random >= 5:
        wine_df = four_df
        print("Quality 4 added")
    elif pick_random >= 4:
        wine_df = five_df
        print("Quality 5 added")
    elif pick_random >= 3:
        wine_df = six_df
        print("Quality 6 added")
    elif pick_random >= 2:
        wine_df = seven_df
        print("Quality 7 added")
    elif pick_random >= 1:
        wine_df = eight_df
        print("Quality 8 added")
    else:
        wine_df = nine_df
        print("Quality 9 added")

    return wine_df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    wine_df = get_random_wine_quality()

    wine_fg = fs.get_feature_group(name="wine_updated3",version=1)
    wine_fg.insert(wine_df)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("wine_daily")
        with stub.run():
            f()
