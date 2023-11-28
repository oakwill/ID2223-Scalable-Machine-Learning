# ID2223 Scalable Machine Learning and Deep Learning | Lab 1 Erland Ekholm

The goal of Lab 1 is to build and deploy two serverless ML prediction systems using Hopsworks, Modal, and Hugging Face Spaces. The first system works with the Iris flowers dataset and generates a flower depending on the user inputs. The second system with the wine dataset, predicts the quality depending on a set of 11 features. Both applications include an interactive component with input generated predictions and a historical record with the trained model statistics.

## Pipelines
Since the procedure and coding of the pipelines for both iris and wine are very correlated, general descriptions will be presented for each part, in the process of creating a serverless ML system.

**wine-eda-and-backfill-featu.re-group.ipynb**

The aim of this file is to download, extract schemas from the these datasets and analyze them, and finally send the dataset as feature groups to Hopsworks. Some adjustments were made in the wine-edition where I excluded 6 features that had little predictive power, and therefore only sent 5 features to modal.

**feature-pipeline-daily.py**

The feature-daily pipeline randomly generates a synthetic flower or wine for which the application can run a prediction using the pre-trained model. The values are generated at random and added to the feature group in Hopsworks. Values are though generated between an interval of the min and max value that each target has. For iris it was a subjective decision regarding these values. However for the wine dataset the min and max values corresponded to the 10% and 90% percentile for each target dataset.

**training-pipeline.py**

After the Hopsworks feature groups are created, we can run the model training pipeline and after running it the first time, it creates a feature view from an existing feature group from feature-pipeline-daily.py. This feature view is the training dataset schema that will be used to train the model. Usual train/test split is done using this code: X_train, X_test, y_train, y_test = feature_view.train_test_split(0.2)

For the iris version a KNeighborsClassifier is used with 2 neighbors, and it had suifficient evaluation results. But for the wine dataset the same method resulted in terrible metrics, such as precision being around 0.22. Instead of KNeighborsClassifier, KNeighborsRegressor was used with 4 neighbors which proved decent results with MAE being around 0,67 and MSE being around 0,74. Also, looking at the plot with the title "Mean Absolute Error for Each Quality Group", we can see that the model is much more accurate for data that is overrepresented such 5 and 6. And less accurate for underrepresented data such as 3 and 8. This suggests that balancing or other regression techniques should be considered in order for the model to get a better accuracy.

**batch-inference-pipeline.ipynb**

The batch-inference pipeline is for displaying the historical record of the model along with running a prediction on the batch data of the feature group. With every daily feature-pipeline-daily run, a new synthetic flower or wine is added to the feature group. The batch-inference pipeline runs inference on the entire batch of data which includes the new synthetic data and displays the predicted vs actual label of that synthetic data. The batch-inference pipeline also displays the historical record of predictions as well as the confusion matrix or histogram over the whole set of the data predicted. These gets updated with each run and includes the new synthetic dataset.


## Hugging Face

_User interface links (Hugging Face)_
- Iris
    - https://huggingface.co/spaces/erlandekh/iris
    - https://huggingface.co/spaces/erlandekh/iris-monitor
- Wine
    - https://huggingface.co/spaces/erlandekh/wine
    - https://huggingface.co/spaces/erlandekh/wine-monitor
