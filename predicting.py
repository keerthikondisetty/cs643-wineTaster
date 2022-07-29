# Importing the libraries
import sys
from pyspark.mllib.classification import LogisticRegressionModel
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler as VA
import os


# spark Instance
spark = SparkSession.builder.appName("wine-quality-prediction").getOrCreate()

# validation Dataset
input_file = "ValidationDataset.csv"


# Absolute path to the training dataset
input_file = os.path.join(os.path.dirname(__file__), input_file)

# get the file mentioned above
data = spark.read.csv(input_file, header='true', inferSchema='true', sep=';')

# load the previously built model
trained_model_path = os.path.join(
    os.path.dirname(__file__), "trained_model/")
print("Trained Model Path::", trained_model_path)

trainedModel = LogisticRegressionModel.load(
    path=trained_model_path, sc=spark.sparkContext)

# get the vector features from the data
feat_cols = data.columns[:-1]
assembler = VA(inputCols=feat_cols, outputCol="features")
data_with_features = assembler.transform(data)

# transform the model and predict
predictions = trainedModel.transform(data_with_features)
predictions.select(predictions.columns[-1].show())

# getting the evaluation results
evaluationSummary = trainedModel.summary

print("F-measured:")
for i, f in enumerate(evaluationSummary.fMeasureByLabel()):
    print("label %d: %s" % (i, f))

feature_measure = evaluationSummary.weightedFMeasure()
print("feature_measure: %s" % feature_measure)

spark.stop()
