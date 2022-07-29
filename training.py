import os
import sys
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession


# spark Instance
spark = SparkSession.builder.appName("wine-quality-prediction").getOrCreate()

# input training dataset file
input_file = sys.argv[1]

# Absolute path to the training dataset
input_file = os.path.join(os.path.dirname(__file__), input_file)

# get the training data set
training_dataset = spark.read.csv(input_file, header='true',
                                  inferSchema='true', sep=';')

# get the features and labels
feature_columns = training_dataset.columns[:-1]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
transformed_data = assembler.transform(training_dataset)

# train the model

org_model = LogisticRegression(
    featuresCol="features", labelCol='""""quality"""""')
model = org_model.fit(transformed_data)


# Save training model
model.write().overwrite().save('trained_model')

# Evaluation
trainingSummary = model.summary
fMeasure = trainingSummary.weightedFMeasure()
print("F-measure: %s"
      % fMeasure)

spark.stop()
