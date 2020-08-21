#Connect to Spark server via PuTTy

#Start Spark with Python

pyspark --master=local

#Import necessary packages

import os
import re
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.ml.classification import LogisticRegression

#Create SparkSession and name it AllstateProjectModelRFR

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("AllstateProjectModelRFR")\
        .getOrCreate()

#Use Spark CSV by Databricks to load data files

trainData = sqlContext.read \
    .format('com.databricks.spark.csv') \
    .options(header='true', inferschema='true') \
    .load("file:///home/usrmkang/mwomac12/allstateClaims/train.csv")

testData = sqlContext.read \
    .format('com.databricks.spark.csv') \
    .options(header='true', inferschema='true') \
    .load("file:///home/usrmkang/mwomac12/allstateClaims/test.csv")

#Alter column name for clarity and preparation for modeling

preparedTrainData = (trainData.withColumnRenamed("loss", "label"))

#Split data random 70/30

[trainSplit, validateSplit] = preparedTrainData.randomSplit([0.7, 0.3])

#Use StringIndexer to find categorical variable columns

isCat = lambda c: c.startswith("cat")
catNewColumn = lambda c: "idx_{0}".format(c) if (isCat(c)) else c

stringIndexer = map(lambda c: StringIndexer(inputCol=c, outputCol=catNewColumn(c))
        .fit(trainData.select(c).union(testData.select(c))), filter(isCat, trainSplit.columns))

#Find cat columns and remove them if above threshold

removeCat = lambda c: not re.match(r"cat(109$|110$|112$|113$|116$)", c)

#Select feature columns

selectFeatureColumns = lambda c: not re.match(r"id|label", c)

featureColumns = map(catNewColumn,
                    filter(selectFeatureColumns,
                        filter(removeCat,
                            trainSplit.columns)))

#Use vector assembler to train

vectAssembler = VectorAssembler(inputCols = featureColumns, outputCol = "features")

#Create RFR Regressor

regressor = RandomForestRegressor(featuresCol = "features", labelCol = "label")

transformer = stringIndexer
transformer.append(vectAssembler)
transformer.append(regressor)

#Construct pipeline

pipeline = Pipeline(stages=transformer)

#Build parameter grid

parameterGrid = ParamGridBuilder() \
    .addGrid(regressor.maxDepth, [5]) \
    .addGrid(regressor.maxBins, [32]) \
    .addGrid(regressor.numTrees, [3]) \
    .build()

#Generate 5 fold cross validator from metrics

crossvalidator = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=parameterGrid,
                          evaluator=RegressionEvaluator(),
                          numFolds= 5)

#Train the model

cvModel = crossvalidator.fit(trainSplit)

#Execution time, start 1:04, finish 1:10 (6 minutes)

#Test model using training data

trainPredictionsAndLabels = cvModel.transform(trainSplit).select("label", "prediction").rdd
validPredictionsAndLabels = cvModel.transform(validateSplit).select("label", "prediction").rdd

trainRegressionMetrics = RegressionMetrics(trainPredictionsAndLabels)
validRegressionMetrics = RegressionMetrics(validPredictionsAndLabels)

bestModel = cvModel.bestModel

#Calculate MSE and RMSE of the training and validation data

"Training data MSE = {0}\n".format(trainRegressionMetrics.meanSquaredError)
"Training data RMSE = {0}\n".format(trainRegressionMetrics.rootMeanSquaredError)

"Validation data MSE = {0}\n".format(validRegressionMetrics.meanSquaredError)
"Validation data RMSE = {0}\n".format(validRegressionMetrics.rootMeanSquaredError)

#Make predictions with test data

predictions = cvModel.transform(testData)

#Grab max and minimum predicted values - data can be exported if entire data set is desired

predictions.agg({"prediction": "max"}).collect()[0][0]

predictions.agg({"prediction": "min"}).collect()[0][0]

#Stop SparkSession

spark.stop()