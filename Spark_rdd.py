import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, when, count
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pyspark.sql.types import DoubleType

# Initialize Spark Session with increased driver maxResultSize and executor memory
spark = SparkSession.builder \
    .appName("chessGameAnalysis") \
    .config("spark.driver.maxResultSize", "4g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

# Read data from source
data_path = "gs://met-777-assign3/newfile/master_training_data_ver7.0d.csv"
data = spark.read.csv(data_path, header=True, inferSchema=True)

# Handle missing values
data = data.na.fill(0)

# Print schema and show summary
data.printSchema()
data.describe().show()

# Cast columns to DoubleType
data = data.withColumn("Ply", col("Ply").cast(DoubleType()))
data = data.withColumn("GamePly", col("GamePly").cast(DoubleType()))
data = data.withColumn("HasCastled", col("HasCastled").cast(DoubleType()))
data = data.withColumn("Eval", col("Eval").cast(DoubleType()))
data = data.withColumn("Result", col("Result").cast(DoubleType()))

# Summary statistics
summary = data.describe()
summary.show()

# Count missing values in each column
missing_values = data.select([count(when(col(c).isNull(), c)).alias(c) for c in data.columns])
missing_values.show()


# Calculate correlation matrix
numeric_columns = ['Ply', 'GamePly', 'HasCastled', 'Eval']
correlations = []
for col1 in numeric_columns:
    for col2 in numeric_columns:
        if col1 != col2:
            corr_value = data.stat.corr(col1, col2)
            correlations.append((col1, col2, corr_value))

correlation_df = spark.createDataFrame(correlations, ["Column1", "Column2", "Correlation"])
correlation_df.show()




# Assemble features for machine learning
feature_columns = ['Ply', 'GamePly', 'HasCastled', 'Eval']
assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')
data = assembler.transform(data)
data.select('features').show(truncate=False)

# Create labels
data = data.withColumn('label', 
                       when(data.Result == 0.0, 0)
                       .when(data.Result == 0.5, 1)
                       .when(data.Result == 1.0, 2)
                       .otherwise(None))

data.select('Result', 'label').distinct().show()

# Split data into training and test sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=1234)

# Initialize classifiers
lr = LogisticRegression(labelCol='label', featuresCol='features')
dt = DecisionTreeClassifier(labelCol='label', featuresCol='features')
rf = RandomForestClassifier(labelCol='label', featuresCol='features')

# Train models
lr_model = lr.fit(train_data)
dt_model = dt.fit(train_data)
rf_model = rf.fit(train_data)

# Make predictions
lr_predictions = lr_model.transform(test_data)
dt_predictions = dt_model.transform(test_data)
rf_predictions = rf_model.transform(test_data)

# Evaluate models
evaluator_accuracy = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
evaluator_f1 = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='f1')
evaluator_precision = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='weightedPrecision')
evaluator_recall = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='weightedRecall')

metrics = {
    'Accuracy': evaluator_accuracy,
    'F1 Score': evaluator_f1,
    'Precision': evaluator_precision,
    'Recall': evaluator_recall
}

results = {}
for metric_name, evaluator in metrics.items():
    results[metric_name] = {
        'Logistic Regression': evaluator.evaluate(lr_predictions),
        'Decision Tree': evaluator.evaluate(dt_predictions),
        'Random Forest': evaluator.evaluate(rf_predictions)
    }

for metric_name, model_results in results.items():
    print(f"\n{metric_name}:")
    for model_name, value in model_results.items():
        print(f"{model_name}: {value:.4f}")



