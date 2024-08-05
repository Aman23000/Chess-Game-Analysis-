import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gcsfs
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

# Save summary statistics to file
summary.toPandas().to_csv("gs://your-bucket/summary_statistics.csv", index=False)

# Count missing values in each column
missing_values = data.select([count(when(col(c).isNull(), c)).alias(c) for c in data.columns])
missing_values.show()

# Save missing values to file
missing_values.toPandas().to_csv("gs://your-bucket/missing_values.csv", index=False)

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

# Save correlation matrix to file
correlation_df.toPandas().to_csv("gs://your-bucket/correlation_matrix.csv", index=False)

# Collecting a smaller manageable subset for visualization
sample_size = 10000
sample_data = data.sample(False, sample_size / data.count(), seed=1234).toPandas()

# Save plots to Google Cloud Storage
fs = gcsfs.GCSFileSystem()

def save_plot_to_gcs(fig, path):
    with fs.open(path, 'wb') as f:
        fig.savefig(f, format='png')

# Visualizations
plt.figure(figsize=(10, 6))
sns.histplot(sample_data['Eval'], bins=50, kde=True)
plt.title('Eval Distribution')
save_plot_to_gcs(plt.gcf(), 'gs://your-bucket/eval_distribution.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=sample_data, x='Ply', y='Eval', hue='Result')
plt.title('Ply vs Eval')
save_plot_to_gcs(plt.gcf(), 'gs://your-bucket/ply_vs_eval.png')
plt.close()

# Histogram for each feature
for col_name in numeric_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(sample_data[col_name], bins=50, kde=True)
    plt.title(f'{col_name} Distribution')
    save_plot_to_gcs(plt.gcf(), f'gs://your-bucket/{col_name}_distribution.png')
    plt.close()

# Scatter plot matrix
sns.pairplot(sample_data[numeric_columns])
plt.suptitle('Scatter Matrix of Features', y=1.02)
save_plot_to_gcs(plt.gcf(), 'gs://your-bucket/scatter_matrix.png')
plt.close()

# Boxplot to see the distribution of features based on Result
plt.figure(figsize=(12, 6))
sns.boxplot(data=sample_data, x='Result', y='Ply')
plt.title('Boxplot of Ply by Result')
save_plot_to_gcs(plt.gcf(), 'gs://your-bucket/boxplot_ply_by_result.png')
plt.close()

plt.figure(figsize=(12, 6))
sns.boxplot(data=sample_data, x='Result', y='Eval')
plt.title('Boxplot of Eval by Result')
save_plot_to_gcs(plt.gcf(), 'gs://your-bucket/boxplot_eval_by_result.png')
plt.close()

# Count plot for categorical distribution of 'Result'
plt.figure(figsize=(8, 6))
sns.countplot(x='Result', data=sample_data, palette='Set3')
plt.title('Count Plot of Result')
save_plot_to_gcs(plt.gcf(), 'gs://your-bucket/countplot_result.png')
plt.close()

# Violin plot to show the density of the data
plt.figure(figsize=(12, 6))
sns.violinplot(data=sample_data, x='Result', y='Eval', palette='muted')
plt.title('Violin Plot of Eval by Result')
save_plot_to_gcs(plt.gcf(), 'gs://your-bucket/violinplot_eval_by_result.png')
plt.close()

plt.figure(figsize=(12, 6))
sns.violinplot(data=sample_data, x='Result', y='Ply', palette='muted')
plt.title('Violin Plot of Ply by Result')
save_plot_to_gcs(plt.gcf(), 'gs://your-bucket/violinplot_ply_by_result.png')
plt.close()

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

# Save results to file
with fs.open("gs://your-bucket/model_performance.txt", 'w') as f:
    for metric_name, values in results.items():
        f.write(f"{metric_name}:\n")
        for model, score in values.items():
            f.write(f"  {model}: {score}\n")

# Plot confusion matrices
def plot_confusion_matrix(predictions, model_name, filename):
    y_true = np.array(predictions.select('label').collect())
    y_pred = np.array(predictions.select('prediction').collect())
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'{model_name} Confusion Matrix')
    save_plot_to_gcs(plt.gcf(), filename)
    plt.close()

plot_confusion_matrix(lr_predictions, 'Logistic Regression', 'gs://your-bucket/confusion_matrix_lr.png')
plot_confusion_matrix(dt_predictions, 'Decision Tree', 'gs://your-bucket/confusion_matrix_dt.png')
plot_confusion_matrix(rf_predictions, 'Random Forest', 'gs://your-bucket/confusion_matrix_rf.png')

# Random Forest feature importances
rf_feature_importances = rf_model.featureImportances.toArray()
plt.figure(figsize=(10, 6))
plt.barh(range(len(rf_feature_importances)), rf_feature_importances, color='orange')
plt.xlabel('Feature Importance')
plt.ylabel('Feature Index')
plt.title('Random Forest Feature Importances')
save_plot_to_gcs(plt.gcf(), 'gs://your-bucket/random_forest_feature_importances.png')
plt.close()
