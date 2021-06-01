#!/opt/conda/envs/dsenv/bin/python

import sys,os
import random


SPARK_HOME = "/usr/hdp/current/spark2-client"
PYSPARK_PYTHON = "/opt/conda/envs/dsenv/bin/python"
os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
os.environ["SPARK_HOME"] = SPARK_HOME
PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.7-src.zip"))
sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))
SPARK_UI_PORT = random.choice(range(10000, 10200))

from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.functions import *

import mlflow
import mlflow.sklearn 
import argparse

if __name__ == "__main__":
    
    #
    # Start spark session
    #
    
    conf = SparkConf()
    conf.set("spark.ui.port", SPARK_UI_PORT)
    spark = SparkSession.builder.config(conf=conf).appName("MLflow model inference with Spark").getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    
    #
    # Read script arguments
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str,
                        help="Path to the test dataset")
    parser.add_argument('--model_name', type=str, default='test'
                        help="Name of the model(default: test)")
    parser.add_argument('--model_version', type=str, default='1'
                        help="Version of the model(default: 1)")
    parser.add_argument('--prediction_path', type=str,
                        help="Path to save predictions")
    args = parser.parse_args()
    test_path = args.test_path
    model_name = args.model_name
    model_version = args.model_version
    prediction_path = args.prediction_path
    
    #
    # Specifying the model
    #
    spark_udf = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{model_name}/{model_version}")
        
    #
    # Read dataset and predict
    #
    spark_df = spark.read.parquet(test_path)
    predictions = spark_df.withColumn("prediction", spark_udf(*spark_df.drop('id').schema.fieldNames()))
    
    #
    # Save predictions
    #
    predictions[['id','prediction']].write.csv(prediction_path, mode="overwrite", header=False)
    mlflow.end_run()
    spark.stop()  
    
    
    
   