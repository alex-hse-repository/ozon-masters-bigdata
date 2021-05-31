#!/opt/conda/envs/dsenv/bin/python

import sys,os
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark import SparkConf

import pandas as pd
import mlflow
import mlflow.sklearn 
import argparse
import sklearn
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
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
    # Start spark session
    #
    SPARK_HOME = "/usr/hdp/current/spark2-client"
    PYSPARK_PYTHON = "/opt/conda/envs/dsenv/bin/python"
    os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
    os.environ["SPARK_HOME"] = SPARK_HOME

    PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
    sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.7-src.zip"))
    sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))
    
    conf = SparkConf()
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")

    spark = SparkSession.builder.config(conf=conf).appName("MLflow model inference with Spark").getOrCreate()
      
    #
    # Specifying the model
    #
    spark_udf = mlflow.pyfunc.spark_udf(spark, model_uri=f"{run[0].info.artifact_uri}/model")
       
    #
    # Read dataset
    #
    data = pd.read_parquet(test_path)
    spark_df = spark.createDataFrame(data)
    spark_df.withColumn("prediction", spark_udf(*spark_df.schema.fieldNames()))
    
    #
    # Save predictions
    #
    spark_df[['id','prediction']].write.mode('overwrite').option(head='false').csv(prediction_path)
    
    spark.stop()    