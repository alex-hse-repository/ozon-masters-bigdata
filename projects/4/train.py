from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

import os
import sys
from pyspark.sql.types import *
from model import pipeline

dataset_path = sys.argv[1]
model_path = sys.argv[2]

schema = StructType([
    StructField("asin", StringType()),
    StructField("id", LongType()),
    StructField("overall", DoubleType()),
    StructField("reviewText", StringType()),
    StructField("reviewTime", DateType()),
    StructField("reviewerID", StringType()),
    StructField("reviewerName", StringType()),
    StructField("vote", IntegerType()),
    StructField("summary", StringType()),
    StructField("unixReviewTime", TimestampType()),
    StructField("verified", BooleanType())
])

dataset = spark.read.json(dataset_path, schema=schema,dateFormat='MM dd, yyyy').cache()
pipeline_model = pipeline.fit(dataset)
pipeline_model.write().overwrite().save(model_path)