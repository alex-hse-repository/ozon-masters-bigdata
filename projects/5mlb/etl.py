#!/opt/conda/envs/dsenv/bin/python

import sys
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.ml.feature import *
from pyspark.ml import Estimator, Transformer
from pyspark.ml import Pipeline

if __name__ == "__main__":
    
    #
    # Run Spark session
    #
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    
    #
    # Read script arguments
    #
    raw_data_path = sys.argv[1]
    processed_data_path = sys.argv[2]

    #
    # Read raw dataset
    #
    schema = StructType([
        StructField("asin", StringType()),
        StructField("id", LongType()),
        StructField("label", IntegerType()),
        StructField("reviewText", StringType()),
        StructField("reviewTime", DateType()),
        StructField("reviewerID", StringType()),
        StructField("reviewerName", StringType()),
        StructField("vote", IntegerType()),
        StructField("summary", StringType()),
        StructField("unixReviewTime", TimestampType()),
        StructField("verified", BooleanType())
    ])

    dataset = spark.read.json(raw_data_path, schema=schema,dateFormat='MM dd, yyyy').cache()
    
    #
    # Preprocessing
    #
    droper = SQLTransformer(statement="SELECT * FROM __THIS__ WHERE reviewText is not null")
    tokenizer = Tokenizer(inputCol="reviewText", outputCol="words")
    hasher = HashingTF(numFeatures=100, binary=True, inputCol=tokenizer.getOutputCol(), outputCol="word_vector")
    pipeline = Pipeline(stages=[
        droper,
        tokenizer,
        hasher        
    ])
    processed_dataset = pipeline.fit(dataset).transform(dataset)
    
    #
    # Save 
    #
    processed_dataset.write.parquet(processed_data_path)
    
    spark.stop()    