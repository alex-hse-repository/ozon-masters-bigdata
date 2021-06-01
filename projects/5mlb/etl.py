#!/opt/conda/envs/dsenv/bin/python

import os,sys
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
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import *
from pyspark.ml import Estimator, Transformer
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, HashingTF
import pyspark.sql.functions as F



if __name__ == "__main__":
      
   
    #
    # Run Spark session
    #
    conf = SparkConf()
    conf.set("spark.ui.port", SPARK_UI_PORT)
    spark = SparkSession.builder.config(conf=conf).appName("HW5b").getOrCreate()
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
    processed_dataset = pipeline.fit(dataset).transform(dataset).select("id", "label", "word_vector")
    
    #
    # Get features from column
    #
    def split_array_to_list(col):
        def to_list(v):
            return v.toArray().tolist()
        return F.udf(to_list, ArrayType(DoubleType()))(col)

    processed_dataset = processed_dataset.select(["id", "label", split_array_to_list(F.col("word_vector")).alias("val")])\
        .select(["id", "label"] + [F.col("val")[i] for i in range(100)])

    #
    # Нечто странное
    #
    train_flag = (dataset.where(F.col("label").isNull()).count() == 0)
    if not train_flag: 
        processed_dataset = processed_dataset.drop('label')
    
    #
    # Save 
    #
    processed_dataset.coalesce(1).write.parquet(processed_data_path,mode="overwrite")
    spark.stop()     
