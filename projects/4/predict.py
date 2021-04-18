from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from pyspark.ml import Pipeline, PipelineModel

model_path = sys.argv[1]
dataset_path = sys.argv[2]
predictions_path = sys.argv[3]

model = PipelineModel.load(model_path)

schema = StructType([
    StructField("asin", StringType()),
    StructField("id", LongType()),
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
predictions = model.transform(dataset)
predictions.write().overwrite().save(predictions_path)