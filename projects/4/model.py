from pyspark.ml.feature import *
from pyspark.ml import Estimator, Transformer
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline

droper = SQLTransformer(statement="SELECT * FROM __THIS__ WHERE reviewText is not null")
tokenizer = Tokenizer(inputCol="reviewText", outputCol="words")
hasher = HashingTF(numFeatures=100, binary=True, inputCol=tokenizer.getOutputCol(), outputCol="word_vector")
lr = LinearRegression(featuresCol=hasher.getOutputCol(), labelCol="overall", maxIter=15)
pipeline = Pipeline(stages=[
    droper,
    tokenizer,
    hasher,
    lr
])