import os
import sys

SPARK_HOME = "/usr/hdp/current/spark2-client"
PYSPARK_PYTHON = "/opt/conda/envs/dsenv/bin/python"
os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
os.environ["SPARK_HOME"] = SPARK_HOME

PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.7-src.zip"))
sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))

import random
from pyspark import SparkContext, SparkConf

spark_ui_port = random.choice(range(10000, 11000))
print(f"Spark UI port: {spark_ui_port}")

conf = SparkConf()
conf.set("spark.ui.port", spark_ui_port)

sc = SparkContext(appName="BFS", conf=conf)

sourse = sys.argv[1]
target = sys.argv[2]
dataset_path = sys.argv[3]
answers_path = sys.argv[4]

def add_node(node):
    if(node[0]==sourse):
        return (node[0],(node[1],0,[[node[0]]],'entered'))
    return (node[0],(node[1],2*v,[],'not entered'))

rdd = sc.textFile(dataset_path)
graph = rdd.map(lambda x : x.split('\t')[::-1]).cache()
links = graph.groupByKey().mapValues(list).cache()
v = links.count()
matrix = links.map(add_node)

def step(node):
    status = node[1][3]
    if(status=='entered'):
        v = node[0]
        neigh = node[1][0]
        dist = node[1][1]
        paths = node[1][2]
        if(v!=target):
            for u in neigh:
                udist = dist+1
                upaths = [path+[u] for path in paths]
                ustatus = 'entered'
                entry = (u,([],udist,upaths,ustatus))
                yield entry
        else:
            finish = True
        entry = (v,(neigh,dist,paths,'ready'))
        yield entry
    yield node
    
def update(version1,version2):
    neigh = version1[0]
    if(len(version2[0])>len(neigh)):
        neigh = version2[0]
    
    dist = min(version1[1],version2[1])
    
    mapping = {'not entered':0,'entered':1,'ready':2}
    inverse_mapping = {0:'not entered',1:'entered',2:'ready'}
    status = inverse_mapping[max(mapping[version1[3]],mapping[version2[3]])]   

    paths = []
    if(version1[3]==status and version1[1]==dist):
        paths.extend(version1[2])
    if(version2[3]==status and version2[1]==dist):
        paths.extend(version2[2])
         
    return (neigh,dist,paths,status)

finish = False 
queue = matrix.filter(lambda x: x[1][3]=='entered')
while((not finish) and (not queue.isEmpty())):
    matrix = matrix.flatMap(step)
    matrix = matrix.reduceByKey(update)
    queue = matrix.filter(lambda x: x[1][3]=='entered')
    
ans = matrix.filter(lambda x: x[0]==target).collect()[0][1][2] 
rdd_ans = sc.parallelize(ans)
rdd_ans.saveAsTextFile(answers_path)
    
sc.stop()
    