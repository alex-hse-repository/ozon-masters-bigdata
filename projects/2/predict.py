#!/opt/conda/envs/dsenv/bin/python

import sys, os
import logging
from joblib import load
import pandas as pd
import numpy as np

sys.path.append('.')

#
# Init the logger
#
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

#load the model
model = load("2.joblib")

numeric_features = ["if"+str(i) for i in range(1,14)]
categorical_features = ["cf"+str(i) for i in range(1,27)] + ["day_number"]
features = numeric_features
fields = ["id"] + numeric_features+categorical_features

for line in sys.stdin:
    X = line.strip().split('\t')
    X = pd.DataFrame([X],columns = fields).replace('\\N',np.NaN)
    pred = model.predict(X[features])
    out = zip(X['id'], pred)
    print("\n".join(["{0}\t{1}".format(*i) for i in out]))
