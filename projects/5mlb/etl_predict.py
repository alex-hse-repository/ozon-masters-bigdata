#!/opt/conda/envs/dsenv/bin/python

import sys,os
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
    parser.add_argument('--test_path_in', type=str,
                        help="Path to the test dataset")
    parser.add_argument('--prediction_path_out', type=str,
                        help="Path to save predictions")
    parser.add_argument('--sklearn_model', type=str, default='test'
                        help="Name of the model(default: test)")
    
    args = parser.parse_args()
    test_path = args.test_path_in
    prediction_path = args.prediction_path_out
    model_name = args.sklearn_model
    
    
    