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
    parser.add_argument('--train_path_in', type=str,
                        help="Path to the train dataset")
    parser.add_argument('--sklearn_model', type=str, default='test'
                        help="Name of the model(default: test)")
    parser.add_argument('--model_param1', type=float, default=1.0,
                            help="Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization. (default: 1.0)")
    args = parser.parse_args()
    train_path = args.train_path_in
    model_name = args.model_name
    regularization = args.model_param1
    
    