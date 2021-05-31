#!/opt/conda/envs/dsenv/bin/python

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
    parser.add_argument('--train_path', type=str,
                        help="Path to the train dataset")
    parser.add_argument('--model_name', type=str, default='test'
                        help="Nmae of the model(default: test)")
    parser.add_argument('--model_param1', type=float, default=1.0,
                            help="Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization. (default: 1.0)")
    args = parser.parse_args()
    train_path = args.train_path
    model_name = args.model_name
    regularization = args.model_param1

    #
    # Specifying the model
    #
    target = 'label'
    model = LogisticRegression(C=regularization)

    #
    # Read dataset
    #
    data = pd.read_parquet(train_path)
    X,y = data.drop([target],axis=1),data[target]
   
    #
    # Train the model
    #
    with mlflow.start_run():
        model.fit(X, y)
        mlflow.sklearn.log_model(model,artifact_path="model",registered_model_name=model_name)
        mlflow.log_param("model_param1", regularization)      