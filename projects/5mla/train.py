#!/opt/conda/envs/dsenv/bin/python

import pandas as pd
import mlflow
import mlflow.sklearn 
import argparse
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    #
    # Read script arguments
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str,
                            help="Path to the train dataset")
    parser.add_argument('--regularization', type=float, default=1.0,
                            help="Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization. (default: 1.0)")
    args = parser.parse_args()
    train_path = args.train_path
    regularization = args.regularization

    #
    # Specifying the model
    #
    numeric_features = ["if"+str(i) for i in range(1,14)]
    categorical_features = ["cf"+str(i) for i in range(1,27)] + ["day_number"]
    #features = numeric_features
    features = ['if1','if2']
    fields = ["id", "label"] + numeric_features + categorical_features

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaller', StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, features)
        ]
    )
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('linearregression', LogisticRegression(C=regularization))
    ])


    #
    # Read dataset
    #
    read_table_opts = dict(sep="\t", names=fields, index_col=False,nrows=10000)
    df = pd.read_table(train_path, **read_table_opts)

    #split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df['label'], test_size=0.33, random_state=42
    )

    #
    # Train the model
    #
    with mlflow.start_run():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)
        model_score = log_loss(y_test,y_prob)
        mlflow.sklearn.log_model(model,artifact_path="model")
        #mlflow.log_params(model.get_params())
        mlflow.log_param("model_param1", regularization)
        mlflow.log_metric('log_loss', model_score)