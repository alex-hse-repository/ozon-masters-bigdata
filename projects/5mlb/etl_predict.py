#!/opt/conda/envs/dsenv/bin/python

import sys,os
import argparse

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
    parser.add_argument('--model_version', type=str, default="1",
                        help="version of the MLFlow registered model")
    args = parser.parse_args()
    test_path = args.test_path_in
    prediction_path = args.prediction_path_out
    model_name = args.sklearn_model
    model_version = args.model_version
    processed_path = "hdfs:///user/alex-hse-repository/5mlb/processed_data_test.parquet"
    
    #
    # Run etl
    #
    os.system(f"python etl.py {test_path} {processed_path}")

    #
    # Run predict
    #
    os.system(f"python predict.py --test_path={processed_path} --model_name={model_name} --model_version={model_version} --prediction_path={prediction_path}")