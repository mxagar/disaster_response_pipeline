"""This module runs the ETL and training pipelines
implemented in the disaster_response module.

To use it, run

    $ python main.py
    
Author: Mikel Sagardia
Date: 2023-03-11
"""
import argparse
import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics import classification_report

#from disaster_response import *
import disaster_response as dr

def run_pipelines(config_file,
                  messages_filepath,
                  categories_filepath,
                  database_filepath,
                  model_filepath,
                  evaluation_filepath,
                  check_etl=True,
                  check_training=True):
    
    # Load config file
    config = dr.load_validate_config(config_filename=config_file)
    
    # Run ETL
    dr.run_etl(config_file,
            messages_filepath,
            categories_filepath,
            database_filepath)

    # Check database
    if check_etl:
        engine = create_engine(f"sqlite:///{config['database_filepath']}")
        df = pd.read_sql(f"SELECT * FROM {dr.DATABASE_TABLE_NAME}", engine)
        print(f"Dataset shape: {df.shape}")

    # Run training
    dr.run_training(config_file,
                 database_filepath,
                 model_filepath,
                 evaluation_filepath)
    
    # Check model
    if check_training:
        model = dr.load_validate_model(model_artifact=config['model_filepath'])
        X, Y = dr.load_XY(config['categorical_columns'],
                        config['nlp_columns'],
                        config['target_columns'],
                        config['database_filepath'])
        pred = model.predict(X)
        for i in range(len(config['target_columns'])):
            print("\nTarget column: {config['target_columns'][i]}")
            print(classification_report(Y.iloc[:,i],pred[:,i]))
    

if __name__ == "__main__":
    # Define parser
    parser = argparse.ArgumentParser(description="ETL and Training Pipelines")
    parser.add_argument("--config_filepath", type=str, required = False,
                        help="File path of the configuration file.")
    parser.add_argument("--messages_filepath", type=str, required = False,
                        help="File path of the dataset with the messages.")
    parser.add_argument("--categories_filepath", type=str, required = False,
                        help="File path of the dataset with the categories.")
    parser.add_argument("--database_filepath", type=str, required = False, 
                        help="File path of the ETL output database, used for training.")
    parser.add_argument("--model_filepath", type=str, required = False,
                        help="File path of the inference artifact, product of the training pipeline.")
    parser.add_argument("--evaluation_filepath", type=str, required = False,
                        help="File path of evaluation report.")
    # Parse arguments
    args = parser.parse_args()
    
    # Check the config file is there
    config_file = "./config.yaml"
    if args.config_filepath:
        config_file = args.config_filepath

    # Run ETL and training pipelines
    run_pipelines(config_file,
                  args.messages_filepath,
                  args.categories_filepath,
                  args.database_filepath,
                  args.model_filepath,
                  args.evaluation_filepath)
