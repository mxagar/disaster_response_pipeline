#import sys
import argparse
#import numpy as np
import pandas as pd

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split

from .file_manager import (logger,
                           load_validate_config,
                           load_validate_database_df,
                           save_evaluation_report,
                           save_model)

URL_REGEX = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(categorical_columns,
              nlp_columns,
              target_columns,
              database_filepath):
    """..."""
    df = load_validate_database_df(categorical_columns,
                                   nlp_columns,
                                   target_columns,
                                   database_filename=database_filepath)
    X = df[nlp_columns+categorical_columns]
    Y = df[target_columns]
    
    return X, Y


def tokenize(text):
    pass


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_evaluation(report, evaluation_filepath):
    save_evaluation_report(report, evaluation_filepath)


def save_model(model, model_filepath):
    save_model(model, model_filepath)


def run_training(config_filepath,
                 database_filepath=None,
                 model_filepath=None,
                 evaluation_filepath=None):
    """..."""
    # Load configuration file
    config = load_validate_config(config_filepath)

    # Overwrite dataset and database file paths, if args passed
    if database_filepath:
        config["database_filepath"] = database_filepath
    if model_filepath:
        config["model_filepath"] = model_filepath
    if model_filepath:
        config["evaluation_filepath"] = evaluation_filepath

    print(f"Loading data from {config['database_filepath']}...")
    X, Y = load_data(config["categorical_columns"],
                     config["nlp_columns"],
                     config["target_columns"],
                     config["database_filepath"])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, config["test_size"])
    logger.info("Dataset loaded from SQLite database, validated and split.")
    
    print('Building model...')
    model_grid = build_model()
    logger.info("Model built.")
    
    print('Training model...')
    model_grid.fit(X_train, Y_train)
    logger.info("Model trained.")
    
    print('Evaluating model...')
    report = evaluate_model(model_grid, X_test, Y_test, category_names)
    save_evaluation(report, config["evaluation_filepath"])
    logger.info("Model evaluated and report saved to %s.", config['evaluation_filepath'])

    print(f"Saving best model {config['model_filepath']}...")
    save_model(model_grid.best_estimator_, config["model_filepath"])
    logger.info("Model trained and saved to %s.", config['model_filepath'])

    print('Model trained and saved!')


if __name__ == '__main__':
    # Define parser
    parser = argparse.ArgumentParser(description="Training Pipeline")
    parser.add_argument("config_filepath", type=str, required = True,
                        help="File path of the configuration file.")
    parser.add_argument("database_filepath", type=str, required = False, 
                        help="File path of the ETL output database, used for training.")
    parser.add_argument("model_filepath", type=str, required = False,
                        help="File path of the inference artifact, product of the training pipeline.")
    parser.add_argument("evaluation_filepath", type=str, required = False,
                        help="File path of evaluation report.")
    # Parse arguments
    args = parser.parse_args()
    
    # Check the config file is there
    try:
        assert args.config_filepath
    except AssertionError as err:
        logger.error("Config file path not passed!")
        raise err
    
    # Run Training pipeline
    run_training(args.config_filepath,
                 args.database_filepath,
                 args.model_filepath,
                 args.evaluation_filepath)
