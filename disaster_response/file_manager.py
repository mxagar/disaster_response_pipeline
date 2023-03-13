"""This module contains
general data structure definitions and their respective
loading, validation and saving functions.
In other words, the module is a data manager for all structures
used in the library, and acts also
as a persistence/loading manager.

Pylint: 7.45/10.

Author: Mikel Sagardia
Date: 2023-03-09
"""
import os
import logging
from typing import Dict #, List, Optional, Tuple,  Sequence
import yaml
#import joblib
import skops.io as sio
import chardet
from pydantic import BaseModel, ValidationError
#import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlite3 import OperationalError
#from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline


# Logging configuration
logging.basicConfig(
    #filename='./logs/disaster_response_pipeline.log', # filename, where it's dumped
    filename='./disaster_response_pipeline.log', # filename, where it's dumped
    level=logging.INFO, # minimum level I log
    filemode='w', # append
    format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
    # add function/module name for tracing
# Thi will be imported in the rest of the modules
logger = logging.getLogger()

DATABASE_TABLE_NAME = "Message"

class ModelConfig(BaseModel):
    """
    Model configuration:
    default model parameter values.
    """
    n_estimators: int
    criterion: str
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int
    min_weight_fraction_leaf: float
    max_features: str
    max_leaf_nodes: None # null
    min_impurity_decrease: float
    bootstrap: bool
    oob_score: bool
    n_jobs: None # null
    random_state: int
    verbose: int
    warm_start: bool
    class_weight: str
    ccp_alpha: float
    max_samples: None # null


class TrainingConfig(BaseModel):
    """
    Training configuration, i.e.,
    hyperparameter tuning definition with grid search.
    """
    hyperparameters: Dict
    cv: int
    #scoring: str


class GeneralConfig(BaseModel):
    """
    General configuration file.
    All configuration relevant to model
    training and data processing
    (i.e., feature engineering, etc.).
    """
    messages_filepath: str
    categories_filepath: str
    test_size: float
    random_seed: int
    model_filepath: str
    evaluation_filepath: str
    random_forest_parameters: ModelConfig
    grid_search: TrainingConfig


def fetch_data(directory):
    """This function downloads or fetches the datasets.
    Currently, it's empty, because the datasets
    should be locally stored in ./data/."""
    pass


def detect_encoding(filename):
    """Detect encoding of the dataset (default: 'utf-8').
    
    Args: None.
    
    Returns:
        encoding (str): encoding as string
    """
    # !pip install chardet
    encoding = 'utf-8'
    with open(filename, 'rb') as file:
        encoding = chardet.detect(file.read())['encoding']

    return encoding


def load_validate_config(
    config_filename: str = "config.yaml") -> dict:
    """Loads and validates general configuration YAML.
    Validation occurs by converting the loaded
    dictionary into the GeneralConfig class/object defined in core.py.

    Args:
        config_filename (str): String of to the local config file path.
    
    Returns:
        config (dict): Validated configuration dictionary.
    """
    config = {}
    try:
        with open(config_filename) as f: # 'config.yaml'
            config = yaml.safe_load(f)
            # Convert dictionary to Config class to validate it
            # Note that the GeneralConfig object is not used after creating it,
            # we simply instantiate if to check that the config file is conform
            # with the fields in GeneralConfig
            _ = GeneralConfig(**config)
    except FileNotFoundError as e:
        logger.error("Configuration YAML not found: %s.", config_filename)
        raise e
    except ValidationError as e:
        logger.error("Configuration file validation error.")
        raise e

    return config


def load_validate_datasets(
    data_messages_path: str = "./data/messages.csv",
    data_categories_path: str = "./data/categories.csv") -> pd.DataFrame:
    """Gets and loads datasets as a dataframes
    and validates them (checking their shape).

    Args:
        data_messages_path (str):
            String of the local messages dataset path.
            Default value: "./data/messages.csv".
        data_categories_path (str):
            String of the local categories dataset path.
            Default value: "./data/categories.csv".
    
    Returns
        messages (pd.DataFrame): Validated messages dataframe.
        categories (pd.DataFrame): Validated categories dataframe.
    """
    # Download datasets to local file directory
    # NOTE: fetch_data() is currently empty, the datasets are stored locally
    folders = ['.']
    folders.extend(data_messages_path.split(os.sep))
    data_path = os.path.join(*folders[:-1])
    fetch_data(directory=data_path)

    # Read datasets
    try:
        encoding = detect_encoding(data_messages_path)
        messages = pd.read_csv(data_messages_path, encoding=encoding) # './data/messages.csv'
    except FileNotFoundError as e:
        logger.error("Dataset file not found: %s.", data_path)
        raise e

    try:
        encoding = detect_encoding(data_messages_path)
        categories = pd.read_csv(data_categories_path, encoding=encoding) # './data/categories.csv'
    except FileNotFoundError as e:
        logger.error("Dataset file not found: %s.", data_path)
        raise e

    # Validate dataset sizes
    try:
        assert messages.shape[0] > 10
        assert messages.shape[1] == 4
    except AssertionError as e:
        logger.error("Unexpected shape of dataset messages: %s.", str(messages.shape))
        raise e

    try:
        assert categories.shape[0] > 10
        assert categories.shape[1] == 2
    except AssertionError as e:
        logger.error("Unexpected shape of dataset categories: %s.", str(categories.shape))
        raise e

    return messages, categories


def save_to_database(df,
                     database_filename="data/DisasterResponse.db"):
    """Save processed dataframe to the SQLite database.
    
    Args:
        df (pd.DataFrame): processed dataframe.
        database_filename (str): file name/path of the database.
    
    Returns: None.
    """
    # Connect to SQLite database with SQLAlchemy
    engine = create_engine(f"sqlite:///{database_filename}")
    # Dump table, remove if it already exists
    df.to_sql(DATABASE_TABLE_NAME, engine, if_exists="replace", index=False)


def load_validate_database_df(categorical_columns,
                              nlp_columns,
                              target_columns,
                              database_filename="data/DisasterResponse.db"):
    """Load dataframe from SQLite database
    and validate it.
    
    Column names are passed for the validation;
    they should be defined in teh config.yaml file.
    
    Args:
        categorical_columns (list): list with the names of the categorical columns.
        nlp_columns (list): list with the names of the text columns.
        target_columns (list): list with the names of the target columns
        database_filename (str): filename of the database.
    
    Returns:
        df (pd.DataFrame): loaded and validated dataframe.
    """
    # Load dataset
    try:
        # Connect to database
        engine = create_engine(f"sqlite:///{database_filename}")
        # Run SELECT * query; WATCH OUT: Using this SQL query text requires sqlalchemy<2.0
        df = pd.read_sql(f"SELECT * FROM {DATABASE_TABLE_NAME}", engine)
    except OperationalError as e:
        logger.error("Table Message cannot be accessed in %s.", database_filename)
        raise e

    # Validate dataframe shape and column names
    try:
        assert df.shape[0] > 10
        assert df.shape[1] == 40
    except AssertionError as e:
        logger.error("Unexpected shape of dataset messages: %s.", str(df.shape))
        raise e

    all_processed_cols = categorical_columns + nlp_columns + target_columns
    for col in all_processed_cols:
        try:
            assert col in list(df.columns)
        except AssertionError as e:
            logger.error("Expected column not found: %s.", col)
            raise e

    return df

def save_model(model: Pipeline,
               model_artifact: str = "./models/classifier.pkl") -> None:
    """Persists the model object into a serialized pickle file.
    
    Args:
        model (Pipeline):
            The (trained) model, based on RandomForestClassifier
        model_artifact (str):
            File path to persist the model.
    
    Returns: None.
    """
    with open(model_artifact, 'wb') as f:
        # wb: write bytes
        sio.dump(model, f)

def save_evaluation_report(report, evaluation_filepath):
    """Persist evaluation report to file.
    
    Args:
        report (list): list of report strings to write to file.
        evaluation_filepath (str): filename of the evaluation report.

    Returns: None."""
    with open(evaluation_filepath, 'w') as f:
        f.write('\n'.join(report))

def load_validate_model(
    model_artifact: str = "./models/classifier.pkl") -> Pipeline:
    """Loads and validates the (trained) model.
    Validation occurs by checking that the loaded
    object type is Pipeline,
    based on a RandomForestClassifier.
    
    Args:
        model_artifact (str):
            String of to the local model file path.
    
    Returns:
        model (Pipeline):
            Validated model, based on a RandomForestClassifier.
    """
    try:
        with open(model_artifact, 'rb') as f: # 'models/classifier.pkl'
            model = sio.load(f, trusted=True)
        # Check that the loaded model is a RandomForestClassifier to validate it
        assert isinstance(model, Pipeline)
    except FileNotFoundError as e:
        logger.error("Model artifact/pickle not found: %s.", model_artifact)
        raise e
    except AssertionError as e:
        logger.error("Model artifact/pickle is not a Pipeline.")
        raise e

    return model
