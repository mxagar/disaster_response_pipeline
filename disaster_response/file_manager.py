"""This module contains
general data structure definitions and their respective
loading, validation and saving functions.
In other words, it is a data manager for all structures
used in the library.

Pylint: X.

Author: Mikel Sagardia
Date: 2023-03-09
"""
import os
import logging
import pickle
import yaml
import chardet
from pydantic import BaseModel, ValidationError
from typing import Dict, List, Optional, Tuple # Sequence
#import numpy as np
import pandas as pd
from sqlalchemy import create_engine


# Logging configuration
logging.basicConfig(
    filename='./logs/disaster_response_pipeline.log', # filename, where it's dumped
    level=logging.INFO, # minimum level I log
    filemode='w', # append
    format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')
    # add function/module name for tracing
# Thi will be imported in the rest of the modules
logger = logging.getLogger()


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
    scoring: str


class GeneralConfig(BaseModel):
    """
    General configuration file.
    All configuration relevant to model
    training and data processing
    (i.e., feature engineering, etc.).
    """
    data_messages_path: str
    data_categories_path: str
    test_size: float
    random_seed: int
    model_artifact: str
    evaluation_artifact: str
    random_forest_parameters: ModelConfig
    random_forest_grid_search: TrainingConfig


def fetch_data(directory):
    """This function downloads or fetches the datasets.
    Currently, it's empty, because the datasets
    should be locally stored in ./data/."""
    pass


def detect_encoding(filename):
    """Detect encoding of the dataset (default: 'utf-8').
    
    Args: None
    
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
        data_path (str):
            String of the local dataset path.
            Default values: "./data/messages.csv", "./data/categories.csv".
    
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
    """..."""
    # Connect to SQLite database with SQLAlchemy
    engine = create_engine(f"sqlite:///{database_filename}")
    # Dump table, remove if it already exists
    df.to_sql('Message', engine, if_exists="replace", index=False)


def load_validate_model(
    model_artifact: str = "./exported_artifacts/model.pickle") -> RandomForestClassifier:
    """Loads and validates the (trained) model.
    Validation occurs by checking that the loaded
    object type is RandomForestClassifier.
    Inputs
    ------
    model_artifact : str
        String of to the local model file path.
    Returns
    -------
    model: RandomForestClassifier
        Validated model.
    """
    try:
        with open(model_artifact, 'rb') as f: # 'exported_artifacts/model.pickle'
            model = pickle.load(f)
        # Check that the loaded model is a RandomForestClassifier to validate it
        assert isinstance(model, RandomForestClassifier)
    except FileNotFoundError as e:
        logger.error("Model artifact/pickle not found: %s.", model_artifact)
        raise e
    except AssertionError as e:
        logger.error("Model artifact/pickle is not a RandomForestClassifier.")
        raise e

    return model

def save_model(model: RandomForestClassifier,
               model_artifact: str = "./exported_artifacts/model.pickle") -> None:
    """Persists the model object into a serialized pickle file.
    Inputs
    ------
    model : RandomForestClassifier
        The (trained) model.
    model_artifact: str (default = "./exported_artifacts/model.pickle")
        File path to persist the model.
    Returns
    -------
    None
    """
    with open(model_artifact, 'wb') as f:
        # wb: write bytes
        pickle.dump(model, f)
        
def load_validate_database_df():
    pass

def save_evaluation_report():
    pass