'''Testing configuration module for Pytest.
This file is read by pytest and the fixtures
defined in it are used in all the tested files.
Note that some variables are extracted from the
configuration YAML config.yaml; to that end,
the configuration dictionary must be loaded in the
first test.

Author: Mikel Sagardia
Date: 2023-03-11
'''
import pytest

import disaster_response as dr

# Fixtures of the disaster_response package functions.
# Fixtures are predefined variables passed to test functions.

## -- Library Parameters

@pytest.fixture
def config_filename():
    '''Configuration filename.'''
    return "config.yaml"

@pytest.fixture
def messages_filepath():
    '''Dataset path for with messages.'''
    #return "./data/messages.csv"
    return pytest.config_dict["messages_filepath"]

@pytest.fixture
def categories_filepath():
    '''Dataset path for with categories.'''
    #return "./data/categories.csv"
    return pytest.config_dict["categories_filepath"]

@pytest.fixture
def model_artifact_path():
    '''Path where model is stored.'''
    #return "./models/classifier.pkl"
    return pytest.config_dict["model_filepath"]

@pytest.fixture
def DATABASE_TABLE_NAME():
    '''Database table name: Message.'''
    return dr.DATABASE_TABLE_NAME

## -- Library Functions

@pytest.fixture
def load_validate_config():
    '''load_validate_config() function from disaster_response.'''
    return dr.load_validate_config

@pytest.fixture
def run_etl():
    '''run_etl() function from disaster_response.'''
    return dr.run_etl

## -- Variable plug-ins

def config_dict_plugin():
    '''Initialize pytest project config container as None:
    pytest.config_dict: dict'''
    return None

def pytest_configure():
    '''Create objects in namespace:
    - `pytest.df_train_test`
    - `pytest.processing_parameters`
    - `pytest.model`
    - `pytest.config_dict`
    '''
    pytest.config_dict = config_dict_plugin()
