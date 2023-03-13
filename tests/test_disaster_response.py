'''This module tests the functions in the package
disaster_response. Those tests need to be carried out
in the specified order, because the returned objects
are re-used as objects in the pytest namespace.

Note that the testing configuration fixtures
are located in `conftest.py`. The content from `conftest.py`
must be consistent with the project configuration
file `config.yaml`.

To install pytest:
    
    >> pip install -U pytest

The script expects

- the configuration file `config.yaml` at the root level
    where we call the tests.
- the proper datasets to be located in `./data`
    or the folder specified in `config.yaml`.

Pylint: 9.17/10.

Author: Mikel Sagardia
Date: 2023-03-11
'''
import pytest
import pandas as pd
from sqlite3 import OperationalError
from sqlalchemy import create_engine

# IMPORTANT: the file conftest.py defines the fixtures used in here
# and it contains the necessary imports!

### -- Tests -- ###

def test_load_validate_config(config_filename, load_validate_config):
    """Test load_validate_config() function."""
    config = None
    try:
        config = load_validate_config(config_filename=config_filename)
    except FileNotFoundError as err:
        print("TESTING load_validate_config(): ERROR - config file not found.")
        raise err

    # Store to Pytest namespace
    pytest.config_dict = config

    # Check configuration dictionary
    try:
        assert isinstance(config, dict)
    except AssertionError as err:
        print("TESTING load_validate_config(): ERROR - config is not a dictionary.")
        raise err
    try:
        assert len(config.keys()) > 0
    except AssertionError as err:
        print("TESTING load_validate_config(): ERROR - config is empty.")
        raise err


def test_run_etl(config_filename, run_etl, DATABASE_TABLE_NAME):
    """Test run_etl() function."""
    # Run ETL
    run_etl(config_filename,
            pytest.config_dict['messages_filepath'],
            pytest.config_dict['categories_filepath'],
            pytest.config_dict['database_filepath'])

    # Check the database is there
    engine = create_engine(f"sqlite:///{pytest.config_dict['database_filepath']}")
    try:
        df = pd.read_sql(f"SELECT * FROM {DATABASE_TABLE_NAME}", engine)
    except OperationalError as err:
        print(f"TESTING run_etl(): ERROR - Table {DATABASE_TABLE_NAME} not found in database.")
        raise err

    # Check the dataframe size
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        print("TESTING run_etl(): ERROR - Data frame has no rows / columns.")
        raise err
