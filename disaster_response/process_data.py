"""This module performs the ETL pipeline
of the project, executing these steps:

- Load the source datasets.
- Clean and merge the datasets.
- Save the processed dataset into a SQLite database.

All settings are in config.yaml.

Pylint: X

To use this, run in a correct environment:

    $ python process_data.py --config_filepath config.yaml

Author: Mikel Sagardia
Date: 2023-03-09
"""
#import sys
#import argparse
#import numpy as np
import pandas as pd

from .file_manager import (logger,
                           load_validate_config,
                           load_validate_datasets,
                           save_to_database)


def load_data(messages_filepath, categories_filepath):
    """Load source datasets, validate them and merge them.
    
    Args:
        messages_filepath (str): file path of the messages dataset.
        categories_filepath (str): file path of the categories dataset.
        
    Returns:
        df (pd.DataFrame): dataframe with the merged datasets.
    """
    # Load and validate datasets
    messages, categories = load_validate_datasets(messages_filepath,
                                                  categories_filepath)
    
    # Merge datasets
    df = pd.merge(left=messages,
            right=categories,
            how='inner',
            left_on='id',
            right_on='id')
    
    return df

def clean_data(df, target_columns, nlp_columns):
    """Clean merged dataset:
    
    - Transform categories into booleans.
    - Check that category values are correct.
    - Drop duplicates and NaNs.
    
    Args:
        df (pd.DataFrame): dataframe to be cleaned.
        target_columns (list): target column names, categories.
        nlp_columns (list): NLP/text column names.
    
    Returns:
        df (pd.DataFrame): cleaned dataframe.
    """
    # Create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=";", n=-1, expand=True)
    
    # Select the first row of the categories dataframe
    # and use it to extract a list of new column names for categories.
    row = categories.iloc[0,:].to_list()
    category_colnames = [r.split('-')[0] for r in row]
    categories.columns = category_colnames
    
    # Check that the categories are the expected (not necessarily in same order)
    for i, col in enumerate(category_colnames):
        try:
            #assert col == target_columns[i]
            assert col in target_columns
        except AssertionError as e:
            logger.error("Unexpected category column found in merged dataset: %s.", col)
            raise e
    
    # Map cell values from string-0/1 to 0/1
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str.get(1)     
        # Convert column from string to numeric
        categories[column] = categories[column].astype(str).astype('int32')
    
    # Drop the original categories column from `df`
    df.drop('categories', inplace=True, axis=1)
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Drop any entry in which the input message is NA
    df.dropna(subset=nlp_columns, inplace=True)
    
    # Drop any entry in which the target categories are NA
    df.dropna(subset=category_colnames, inplace=True)

    return df

def save_data(df, database_filename):
    """Save cleaned dataset to database.
    This is a wrapper function to the function in
    file_manager.py.
    
    Args:
        df (pd.DataFrame): cleaned dataframe.
        database_filename (str): database filename.
        
    Returns: None.
    """
    save_to_database(df, database_filename)

def run_etl(config_filepath,
            messages_filepath=None,
            categories_filepath=None,
            database_filepath=None):
    """Run ETL pipeline; this is the main function of the module,
    which executes the following steps:
    
    - Load the source datasets.
    - Clean and merge the datasets.
    - Save the processed dataset into a SQLite database.
    
    Args:
        config_filepath (str): configuration file path.
        messages_filepath (str): file path of the dataset with the messages.
        categories_filepath (str): file path of the dataset with the categories.
        database_filepath (str): database filename/path.
    
    Returns: None.
    """
    # Load configuration file
    config = load_validate_config(config_filepath)

    # Overwrite dataset and database file paths, if args passed
    if messages_filepath:
        config["messages_filepath"] = messages_filepath
    if categories_filepath:
        config["categories_filepath"] = categories_filepath
    if database_filepath:
        config["database_filepath"] = database_filepath

    # Load datasets and merge/join
    print(f"Loading data from {config['messages_filepath']} and {config['categories_filepath']}...")
    df = load_data(config["messages_filepath"], config["categories_filepath"])
    logger.info("Datasets correctly loaded.")

    # Clean datasets
    print("Cleaning data...")
    df = clean_data(df,
                    config["target_columns"],
                    config["nlp_columns"])
    logger.info("Datasets correctly cleaned.")

    # Load cleaned dataset to databse
    print(f"Saving data to database {config['database_filepath']}")
    save_data(df, config['database_filepath'])
    logger.info("Cleaned and merged datasets correctly loaded to database.")

    print("ETL completed!")
