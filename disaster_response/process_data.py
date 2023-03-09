"""This module performs the ETL pipeline
of the project, executing these steps:

- A
- B

Pylint: 

Author: Mikel Sagardia
"""
#import sys
import argparse
#import numpy as np
import pandas as pd

from .file_manager import (logger,
                           load_validate_config,
                           load_validate_datasets,
                           save_to_database)


def load_data(messages_filepath, categories_filepath):
    """..."""
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

def clean_data(df):
    """..."""
    # Create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=";", n=-1, expand=True)
    
    # Select the first row of the categories dataframe
    # and use it to extract a list of new column names for categories.
    row = categories.iloc[0,:].to_list()
    category_colnames = [r.split('-')[0] for r in row]
    categories.columns = category_colnames
    
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

    return df

def save_data(df, database_filename):
    """..."""
    save_to_database(df, database_filename)

def run_etl(config_filepath,
            messages_filepath=None,
            categories_filepath=None,
            database_filepath=None):
    """..."""
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
    df = clean_data(df)
    logger.info("Datasets correctly cleaned.")

    # Load cleaned dataset to databse
    print(f"Saving data to database {config['database_filepath']}"...)
    save_data(df, config['database_filepath'])
    logger.info("Cleaned and merged datasets correctly loaded to database.")

    print("ETL completed!")


if __name__ == '__main__':
    # Define parser
    parser = argparse.ArgumentParser(description="ETL Pipeline")
    parser.add_argument("config_filepath", type=str, required = True,
                        help="File path of the configuration file.")
    parser.add_argument("messages_filepath", type=str
                        help="File path of the dataset with the messages.")
    parser.add_argument("categories_filepath", type=str, required = False,
                        help="File path of the dataset with the categories.")
    parser.add_argument("database_filepath", type=str, required = False, 
                        help="File path of the ETL output database."
    )
    # Parse arguments
    args = parser.parse_args()
    
    # Check the config file is there
    try:
        assert args.config_filepath
    except AssertionError as err:
        logger.error("Config file path not passed!")
        raise err
    
    # Run ETL pipeline
    run_etl(args.config_filepath,
            args.messages_filepath,
            args.categories_filepath,
            args.database_filepath)
