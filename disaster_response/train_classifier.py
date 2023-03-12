"""This module contains the training pipeline
which outputs the ready inference pipeline along
with an evaluation report to file.

Before running this module, we need to run
the ETL pipeline, which creates the processed dataset
ready to feed to the model trained here.

The used model is a RandomForestClassifier embedded in 
a MultiOutputClassifier, because we have 36 possible
targets.

GridSearchCV is applied to optimize the hyperparameters.

All settings are in config.yaml.

Pylint: X

To use this, run in a correct environment:

    $ python train_classifier.py --config_filepath config.yaml

Author: Mikel Sagardia
Date: 2023-03-10
"""
#import sys
#import re
#import argparse
#import numpy as np
#import pandas as pd
#from string import punctuation

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report #, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline #, FeatureUnion
#from sklearn.base import BaseEstimator, TransformerMixin

from .file_manager import (logger,
                           load_validate_config,
                           load_validate_database_df,
                           save_evaluation_report,
                           save_model)

def load_XY(categorical_columns,
              nlp_columns,
              target_columns,
              database_filepath):
    """Load dataset to train model from SQLite database.
    The auxiliary function load_validate_database_df()
    from the file manager is used, which validates the dataframe.
    The loaded dataframe is split into X (features) and Y (targets).
    
    Args:
        categorical_columns (list): list with the names of the categorical columns.
        nlp_columns (list): list with the names of the text columns.
        target_columns (list): list with the names of the target columns
        database_filename (str): filename of the database.
    
    Returns:
        X (pd.DataFrame): dataframe with the features.
        Y (pd.DataFrame): dataframe with the multiple targets.
    """
    """Load dataset to train model from SQLite database.
    The auxiliary function load_validate_database_df()
    from the file manager is used, which validates the dataframe.
    The loaded dataframe is split into X (features) and Y (targets).
    
    Args:
        categorical_columns (list): list with the names of the categorical columns.
        nlp_columns (list): list with the names of the text columns.
        target_columns (list): list with the names of the target columns
        database_filename (str): filename of the database.
    
    Returns:
        X (pd.DataFrame): dataframe with the features.
        Y (pd.DataFrame): dataframe with the multiple targets.
    """
    df = load_validate_database_df(categorical_columns,
                                   nlp_columns,
                                   target_columns,
                                   database_filename=database_filepath)
    #X = df[nlp_columns+categorical_columns]
    X = df[nlp_columns]
    Y = df[target_columns]
    
    return X, Y


def tokenize(text):
    """Perform the NLP:
    
    - Clean: remove punctuation
    - Normalize (to lower)
    - Tokenize
    - Lemmatize
    
    Args:
        text (string): message
    
    Returns:
        clean_tokens (list): list of processed lemmas
    """
    # Remove punctuation
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    text =  ''.join([c for c in text if c not in punctuation])

    # Tokenize
    tokens = word_tokenize(text)

    # Lemmatize

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# NOTE:
# In order to be able to serialize an object (pickle)
# we cannot use lambdas in it
def squeeze(x):
    return x.squeeze()


def toarray(x):
    return x.toarray()


def build_model(config):
    """A multi-target classification model is generated
    embedded in a Pipeline and a GridSearch.
    
    Args:
        config (dict): dictionary with all config parameters
            from config.yaml.
    
    Returns: None.
    """
    """A multi-target classification model is generated
    embedded in a Pipeline and a GridSearch.
    
    Args:
        config (dict): dictionary with all config parameters
            from config.yaml.
    
    Returns: None.
    """
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('onehot', OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore"))])

    # CountVectorizer and TfidfTransformer have some issues
    # with the size of the arrays, and they require some reshaping
    # as done here with FunctionTransformer; lambdas cannot be serialized
    nlp_transformer = Pipeline([
        ("squeeze", FunctionTransformer(squeeze)),
        ('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None)),
        ('tfidf', TfidfTransformer()),
        ("toarray", FunctionTransformer(toarray)),
    ])

    processor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, config["categorical_columns"]),
            ("nlp", nlp_transformer, config["nlp_columns"]),
        ],
        remainder="drop",  # This drops the columns that we do not transform
    )

    # MultiOutputClassifier is used to handle N-D target
    # However note that "this strategy consists in fitting one classifier per target",
    # thus, it will take much longer (xN)
    pipe = Pipeline(
        steps=[
            #("processor", processor),
            ("processor", nlp_transformer),
            ("classifier", MultiOutputClassifier(
                RandomForestClassifier(**config["random_forest_parameters"]))),
        ]
    )
    
    param_grid = {
        'classifier__estimator__n_estimators': config["grid_search"]["hyperparameters"]["n_estimators"], # 100, 200
        'classifier__estimator__min_samples_split': config["grid_search"]["hyperparameters"]["min_samples_split"] # 2, 3
    }
    
    # Define Grid Search
    # Define Grid Search
    grid = GridSearchCV(pipe,
                        param_grid=param_grid,
                        #scoring=config["grid_search"]["scoring"], # "f1"
                        cv=config["grid_search"]["cv"]) # 3
    
    return grid


def evaluate_model(model, X_test, Y_test, category_names):
    """The trained model/pipeline is evaluated.
    The results are printed and returned in a list of strings.
    
    Args:
        model (GridSearchCV): trained inference pipeline.
        X_test (pd.DataFrame): features of the test split.
        Y_test (pd.DataFrame): targets of the test split.
        category_names (list): names of the target columns.
    
    Returns:
        report (list): list of strings that contain the evaluation
            results.
    """
    """The trained model/pipeline is evaluated.
    The results are printed and returned in a list of strings.
    
    Args:
        model (GridSearchCV): trained inference pipeline.
        X_test (pd.DataFrame): features of the test split.
        Y_test (pd.DataFrame): targets of the test split.
        category_names (list): names of the target columns.
    
    Returns:
        report (list): list of strings that contain the evaluation
            results.
    """
    # Predict test split
    Y_pred = model.predict(X_test)
    
    # Generate report
    report = []
    report.append(f"Best estimator: \n{model.best_estimator_}")
    for i in range(len(category_names)): # Y_test.columns
        report.append(f"\nTarget column: {category_names[i]}")
        report.append(classification_report(Y_test.iloc[:,i],Y_pred[:,i]))

    # Print report
    for r in report:
        print(r)

    return report


def save_evaluation(report, evaluation_filepath):
    """Save evaluation report to file.
    Wrapper function of the function
    save_evaluation_report() in the file manager module.
    
    Args:
        report (list): list of evaluation strings.
        evaluation_filepath (str): file path to save the evaluation report.
    Returns: None.
    """
    """Save evaluation report to file.
    Wrapper function of the function
    save_evaluation_report() in the file manager module.
    
    Args:
        report (list): list of evaluation strings.
        evaluation_filepath (str): file path to save the evaluation report.
    Returns: None.
    """
    save_evaluation_report(report, evaluation_filepath)


def save_pipeline(model, model_filepath):
    """Save inference artifact report to file.
    Wrapper function of the function
    save_model() in the file manager module.

    Args:
        model (GridSearchCV): trained inference pipeline.
        model_filepath (str): file path to save the inference pipeline.

    Returns: None.
    """
    save_model(model, model_filepath)


def run_training(config_filepath,
                 database_filepath=None,
                 model_filepath=None,
                 evaluation_filepath=None):
    """Training pipeline, with the following steps:
    
    - Load preprocessed (ETL) dataset from database.
    - Build an inference pipeline with a model embedded
    in a GridSearchCV, and based on a RandomForestClassifier.
    - Train the inference pipeline.
    - Evaluate the trained pipeline with the test split.
    - Save the inference artifact (model) and teh evaluation report.
    
    All the necessary parameters (e.g., paths, etc.) and in the
    configuration file config.yaml. However, some filenames/paths
    can be overwritten.
    
    Args:
        config_filepath (str): configuration file path.
        database_filepath (str): path to the database where the data is.
        model_filepath (str): path where the inference artifact is saved.
        evaluation_filepath (str): path where the evaluation report is saved.
        
    Returns: None.
    """
    """Training pipeline, with the following steps:
    
    - Load preprocessed (ETL) dataset from database.
    - Build an inference pipeline with a model embedded
    in a GridSearchCV, and based on a RandomForestClassifier.
    - Train the inference pipeline.
    - Evaluate the trained pipeline with the test split.
    - Save the inference artifact (model) and teh evaluation report.
    
    All the necessary parameters (e.g., paths, etc.) and in the
    configuration file config.yaml. However, some filenames/paths
    can be overwritten.
    
    Args:
        config_filepath (str): configuration file path.
        database_filepath (str): path to the database where the data is.
        model_filepath (str): path where the inference artifact is saved.
        evaluation_filepath (str): path where the evaluation report is saved.
        
    Returns: None.
    """
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
    X, Y = load_XY(config["categorical_columns"],
                     config["nlp_columns"],
                     config["target_columns"],
                     config["database_filepath"])
    X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                        Y,
                                                        test_size=config["test_size"],
                                                        random_state=config["random_seed"])
    logger.info("Dataset loaded from SQLite database, validated and split.")
    
    print('Building model...')
    model_grid = build_model(config)
    logger.info("Model built.")
    
    print('Training model...')
    model_grid.fit(X_train, Y_train)
    logger.info("Model trained; best model: %s", str(model_grid.best_params_))
    
    print('Evaluating model...')
    report = evaluate_model(model_grid, X_test, Y_test, config["target_columns"])
    save_evaluation(report, config["evaluation_filepath"])
    logger.info("Model evaluated and report saved to %s.", config['evaluation_filepath'])

    print(f"Saving best model to {config['model_filepath']}...")
    save_pipeline(model_grid.best_estimator_, config["model_filepath"])
    logger.info("Model trained and saved to %s.", config['model_filepath'])

    print('Model trained and saved!')
