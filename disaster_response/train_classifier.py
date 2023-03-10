#import sys
import argparse
#import numpy as np
import pandas as pd

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
    """Perform the NLP:
    
    - Clean
    - Normalize (to lower)
    - Tokenize
    - Lemmatize
    
    Args:
        text (string): message
    
    Returns:
        clean_tokens (list): list of processed lemmas
    """
    detected_urls = re.findall(URL_REGEX, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(config):
    """..."""
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('onehot', OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore"))])

    # CountVectorizer and TfidfTransformer have some issues
    # with the size of the arrays, and they require some reshaping
    # as done here with FunctionTransformer
    nlp_transformer = Pipeline([
        #('reshape', FunctionTransformer(np.reshape, kw_args={"newshape": -1})),
        ("squeeze", FunctionTransformer(lambda x: x.squeeze())),
        ('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None)),
        ('tfidf', TfidfTransformer()),
        ("toarray", FunctionTransformer(lambda x: x.toarray())),
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
            ("processor", processor),
            ("classifier", MultiOutputClassifier(
                RandomForestClassifier(**config["random_forest_parameters"]))),
            #("classifier", RandomForestClassifier(**config["random_forest_parameters"])),
        ]
    )
    
    param_grid = {
    'classifier__estimator__n_estimators': config["grid_search"]["hyperparameters"]["n_estimators"], # 100, 200
    'classifier__estimator__min_samples_split': config["grid_search"]["hyperparameters"]["min_samples_split"] # 2, 3
    }
    
    grid = GridSearchCV(pipe,
                        param_grid=param_grid,
                        #scoring=config["grid_search"]["scoring"], # "f1"
                        cv=config["grid_search"]["cv"]) # 3
    
    return grid


def evaluate_model(model, X_test, Y_test, category_names):
    """..."""
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
    """..."""
    save_evaluation_report(report, evaluation_filepath)


def save_model(model, model_filepath):
    """..."""
    print(f"Saving best model: \n{model.best_estimator_}")
    save_model(model.best_estimator_, model_filepath)


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
    logger.info("Model trained.")
    
    print('Evaluating model...')
    report = evaluate_model(model_grid, X_test, Y_test, config["target_columns"])
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
