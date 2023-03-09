# Disaster Response Pipeline

This file contains a summary of the original project assignment instructions.

## Requirements

ETL Pipeline: `process_data.py`

- Load 2 datasets
- Join/merge 2 datasets
- Clean: encode categories, drop duplicates, use clear names, etc.
- Store to SQLite

ML Pipeline: `train_classifier.py`

- Load data from SQLite
- Segregation
- Build text processing pipeline
- Train with `GridSearchCV`
- Evaluate and display results
- Export pickle

Flask web app: `app/run.py`

- Modify file paths
- Add two Plotly visualizations based on the data extracted from SQLite 

Code Quality

- Git is used
- Documentation
- Modular and clean

## Tips

- Notebooks: not compulsory, but encouraged to complete them, as they help.
- After completing the notebooks, transfer the code to the scripts.
- Web app: be can re-use and modify the provided code.

## Rubric Summary

- Github and Code Quality
  - Github repo with at least 3 commits
  - Correct README: summary, how to run, explanation of the files, etc.
  - Code with proper docstrings.
  - Code: easy to understand structure, PEP8.
- ETL Pipeline
  - Script does all what's required and runs without errors.
- ML Pipeline
  - Script does all what's required and runs without errors.
  - Use a custom `tokenize()` function: case normalize, lemmatize, tokenize
  - Text vectorized as TFIDF.
  - Multi-output classification: 36 categories.
  - `GridSearchCV` used; training only with `train` split.
  - Evaluation display for `test` split: F1 score, precision and recall.
- Deployment
  - Script does all what's required and runs without errors.
  - At least 2 visualizations with the data from SQLite.
  - When a user inputs a message, classification results are returned for 36 categories.

## Original Execution Instructions
 
> 1. Run the following commands in the project's root directory to set up your database and model.
> 
>     - To run ETL pipeline that cleans data and stores in database
>         `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
>     - To run ML pipeline that trains classifier and saves
>         `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
> 
> 2. Go to `app` directory: `cd app`
> 
> 3. Run your web app: `python run.py`
> 
> 4. Click the `PREVIEW` button to open the homepage
